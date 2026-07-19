#!/usr/bin/env python3
"""Versioned MBPP evaluator using only current-task prompt assertions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

from differential_selector import (
    evaluate_public_candidate,
    extract_python_code,
    prompt_assertions,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humaneval_execution import check_correctness


EVALUATOR_VERSION = "mbpp_prompt_assertion_execution_v2"


def read_records(path: Path, identity_key: str) -> Dict[str, dict]:
    indexed = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            identity = row.get(identity_key)
            if identity is None:
                raise ValueError(f"missing {identity_key} at {path}:{line_number}")
            identity = str(identity)
            if identity in indexed:
                raise ValueError(f"duplicate identity {identity!r} in {path}")
            indexed[identity] = row
    if not indexed:
        raise ValueError(f"no records in {path}")
    return indexed


def read_lm_eval_samples(path: Path) -> Dict[str, dict]:
    """Index the unmodified lm-eval sample log by the stable MBPP task id."""
    indexed = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            doc = row.get("doc") or {}
            identity = doc.get("task_id")
            if identity is None:
                raise ValueError(f"missing doc.task_id at {path}:{line_number}")
            identity = str(identity)
            if identity in indexed:
                raise ValueError(f"duplicate identity {identity!r} in {path}")
            if row.get("filter") not in (None, "none"):
                raise ValueError(
                    f"unexpected sample filter {row.get('filter')!r} for {identity}"
                )
            indexed[identity] = row
    if not indexed:
        raise ValueError(f"no samples in {path}")
    return indexed


def trace_from_sample(sample: dict, task_record: dict) -> dict:
    """Build the minimal trace view without changing baseline generation."""
    identity = str((sample.get("doc") or {}).get("task_id"))
    arguments = sample.get("arguments") or {}
    gen_args = arguments.get("gen_args_0") or {}
    prompt = gen_args.get("arg_0")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"missing generation prompt for {identity}")
    responses = sample.get("resps") or []
    generation = responses[0][0] if responses and responses[0] else None
    if not isinstance(generation, str):
        raise ValueError(f"missing generated response for {identity}")
    if generation != task_record.get("decoded_generation"):
        raise ValueError(f"generation mismatch for {identity}")
    if sample.get("prompt_hash") != task_record.get("prompt_hash"):
        raise ValueError(f"sample/task prompt hash mismatch for {identity}")
    if sample.get("target_hash") != task_record.get("target_hash"):
        raise ValueError(f"sample/task target hash mismatch for {identity}")
    return {
        "absolute_index": int(task_record["absolute_index"]),
        "task_id": identity,
        "prompt_hash": sample["prompt_hash"],
        "prompt_text": prompt,
        "entry_point": None,
        "decoded_generation": generation,
        "raw_gold": task_record.get("raw_gold"),
        "nfe": int(task_record["nfe"]),
        "seed": task_record.get("seed"),
        "generation_settings": task_record.get("generation_settings"),
    }


def independent_assertion_execution(
    generation: str,
    prompt: str,
    entry_point: str | None,
    task_id: str,
    timeout: float,
) -> dict:
    assertions = prompt_assertions(prompt, entry_point)
    program = extract_python_code(generation)
    if assertions:
        program += "\n\n" + "\n".join(f"assert {item}" for item in assertions)
    result = check_correctness(program, timeout, task_id, 0)
    return {
        "assertion_count": len(assertions),
        "passed": bool(assertions) and bool(result["passed"]),
        "result": result["result"],
    }


def audit_record(trace: dict, task_record: dict, timeout: float = 1.5) -> dict:
    task_id = str(trace["task_id"])
    if str(task_record["stable_task_id"]) != task_id:
        raise ValueError(f"stable identity mismatch for {task_id}")
    if trace["prompt_hash"] != task_record["prompt_hash"]:
        raise ValueError(f"prompt hash mismatch for {task_id}")
    prompt = trace["prompt_text"]
    generation = trace["decoded_generation"]
    entry_point = trace.get("entry_point")
    primary = evaluate_public_candidate(generation, prompt, entry_point)
    primary_correct = bool(
        primary["visible_check_count"]
        and primary["visible_checks_passed"] == primary["visible_check_count"]
    )
    crosscheck = independent_assertion_execution(
        generation, prompt, entry_point, task_id, timeout
    )
    crosscheck_match = (
        int(primary["visible_check_count"]) == int(crosscheck["assertion_count"])
        and primary_correct == bool(crosscheck["passed"])
    )
    raw_gold = trace.get("raw_gold")
    return {
        "absolute_index": int(trace["absolute_index"]),
        "stable_task_id": task_id,
        "task_id": task_id,
        "prompt_hash": trace["prompt_hash"],
        "target_hash": task_record["target_hash"],
        "raw_gold": raw_gold,
        "normalized_gold": raw_gold,
        "decoded_generation": generation,
        "extracted_prediction": extract_python_code(generation),
        "correct": primary_correct,
        "nfe": int(trace["nfe"]),
        "evaluator_version": EVALUATOR_VERSION,
        "seed": trace.get("seed"),
        "generation_settings": trace.get("generation_settings"),
        "assertion_diagnostics": {
            "visible_assertion_count": int(primary["visible_assertion_count"]),
            "visible_check_count": int(primary["visible_check_count"]),
            "visible_checks_passed": int(primary["visible_checks_passed"]),
            "compile_valid": bool(primary["compile_valid"]),
            "independent_result": crosscheck["result"],
            "independent_passed": bool(crosscheck["passed"]),
            "crosscheck_match": crosscheck_match,
            "uses_hidden_tests": False,
            "uses_reference_solution": False,
        },
        "residual_mask_count": generation.count("[MASK]")
        + generation.count("<|mask|>"),
    }


def audit_files(
    trace_path: Path | None,
    samples_path: Path | None,
    task_path: Path,
    timeout: float,
) -> Tuple[list, dict]:
    tasks = read_records(task_path, "stable_task_id")
    if (trace_path is None) == (samples_path is None):
        raise ValueError("provide exactly one of trace_path or samples_path")
    if trace_path is not None:
        traces = read_records(trace_path, "task_id")
        source_path = trace_path
        source_kind = "dependency_trace"
    else:
        samples = read_lm_eval_samples(samples_path)
        missing_sample_tasks = sorted(set(samples) - set(tasks))
        missing_samples = sorted(set(tasks) - set(samples))
        if missing_sample_tasks or missing_samples:
            raise ValueError(
                f"identity mismatch missing_task_records={missing_sample_tasks[:5]} "
                f"missing_samples={missing_samples[:5]}"
            )
        traces = {
            identity: trace_from_sample(sample, tasks[identity])
            for identity, sample in samples.items()
        }
        source_path = samples_path
        source_kind = "lm_eval_samples"
    missing_task_records = sorted(set(traces) - set(tasks))
    missing_traces = sorted(set(tasks) - set(traces))
    if missing_task_records or missing_traces:
        raise ValueError(
            f"identity mismatch missing_task_records={missing_task_records[:5]} "
            f"missing_traces={missing_traces[:5]}"
        )
    records = [audit_record(traces[key], tasks[key], timeout) for key in traces]
    records.sort(key=lambda row: row["absolute_index"])
    crosscheck_mismatches = [
        row["stable_task_id"]
        for row in records
        if not row["assertion_diagnostics"]["crosscheck_match"]
    ]
    summary = {
        "evaluator_version": EVALUATOR_VERSION,
        "total": len(records),
        "correct": sum(bool(row["correct"]) for row in records),
        "accuracy": sum(bool(row["correct"]) for row in records) / len(records),
        "duplicate_ids": 0,
        "missing_ids": 0,
        "prompt_hash_mismatches": 0,
        "target_hash_mismatches": 0,
        "crosscheck_mismatch_ids": crosscheck_mismatches,
        "all_crosschecks_pass": not crosscheck_mismatches,
        "nfe_min": min(row["nfe"] for row in records),
        "nfe_max": max(row["nfe"] for row in records),
        "nfe_total": sum(row["nfe"] for row in records),
        "residual_mask_count": sum(row["residual_mask_count"] for row in records),
        "generation_source_kind": source_kind,
        "generation_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "task_records_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace", type=Path)
    source.add_argument("--samples", type=Path)
    parser.add_argument("--task-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1.5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records, summary = audit_files(
        args.trace, args.samples, args.task_records, args.timeout
    )
    with (args.output_dir / "audit_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    raise SystemExit(0 if summary["all_crosschecks_pass"] else 2)


if __name__ == "__main__":
    main()
