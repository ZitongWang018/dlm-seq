#!/usr/bin/env python3
"""Append-safe execution health audit for a HumanEval trace prefix.

Hidden tests are used only after generation to detect metric/runtime failures.
This module is never imported by the decoder or selector and its results are
for registered health monitoring, not candidate selection or tuning.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from audit_partial_gsm8k_trace import load_append_only_jsonl, wilson_interval

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humaneval_execution import check_correctness


VERSION = "humaneval_partial_trace_execution_health_v3"


def index(rows, identity, label):
    output = {}
    for row in rows:
        task_id = identity(row)
        if not task_id or task_id in output:
            raise ValueError(f"missing or duplicate {label} id: {task_id!r}")
        output[task_id] = row
    return output


def build_check_program(generation, sample, sanitize_fn):
    """Mirror postprocess_code.py before invoking the same executor."""

    doc = sample["doc"]
    extracted = generation.split("```python\n", 1)[-1].split("```")[0]
    prediction = sanitize_fn(
        doc["prompt"] + "\n" + extracted,
        doc["entry_point"],
    )
    return prediction + "\n" + sample["target"]


def _valid_nfe(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and value > 0
    )


def audit(
    trace_rows,
    sample_rows,
    timeout,
    sanitize_fn,
    *,
    expected_total=None,
    trailing_partial=0,
):
    traces = index(trace_rows, lambda row: str(row.get("task_id") or ""), "trace")
    samples = index(
        sample_rows,
        lambda row: str((row.get("doc") or {}).get("task_id") or ""),
        "sample",
    )
    unknown = sorted(set(traces) - set(samples))
    if unknown:
        raise ValueError(f"trace ids absent from samples: {unknown[:5]}")

    records = []
    invalid_ids = 0
    absolute_ids = []
    invalid_nfe = 0
    non_null_correct = 0
    for task_id, trace in traces.items():
        sample = samples[task_id]
        if trace.get("prompt_hash") != sample.get("prompt_hash"):
            raise ValueError(f"prompt hash mismatch: {task_id}")
        if trace.get("prompt_text") != sample["doc"]["prompt"]:
            raise ValueError(f"prompt text mismatch: {task_id}")
        if not sample.get("target_hash"):
            raise ValueError(f"missing target hash: {task_id}")

        absolute_index = trace.get("absolute_index")
        if isinstance(absolute_index, int) and not isinstance(absolute_index, bool):
            absolute_ids.append(absolute_index)
        else:
            invalid_ids += 1
            absolute_index = -1
        nfe = trace.get("nfe")
        invalid_nfe += not _valid_nfe(nfe)
        non_null_correct += trace.get("correct") is not None

        generation = trace["decoded_generation"]
        result = check_correctness(
            build_check_program(generation, sample, sanitize_fn),
            timeout,
            task_id,
            int(absolute_index),
        )
        records.append(
            {
                "absolute_index": int(absolute_index),
                "task_id": task_id,
                "prompt_hash": trace["prompt_hash"],
                "target_hash": sample["target_hash"],
                "decoded_generation": generation,
                "correct": bool(result["passed"]),
                "execution_result": result["result"],
                "nfe": int(nfe) if _valid_nfe(nfe) else nfe,
                "residual_mask_count": generation.count("[MASK]")
                + generation.count("<|mask|>"),
                "evaluator_version": VERSION,
                "uses_hidden_tests": True,
                "health_only_dev_prefix": True,
            }
        )
    records.sort(key=lambda row: row["absolute_index"])

    duplicate_absolute_ids = len(absolute_ids) - len(set(absolute_ids))
    min_id = min(absolute_ids) if absolute_ids else None
    max_id = max(absolute_ids) if absolute_ids else None
    missing_prefix_ids = (
        sorted(set(range(max_id + 1)) - set(absolute_ids)) if max_id is not None else []
    )
    out_of_range_ids = (
        sorted(value for value in absolute_ids if value < 0 or value >= expected_total)
        if expected_total is not None
        else sorted(value for value in absolute_ids if value < 0)
    )
    correct = sum(row["correct"] for row in records)
    residual_masks = sum(row["residual_mask_count"] for row in records)
    valid_nfes = [float(row["nfe"]) for row in records if _valid_nfe(row["nfe"])]
    low, high = wilson_interval(correct, len(records))

    anomalies = []
    if invalid_ids:
        anomalies.append("invalid_absolute_ids")
    if duplicate_absolute_ids:
        anomalies.append("duplicate_absolute_ids")
    if missing_prefix_ids:
        anomalies.append("missing_prefix_ids")
    if out_of_range_ids:
        anomalies.append("out_of_range_ids")
    if invalid_nfe:
        anomalies.append("invalid_or_missing_nfe")
    if residual_masks:
        anomalies.append("residual_masks")
    if non_null_correct:
        anomalies.append("generation_trace_contains_non_null_correct")

    summary = {
        "evaluator_version": VERSION,
        "records": len(records),
        "expected_total": expected_total,
        "complete": expected_total is not None and len(records) == expected_total,
        "correct": correct,
        "accuracy": correct / len(records) if records else None,
        "wilson_95_low": low if records else None,
        "wilson_95_high": high if records else None,
        "nfe_count": len(valid_nfes),
        "nfe_total": sum(valid_nfes),
        "nfe_min": min(valid_nfes) if valid_nfes else None,
        "nfe_max": max(valid_nfes) if valid_nfes else None,
        "invalid_or_missing_nfe": invalid_nfe,
        "residual_mask_count": residual_masks,
        "duplicate_absolute_ids": duplicate_absolute_ids,
        "invalid_ids": invalid_ids,
        "min_id": min_id,
        "max_id": max_id,
        "missing_prefix_ids": missing_prefix_ids,
        "out_of_range_ids": out_of_range_ids,
        "non_null_correct": non_null_correct,
        "trailing_partial_lines": trailing_partial,
        "unknown_ids": 0,
        "prompt_hash_mismatches": 0,
        "uses_hidden_tests": True,
        "health_only_dev_prefix": True,
        "post_generation_only": True,
        "for_candidate_selection": False,
        "anomalies": anomalies,
        "pass": not anomalies,
    }
    return records, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--expected-total", type=int)
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args()
    if not args.summary_only and not args.output_dir:
        parser.error("--output-dir is required unless --summary-only is used")

    trace_rows, trailing_partial = load_append_only_jsonl(Path(args.trace))
    sample_rows, sample_partial = load_append_only_jsonl(Path(args.samples))
    if sample_partial:
        raise ValueError("sample log has an incomplete trailing record")
    sys.path.insert(0, str(Path(args.runtime_root).resolve()))
    from sanitize import sanitize

    records, summary = audit(
        trace_rows,
        sample_rows,
        args.timeout,
        sanitize,
        expected_total=args.expected_total,
        trailing_partial=trailing_partial,
    )
    if not args.summary_only:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=False)
        with (output / "audit_records.jsonl").open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        (output / "audit_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, sort_keys=True))
    raise SystemExit(0 if summary["pass"] else 2)


if __name__ == "__main__":
    main()
