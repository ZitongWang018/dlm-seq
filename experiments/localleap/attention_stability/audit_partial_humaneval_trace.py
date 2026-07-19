#!/usr/bin/env python3
"""Health-only execution audit for a completed prefix of a HumanEval trace."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humaneval_execution import check_correctness


VERSION = "humaneval_partial_trace_execution_health_v1"


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def index(rows, identity, label):
    output = {}
    for row in rows:
        task_id = identity(row)
        if not task_id or task_id in output:
            raise ValueError(f"missing or duplicate {label} id: {task_id!r}")
        output[task_id] = row
    return output


def build_check_program(generation, doc):
    return (
        doc["prompt"]
        + generation
        + "\n"
        + doc["test"]
        + f"\ncheck({doc['entry_point']})\n"
    )


def audit(trace_rows, sample_rows, timeout):
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
    for task_id, trace in traces.items():
        sample = samples[task_id]
        if trace.get("prompt_hash") != sample.get("prompt_hash"):
            raise ValueError(f"prompt hash mismatch: {task_id}")
        if trace.get("prompt_text") != sample["doc"]["prompt"]:
            raise ValueError(f"prompt text mismatch: {task_id}")
        generation = trace["decoded_generation"]
        result = check_correctness(
            build_check_program(generation, sample["doc"]),
            timeout,
            task_id,
            int(trace["absolute_index"]),
        )
        records.append(
            {
                "absolute_index": int(trace["absolute_index"]),
                "task_id": task_id,
                "prompt_hash": trace["prompt_hash"],
                "target_hash": sample["target_hash"],
                "decoded_generation": generation,
                "correct": bool(result["passed"]),
                "execution_result": result["result"],
                "nfe": int(trace["nfe"]),
                "residual_mask_count": generation.count("[MASK]")
                + generation.count("<|mask|>"),
                "evaluator_version": VERSION,
                "uses_hidden_tests": True,
                "health_only_dev_prefix": True,
            }
        )
    records.sort(key=lambda row: row["absolute_index"])
    summary = {
        "evaluator_version": VERSION,
        "total": len(records),
        "correct": sum(row["correct"] for row in records),
        "accuracy": sum(row["correct"] for row in records) / len(records),
        "nfe_total": sum(row["nfe"] for row in records),
        "nfe_min": min(row["nfe"] for row in records),
        "nfe_max": max(row["nfe"] for row in records),
        "residual_mask_count": sum(row["residual_mask_count"] for row in records),
        "duplicate_ids": 0,
        "unknown_ids": 0,
        "prompt_hash_mismatches": 0,
        "uses_hidden_tests": True,
        "health_only_dev_prefix": True,
    }
    return records, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    records, summary = audit(read_jsonl(args.trace), read_jsonl(args.samples), args.timeout)
    with (output / "audit_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
