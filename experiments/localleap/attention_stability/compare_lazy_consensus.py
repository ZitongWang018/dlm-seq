#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_jsonl(path):
    records = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            task_id = record["task_id"]
            if task_id in records:
                raise ValueError(f"duplicate task id in {path}: {task_id}")
            records[task_id] = record
    return records


def compare_runs(full_audit_path, lazy_audit_path, full_trace_path, lazy_trace_path):
    full_audit = load_jsonl(full_audit_path)
    lazy_audit = load_jsonl(lazy_audit_path)
    full_trace = load_jsonl(full_trace_path)
    lazy_trace = load_jsonl(lazy_trace_path)
    ids = sorted(full_audit)
    if ids != sorted(lazy_audit) or ids != sorted(full_trace) or ids != sorted(lazy_trace):
        raise ValueError("audit/trace task ids do not align")
    if not ids:
        raise ValueError("equivalence comparison requires at least one task")

    mismatches = []
    full_nfe = 0
    lazy_nfe = 0
    baseline_skips = 0
    for task_id in ids:
        full_record = full_audit[task_id]
        lazy_record = lazy_audit[task_id]
        full_diag = full_trace[task_id]["decode_diagnostics"]
        lazy_diag = lazy_trace[task_id]["decode_diagnostics"]
        checks = {
            "prompt_hash": full_record["prompt_hash"] == lazy_record["prompt_hash"],
            "decoded_generation": full_record["decoded_generation"]
            == lazy_record["decoded_generation"],
            "correct": bool(full_record["correct"]) == bool(lazy_record["correct"]),
            "selected_name": full_diag["selected_name"] == lazy_diag["selected_name"],
        }
        failed = [name for name, matched in checks.items() if not matched]
        if failed:
            mismatches.append({"task_id": task_id, "fields": failed})
        full_nfe += int(full_trace[task_id]["nfe"])
        lazy_nfe += int(lazy_trace[task_id]["nfe"])
        baseline_skips += "baseline" not in lazy_diag["candidate_nfe"]

    return {
        "evaluator_version": "lazy_consensus_equivalence_v1",
        "total": len(ids),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "full_total_nfe": full_nfe,
        "lazy_total_nfe": lazy_nfe,
        "nfe_saved": full_nfe - lazy_nfe,
        "nfe_reduction_fraction": (full_nfe - lazy_nfe) / full_nfe,
        "baseline_skips": baseline_skips,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-audit", required=True)
    parser.add_argument("--lazy-audit", required=True)
    parser.add_argument("--full-trace", required=True)
    parser.add_argument("--lazy-trace", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = compare_runs(
        args.full_audit, args.lazy_audit, args.full_trace, args.lazy_trace
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    if (
        summary["mismatch_count"]
        or summary["lazy_total_nfe"] >= summary["full_total_nfe"]
        or summary["baseline_skips"] == 0
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
