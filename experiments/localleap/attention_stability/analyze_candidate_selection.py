#!/usr/bin/env python3
"""Summarize a trajectory selector against independently executed candidates."""

import argparse
import json
from pathlib import Path


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--fast-cleaned", required=True)
    parser.add_argument("--accuracy-cleaned", required=True)
    parser.add_argument("--output")
    parser.add_argument("--records-output")
    parser.add_argument("--status", default="PRELIMINARY_DEV_ONLY")
    args = parser.parse_args()

    trace = {row["task_id"]: row for row in rows(args.trace)}
    fast = {row["task_id"]: bool(row["pass_at_1"]) for row in rows(args.fast_cleaned)}
    accuracy = {
        row["task_id"]: bool(row["pass_at_1"])
        for row in rows(args.accuracy_cleaned)
    }
    if len(trace) != len(fast) or len(trace) != len(accuracy):
        raise SystemExit("record counts differ")
    if set(trace) != set(fast) or set(trace) != set(accuracy):
        raise SystemExit("task IDs do not align")

    selected_names = {
        task_id: row["decode_diagnostics"]["selected_name"]
        for task_id, row in trace.items()
    }
    if any(name not in {"fast", "accuracy"} for name in selected_names.values()):
        raise SystemExit("unexpected selected trajectory")
    selected = {
        task_id: accuracy[task_id]
        if selected_names[task_id] == "accuracy"
        else fast[task_id]
        for task_id in trace
    }
    records = []
    for task_id in sorted(trace, key=lambda value: int(value.split("/")[-1])):
        diagnostics = trace[task_id]["decode_diagnostics"]
        verification = diagnostics.get("shared_skeleton_verification") or {}
        scores = verification.get("candidate_scores") or {}
        records.append({
            "task_id": task_id,
            "selected_name": selected_names[task_id],
            "fast_correct": fast[task_id],
            "accuracy_correct": accuracy[task_id],
            "selected_correct": selected[task_id],
            "fast_score": scores.get("fast"),
            "accuracy_score": scores.get("accuracy"),
            "score_margin_accuracy_minus_fast": (
                scores.get("accuracy") - scores.get("fast")
                if scores.get("accuracy") is not None
                and scores.get("fast") is not None
                else None
            ),
            "disagreement_token_count": verification.get(
                "disagreement_token_count"
            ),
            "block_evidence_margin": diagnostics.get("block_evidence_margin"),
            "fast_invalidations": diagnostics.get("candidate_summaries", {})
            .get("fast", {})
            .get("response_invalidations"),
            "accuracy_invalidations": diagnostics.get("candidate_summaries", {})
            .get("accuracy", {})
            .get("response_invalidations"),
        })
    summary = {
        "status": args.status,
        "total": len(trace),
        "fast_correct": sum(fast.values()),
        "accuracy_correct": sum(accuracy.values()),
        "selected_correct": sum(selected.values()),
        "oracle_correct": sum(
            fast[task_id] or accuracy[task_id] for task_id in trace
        ),
        "method_only": sum(
            selected[task_id] and not fast[task_id] for task_id in trace
        ),
        "fast_only": sum(
            fast[task_id] and not selected[task_id] for task_id in trace
        ),
        "selected_accuracy_count": sum(
            name == "accuracy" for name in selected_names.values()
        ),
        "duplicate_ids": 0,
        "missing_ids": 0,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    if args.records_output:
        with Path(args.records_output).open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
