#!/usr/bin/env python3
"""Verify replayed selection correctness with the official code executor."""

import argparse
import json
from pathlib import Path


VERSION = "selected_execution_crosscheck_v1"


def index(rows, name):
    output = {}
    for row in rows:
        task_id = row.get("task_id")
        if not task_id or task_id in output:
            raise ValueError(f"missing or duplicate task_id in {name}: {task_id!r}")
        output[task_id] = row
    return output


def verify(audit_rows, execution_rows):
    audit = index(audit_rows, "audit")
    execution = index(execution_rows, "execution")
    if set(audit) != set(execution):
        raise ValueError("audit and execution task ids do not align")
    mismatches = [
        task_id
        for task_id in audit
        if bool(audit[task_id]["correct"])
        != bool(execution[task_id]["pass_at_1"])
    ]
    return {
        "evaluator_version": VERSION,
        "total": len(audit),
        "audit_correct": sum(bool(row["correct"]) for row in audit.values()),
        "execution_correct": sum(
            bool(row["pass_at_1"]) for row in execution.values()
        ),
        "correctness_mismatch_ids": sorted(mismatches),
        "all_correctness_matches": not mismatches,
    }


def read(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_records")
    parser.add_argument("execution_records")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    summary = verify(read(args.audit_records), read(args.execution_records))
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["all_correctness_matches"] else 2)


if __name__ == "__main__":
    main()
