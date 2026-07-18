#!/usr/bin/env python3
"""Write a stable-id HumanEval slice for paired holdout auditing."""

import argparse
import json
from pathlib import Path


def humaneval_index(record):
    task_id = str(record.get("stable_task_id") or record.get("task_id") or "")
    prefix = "HumanEval/"
    if not task_id.startswith(prefix):
        raise ValueError(f"not a HumanEval stable id: {task_id!r}")
    return int(task_id[len(prefix):])


def slice_records(records, start, end):
    selected = [record for record in records if start <= humaneval_index(record) < end]
    ids = [humaneval_index(record) for record in selected]
    expected = list(range(start, end))
    if sorted(ids) != expected:
        raise ValueError(f"slice ids mismatch: got={sorted(ids)} expected={expected}")
    return sorted(selected, key=humaneval_index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.input).read_text().splitlines()]
    selected = slice_records(records, args.start, args.end)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in selected))


if __name__ == "__main__":
    main()
