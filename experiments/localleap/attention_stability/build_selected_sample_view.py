#!/usr/bin/env python3
"""Build an lm-eval sample file from independently selected audit records."""

import argparse
import copy
import json
from pathlib import Path


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_view(samples, selected_records):
    sample_by_id = {row["doc"]["task_id"]: row for row in samples}
    selected_by_id = {row["task_id"]: row for row in selected_records}
    if len(sample_by_id) != len(samples) or len(selected_by_id) != len(selected_records):
        raise ValueError("duplicate task ids")
    if set(sample_by_id) != set(selected_by_id):
        raise ValueError("sample and selected task ids do not align")
    output = []
    for task_id in sorted(sample_by_id, key=lambda value: int(value.split("/")[-1])):
        sample = copy.deepcopy(sample_by_id[task_id])
        sample["resps"][0][0] = selected_by_id[task_id]["decoded_generation"]
        output.append(sample)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--selected-records", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output_path = Path(args.output)
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite {output_path}")
    rows = build_view(read_jsonl(args.samples), read_jsonl(args.selected_records))
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
