#!/usr/bin/env python3
"""Attach the exact public prompt from lm-eval samples to audit records."""

import argparse
import copy
import json
from pathlib import Path


def stable_id(record):
    value = record.get("stable_task_id", record.get("task_id"))
    if value is None:
        raise ValueError("record has no stable task id")
    return str(value)


def sample_id(sample):
    value = sample.get("doc", {}).get("task_id")
    if value is None:
        raise ValueError("sample has no doc.task_id")
    return str(value)


def sample_prompt(sample):
    arguments = sample.get("arguments", {}).get("gen_args_0", {})
    if "arg_0" in arguments:
        return arguments["arg_0"]
    if "prompt_text" in sample:
        return sample["prompt_text"]
    raise ValueError(f"sample {sample_id(sample)} has no exact generation prompt")


def enrich(records, samples):
    record_by_id = {stable_id(row): row for row in records}
    sample_by_id = {sample_id(row): row for row in samples}
    if len(record_by_id) != len(records) or len(sample_by_id) != len(samples):
        raise ValueError("duplicate stable ids")
    if set(record_by_id) != set(sample_by_id):
        raise ValueError("audit and sample stable ids do not align")
    output = []
    for task_id in sorted(
        record_by_id, key=lambda value: int(record_by_id[value]["absolute_index"])
    ):
        record = copy.deepcopy(record_by_id[task_id])
        sample = sample_by_id[task_id]
        for field in ("prompt_hash", "target_hash"):
            if sample.get(field) != record.get(field):
                raise ValueError(f"{field} mismatch for {task_id}")
        record["task_id"] = task_id
        record["prompt_text"] = sample_prompt(sample)
        record["entry_point"] = sample.get("doc", {}).get("entry_point")
        output.append(record)
    return output


def read(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_records")
    parser.add_argument("samples")
    parser.add_argument("output")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    rows = enrich(read(args.audit_records), read(args.samples))
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
