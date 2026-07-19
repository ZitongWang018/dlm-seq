#!/usr/bin/env python3
"""Audit and strictly compare model inputs captured inside generate_until."""

import argparse
import hashlib
import json
from pathlib import Path


VERSION = "localleap_runtime_model_input_audit_v1"
FIELDS = (
    "raw_prompt_hash",
    "model_input_text_hash",
    "model_input_token_ids_hash",
    "model_input_token_count",
    "implicit_attention_mask_hash",
    "document_hash",
    "tokenizer_call",
)


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def index(rows):
    result = {}
    for row in rows:
        if row.get("schema") != "localleap_runtime_model_input_v1":
            raise ValueError(f"unexpected runtime input schema: {row.get('schema')}")
        stable_id = str(row["stable_task_id"])
        if stable_id in result:
            raise ValueError(f"duplicate stable id: {stable_id}")
        result[stable_id] = row
    return result


def set_hash(rows):
    payload = [
        [stable_id] + [rows[stable_id][field] for field in FIELDS]
        for stable_id in sorted(rows)
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def audit(args):
    rows = index(read_jsonl(args.trace))
    absolute_indices = sorted(int(row["absolute_index"]) for row in rows.values())
    expected_indices = list(range(args.expected_records))
    result = {
        "schema": VERSION,
        "trace": str(Path(args.trace).resolve()),
        "record_count": len(rows),
        "expected_records": args.expected_records,
        "duplicate_ids": 0,
        "missing_or_extra_absolute_indices": sorted(
            set(absolute_indices).symmetric_difference(expected_indices)
        ),
        "runtime_input_set_hash": set_hash(rows),
    }
    result["pass"] = (
        result["record_count"] == args.expected_records
        and not result["missing_or_extra_absolute_indices"]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 2)


def compare(args):
    reference = index(read_jsonl(args.reference))
    candidate = index(read_jsonl(args.candidate))
    missing = sorted(set(reference) - set(candidate))
    extra = sorted(set(candidate) - set(reference))
    mismatches = []
    for stable_id in sorted(set(reference) & set(candidate)):
        changed = [
            field
            for field in FIELDS
            if reference[stable_id][field] != candidate[stable_id][field]
        ]
        if changed:
            mismatches.append({"stable_task_id": stable_id, "fields": changed})
    result = {
        "schema": VERSION,
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "missing_ids": missing,
        "extra_ids": extra,
        "mismatches": mismatches,
        "all_equal": not missing and not extra and not mismatches,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["all_equal"] else 2)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("trace")
    audit_parser.add_argument("--expected-records", type=int, required=True)
    audit_parser.add_argument("--output", required=True)
    audit_parser.set_defaults(func=audit)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("reference")
    compare_parser.add_argument("candidate")
    compare_parser.add_argument("--output", required=True)
    compare_parser.set_defaults(func=compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
