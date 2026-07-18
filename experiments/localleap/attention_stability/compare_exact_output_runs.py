#!/usr/bin/env python3
"""Audit an exact-output acceleration against its frozen accuracy parent."""

import argparse
import hashlib
import json
from pathlib import Path


EVALUATOR_VERSION = "exact_output_equivalence_v1"
REQUIRED_FIELDS = {
    "absolute_index",
    "task_id",
    "prompt_hash",
    "raw_gold",
    "normalized_gold",
    "decoded_generation",
    "correct",
    "nfe",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(path):
    records = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing_fields = sorted(REQUIRED_FIELDS - set(record))
            if missing_fields:
                raise ValueError(
                    f"missing required fields {missing_fields} at {path}:{line_number}"
                )
            task_id = record.get("task_id")
            if not task_id:
                raise ValueError(f"missing task_id at {path}:{line_number}")
            if task_id in records:
                raise ValueError(f"duplicate task_id {task_id!r} in {path}")
            records[task_id] = record
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def compare_records(parent_records, candidate_records):
    parent_ids = set(parent_records)
    candidate_ids = set(candidate_records)
    missing_ids = sorted(parent_ids - candidate_ids)
    extra_ids = sorted(candidate_ids - parent_ids)
    comparisons = []
    for task_id in sorted(parent_ids & candidate_ids):
        parent = parent_records[task_id]
        candidate = candidate_records[task_id]
        parent_nfe = int(parent["nfe"])
        candidate_nfe = int(candidate["nfe"])
        comparisons.append(
            {
                "task_id": task_id,
                "absolute_index": parent.get("absolute_index"),
                "prompt_hash_equal": (
                    parent.get("prompt_hash") == candidate.get("prompt_hash")
                ),
                "raw_gold_equal": parent.get("raw_gold") == candidate.get("raw_gold"),
                "normalized_gold_equal": (
                    parent.get("normalized_gold") == candidate.get("normalized_gold")
                ),
                "decoded_generation_equal": (
                    parent.get("decoded_generation")
                    == candidate.get("decoded_generation")
                ),
                "correct_equal": parent.get("correct") == candidate.get("correct"),
                "parent_nfe": parent_nfe,
                "candidate_nfe": candidate_nfe,
                "nfe_nonincrease": candidate_nfe <= parent_nfe,
            }
        )
    parent_nfe_total = sum(item["parent_nfe"] for item in comparisons)
    candidate_nfe_total = sum(item["candidate_nfe"] for item in comparisons)
    invariants = {
        "identity_alignment": not missing_ids
        and not extra_ids
        and all(
            item["prompt_hash_equal"]
            and item["raw_gold_equal"]
            and item["normalized_gold_equal"]
            for item in comparisons
        ),
        "decoded_generation_exact": all(
            item["decoded_generation_equal"] for item in comparisons
        ),
        "correctness_exact": all(item["correct_equal"] for item in comparisons),
        "per_record_nfe_nonincrease": all(
            item["nfe_nonincrease"] for item in comparisons
        ),
    }
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "parent_count": len(parent_records),
        "candidate_count": len(candidate_records),
        "paired_count": len(comparisons),
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "invariants": invariants,
        "all_invariants_pass": all(invariants.values()),
        "parent_nfe_total": parent_nfe_total,
        "candidate_nfe_total": candidate_nfe_total,
        "nfe_reduction": parent_nfe_total - candidate_nfe_total,
        "nfe_reduction_fraction": (
            (parent_nfe_total - candidate_nfe_total) / parent_nfe_total
            if parent_nfe_total
            else None
        ),
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_records")
    parser.add_argument("candidate_records")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = compare_records(
        load_records(args.parent_records), load_records(args.candidate_records)
    )
    summary["parent_records_sha256"] = _sha256(args.parent_records)
    summary["candidate_records_sha256"] = _sha256(args.candidate_records)
    comparisons = summary.pop("comparisons")
    with (output_dir / "exact_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in comparisons:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output_dir / "exact_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    raise SystemExit(0 if summary["all_invariants_pass"] else 2)


if __name__ == "__main__":
    main()
