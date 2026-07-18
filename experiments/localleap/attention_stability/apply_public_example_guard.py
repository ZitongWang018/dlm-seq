#!/usr/bin/env python3
"""Apply the preregistered public-example guard to aligned audited drafts."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from differential_selector import select_public_example_guard


EVALUATOR_VERSION = "public_example_guard_replay_v1"
REQUIRED = {
    "absolute_index",
    "task_id",
    "prompt_hash",
    "raw_gold",
    "normalized_gold",
    "decoded_generation",
    "correct",
    "nfe",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_records(path, require_prompt=False):
    records = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        required = REQUIRED | ({"prompt_text", "entry_point"} if require_prompt else set())
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"missing fields {missing} at {path}:{line_number}")
        task_id = record["task_id"]
        if task_id in records:
            raise ValueError(f"duplicate task_id {task_id!r} in {path}")
        records[task_id] = record
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def apply_guard(parent_records, baseline_records):
    if set(parent_records) != set(baseline_records):
        raise ValueError("parent and baseline task IDs do not align")
    output = []
    for task_id in sorted(
        parent_records, key=lambda value: parent_records[value]["absolute_index"]
    ):
        parent = parent_records[task_id]
        baseline = baseline_records[task_id]
        for field in ("absolute_index", "prompt_hash", "raw_gold", "normalized_gold"):
            if parent[field] != baseline[field]:
                raise ValueError(f"{field} mismatch for {task_id}")
        guard_name, diagnostics = select_public_example_guard(
            baseline["decoded_generation"],
            parent["decoded_generation"],
            parent["prompt_text"],
            parent["entry_point"],
        )
        selected = baseline if guard_name == "baseline" else parent
        record = copy.deepcopy(selected)
        record["prompt_text"] = parent["prompt_text"]
        record["entry_point"] = parent["entry_point"]
        record["nfe"] = int(parent["nfe"]) + int(baseline["nfe"])
        record["evaluator_version"] = EVALUATOR_VERSION
        record["generation_settings"] = copy.deepcopy(
            parent.get("generation_settings", {})
        )
        record["generation_settings"]["trajectory_selector"] = (
            "confirmed_bidirectional_public_guard_v11"
        )
        parent_diagnostics = copy.deepcopy(parent.get("decode_diagnostics") or {})
        parent_name = parent_diagnostics.get("selected_name", "parent")
        diagnostics["parent_name"] = parent_name
        diagnostics["final_selected_name"] = (
            "baseline" if guard_name == "baseline" else parent_name
        )
        parent_diagnostics["pre_guard_selected_name"] = parent_name
        parent_diagnostics["selected_name"] = diagnostics["final_selected_name"]
        parent_diagnostics["public_example_guard"] = diagnostics
        parent_diagnostics["candidate_nfe"] = {
            **(parent_diagnostics.get("candidate_nfe") or {}),
            "baseline": int(baseline["nfe"]),
        }
        record["decode_diagnostics"] = parent_diagnostics
        record["candidate_generations"] = {
            "parent": parent["decoded_generation"],
            "baseline": baseline["decoded_generation"],
        }
        output.append(record)
    return output


def summarize(records, parent_records, baseline_records):
    selected_baseline = sum(
        record["decode_diagnostics"]["public_example_guard"]["selected_name"]
        == "baseline"
        for record in records
    )
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "total": len(records),
        "method_correct": sum(bool(record["correct"]) for record in records),
        "parent_correct": sum(bool(record["correct"]) for record in parent_records.values()),
        "baseline_correct": sum(
            bool(record["correct"]) for record in baseline_records.values()
        ),
        "method_only_vs_parent": sum(
            bool(record["correct"])
            and not bool(parent_records[record["task_id"]]["correct"])
            for record in records
        ),
        "parent_only_vs_method": sum(
            bool(parent_records[record["task_id"]]["correct"])
            and not bool(record["correct"])
            for record in records
        ),
        "selected_baseline_count": selected_baseline,
        "uses_hidden_tests": False,
        "uses_reference_solution": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_records")
    parser.add_argument("baseline_records")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    parent = load_records(args.parent_records, require_prompt=True)
    baseline = load_records(args.baseline_records)
    records = apply_guard(parent, baseline)
    summary = summarize(records, parent, baseline)
    summary["parent_records_sha256"] = sha256(args.parent_records)
    summary["baseline_records_sha256"] = sha256(args.baseline_records)
    with (output_dir / "audit_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
