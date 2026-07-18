#!/usr/bin/env python3
"""Independently verify guarded replay selection and inherited correctness."""

import argparse
import json
from pathlib import Path


VERSION = "public_guard_replay_crosscheck_v1"


def stable_id(record):
    value = record.get("stable_task_id", record.get("task_id"))
    if value is None:
        raise ValueError("missing stable task id")
    return str(value)


def index(records, name):
    output = {}
    for record in records:
        task_id = stable_id(record)
        if task_id in output:
            raise ValueError(f"duplicate id {task_id} in {name}")
        output[task_id] = record
    return output


def verify(method_records, parent_records, baseline_records):
    method = index(method_records, "method")
    parent = index(parent_records, "parent")
    baseline = index(baseline_records, "baseline")
    if set(method) != set(parent) or set(method) != set(baseline):
        raise ValueError("method/parent/baseline ids do not align")
    errors = []
    selected_baseline = 0
    expected_correct = 0
    for task_id in method:
        current = method[task_id]
        left = parent[task_id]
        right = baseline[task_id]
        for field in ("prompt_hash", "target_hash"):
            if current.get(field) != left.get(field) or current.get(field) != right.get(field):
                errors.append(f"{task_id}:{field}")
        diagnostics = (current.get("decode_diagnostics") or {}).get(
            "public_example_guard"
        ) or {}
        passes = diagnostics.get("visible_examples_passed") or {}
        chose_baseline = diagnostics.get("selected_name") == "baseline"
        rule_chose_baseline = (
            int(passes.get("baseline", 0)) > int(passes.get("parent", 0))
        )
        if chose_baseline != rule_chose_baseline:
            errors.append(f"{task_id}:selection_rule")
        selected = right if chose_baseline else left
        selected_baseline += int(chose_baseline)
        expected_correct += int(bool(selected["correct"]))
        if current["decoded_generation"] != selected["decoded_generation"]:
            errors.append(f"{task_id}:decoded_generation")
        if bool(current["correct"]) != bool(selected["correct"]):
            errors.append(f"{task_id}:correct")
    return {
        "evaluator_version": VERSION,
        "total": len(method),
        "selected_baseline_count": selected_baseline,
        "independently_recomputed_correct": expected_correct,
        "recorded_method_correct": sum(
            int(bool(record["correct"])) for record in method.values()
        ),
        "error_count": len(errors),
        "errors": errors,
        "all_checks_pass": not errors,
    }


def read(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method_records")
    parser.add_argument("parent_records")
    parser.add_argument("baseline_records")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    summary = verify(
        read(args.method_records), read(args.parent_records), read(args.baseline_records)
    )
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["all_checks_pass"] else 2)


if __name__ == "__main__":
    main()
