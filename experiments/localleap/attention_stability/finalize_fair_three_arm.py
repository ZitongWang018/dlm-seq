#!/usr/bin/env python3
"""Finalize a preregistered, task-unified three-arm LocalLeap evaluation."""

import argparse
import json
from pathlib import Path


VERSION = "localleap_fair_three_arm_finalizer_v1"
TASKS = ("humaneval", "math500", "gsm8k", "mbpp")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_pair(row, task, arm):
    required_zero = (
        "prompt_hash_mismatches",
        "target_hash_mismatches",
        "duplicate_or_missing_ids",
        "source_hash_mismatches",
    )
    for key in required_zero:
        if row.get(key) != 0:
            raise ValueError(f"{task}/{arm}: {key}={row.get(key)}")
    if not row.get("source_hashes_verified"):
        raise ValueError(f"{task}/{arm}: source hashes were not verified")
    if row["baseline_total_nfe"] <= 0 or row["method_total_nfe"] <= 0:
        raise ValueError(f"{task}/{arm}: invalid NFE totals")
    if row["total"] <= 0:
        raise ValueError(f"{task}/{arm}: empty comparison")


def arm_summary(rows, require_fast):
    baseline_correct = sum(row["baseline_correct"] for row in rows.values())
    method_correct = sum(row["method_correct"] for row in rows.values())
    baseline_nfe = sum(row["baseline_total_nfe"] for row in rows.values())
    method_nfe = sum(row["method_total_nfe"] for row in rows.values())
    baseline_wall = sum(
        row["baseline_wall_seconds"] for row in rows.values()
        if row.get("baseline_wall_seconds") is not None
    )
    method_wall = sum(
        row["method_wall_seconds"] for row in rows.values()
        if row.get("method_wall_seconds") is not None
    )
    nonregressive = all(
        row["method_correct"] >= row["baseline_correct"] for row in rows.values()
    )
    strict_accuracy_gain = method_correct > baseline_correct
    per_task_nfe_nonincrease = all(
        row["method_total_nfe"] <= row["baseline_total_nfe"] for row in rows.values()
    )
    strict_nfe_reduction = method_nfe < baseline_nfe
    wall_complete = all(
        row.get("baseline_wall_seconds") is not None
        and row.get("method_wall_seconds") is not None
        for row in rows.values()
    )
    strict_wall_reduction = wall_complete and method_wall < baseline_wall
    eligible = nonregressive and strict_accuracy_gain
    if require_fast:
        eligible = (
            eligible
            and per_task_nfe_nonincrease
            and strict_nfe_reduction
            and strict_wall_reduction
        )
    return {
        "eligible": eligible,
        "requires_fast": require_fast,
        "per_task_accuracy_nonregression": nonregressive,
        "strict_aggregate_accuracy_gain": strict_accuracy_gain,
        "per_task_nfe_nonincrease": per_task_nfe_nonincrease,
        "strict_aggregate_nfe_reduction": strict_nfe_reduction,
        "strict_aggregate_wall_reduction": strict_wall_reduction,
        "baseline_correct": baseline_correct,
        "method_correct": method_correct,
        "accuracy_gain_count": method_correct - baseline_correct,
        "baseline_total_nfe": baseline_nfe,
        "method_total_nfe": method_nfe,
        "nfe_reduction_fraction": (baseline_nfe - method_nfe) / baseline_nfe,
        "baseline_wall_seconds_sum": baseline_wall,
        "method_wall_seconds_sum": method_wall,
        "tasks": rows,
    }


def finalize(spec):
    arms = {}
    for arm in ("accuracy", "fast"):
        rows = {}
        for task in TASKS:
            row = load(spec["tasks"][task][arm])
            validate_pair(row, task, arm)
            input_compare = load(spec["tasks"][task][f"{arm}_input_compare"])
            if not input_compare.get("all_equal"):
                raise ValueError(f"{task}/{arm}: actual model inputs differ")
            rows[task] = row
        arms[arm] = arm_summary(rows, require_fast=arm == "fast")

    if arms["accuracy"]["eligible"]:
        selected = spec["accuracy_profile"]
        reason = "accuracy_arm_passed_all_task_nonregression_and_aggregate_gain"
        qualified = True
    elif arms["fast"]["eligible"]:
        selected = spec["fast_profile"]
        reason = "fast_arm_passed_accuracy_nfe_and_wall_gates"
        qualified = True
    else:
        selected = spec["fallback_profile"]
        reason = "no_new_arm_passed_preregistered_unified_gate_retain_v11_family"
        qualified = False
    return {
        "schema": VERSION,
        "single_algorithm": True,
        "task_specific_routing": False,
        "fair_protocol": {
            "steps": 128,
            "gen_length": 256,
            "block_length": 32,
            "temperature": 0.0,
            "fewshot": {
                "humaneval": 0,
                "math500": 0,
                "gsm8k": 0,
                "mbpp": 0,
            },
        },
        "selected_profile": selected,
        "selection_reason": reason,
        "new_candidate_qualified": qualified,
        "arms": arms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = finalize(load(args.spec))
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
