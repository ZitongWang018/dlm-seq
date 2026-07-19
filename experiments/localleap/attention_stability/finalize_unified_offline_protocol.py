#!/usr/bin/env python3
"""Finalize one task-unified algorithm after strict offline protocol checks."""

import argparse
import json
from pathlib import Path


VERSION = "localleap_unified_offline_protocol_finalizer_v1"
TASKS = ("humaneval", "math500", "gsm8k", "mbpp")
EXPECTED_TOTALS = {"humaneval": 164, "math500": 500, "gsm8k": 1319, "mbpp": 500}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_pair(row, task, arm, allow_frozen_algorithm_source_change=False):
    if row.get("total") != EXPECTED_TOTALS[task]:
        raise ValueError(f"{task}/{arm}: total={row.get('total')}")
    for key in ("prompt_hash_mismatches", "target_hash_mismatches", "duplicate_or_missing_ids"):
        if row.get(key) != 0:
            raise ValueError(f"{task}/{arm}: {key}={row.get(key)}")
    if row.get("evaluator_version_mismatches") != 0:
        raise ValueError(f"{task}/{arm}: evaluator versions differ")
    source_mismatches = row.get("source_hash_mismatches")
    if source_mismatches != 0 and not allow_frozen_algorithm_source_change:
        raise ValueError(f"{task}/{arm}: source_hash_mismatches={source_mismatches}")
    if row.get("baseline_total_nfe", 0) <= 0 or row.get("method_total_nfe", 0) <= 0:
        raise ValueError(f"{task}/{arm}: invalid NFE")


def summarize(rows):
    baseline_correct = sum(row["baseline_correct"] for row in rows.values())
    method_correct = sum(row["method_correct"] for row in rows.values())
    baseline_nfe = sum(row["baseline_total_nfe"] for row in rows.values())
    method_nfe = sum(row["method_total_nfe"] for row in rows.values())
    baseline_wall = sum(row.get("baseline_wall_seconds") or 0 for row in rows.values())
    method_wall = sum(row.get("method_wall_seconds") or 0 for row in rows.values())
    return {
        "per_task_accuracy_nonregression": all(
            row["method_correct"] >= row["baseline_correct"] for row in rows.values()
        ),
        "strict_aggregate_accuracy_gain": method_correct > baseline_correct,
        "baseline_correct": baseline_correct,
        "method_correct": method_correct,
        "accuracy_gain_count": method_correct - baseline_correct,
        "baseline_total_nfe": baseline_nfe,
        "method_total_nfe": method_nfe,
        "nfe_change_fraction": (method_nfe - baseline_nfe) / baseline_nfe,
        "baseline_wall_seconds_sum": baseline_wall,
        "method_wall_seconds_sum": method_wall,
        "wall_change_fraction": (
            (method_wall - baseline_wall) / baseline_wall if baseline_wall else None
        ),
        "tasks": rows,
    }


def finalize(spec):
    if not spec.get("candidate_profile"):
        raise ValueError("missing candidate profile")
    offline = load(spec["offline_manifest_verification"])
    if not offline.get("pass"):
        raise ValueError("offline artifact verification failed")
    candidate_rows = {}
    fast_rows = {}
    for task in TASKS:
        task_spec = spec["tasks"][task]
        candidate = load(task_spec["candidate_pair"])
        fast = load(task_spec["fast_pair"])
        validate_pair(candidate, task, "candidate", allow_frozen_algorithm_source_change=True)
        validate_pair(fast, task, "fast")
        candidate_inputs = load(task_spec["candidate_input_compare"])
        fast_inputs = load(task_spec["fast_input_compare"])
        if not candidate_inputs.get("all_equal"):
            raise ValueError(f"{task}/candidate: actual model inputs differ")
        if not fast_inputs.get("all_equal"):
            raise ValueError(f"{task}/fast: actual model inputs differ")
        candidate_config = load(task_spec["candidate_config_compare"])
        fast_config = load(task_spec["fast_config_compare"])
        if not candidate_config.get("all_equal_core"):
            raise ValueError(f"{task}/candidate: core configuration differs")
        if not fast_config.get("all_equal_core"):
            raise ValueError(f"{task}/fast: core configuration differs")
        if candidate_config.get("candidate_profile") != spec["candidate_profile"]:
            raise ValueError(f"{task}: candidate profile is not task-unified")
        leakage = load(task_spec["candidate_leakage"])
        if not leakage.get("pass") or leakage.get("evaluator_version") != "generation_information_leakage_audit_v2":
            raise ValueError(f"{task}: leakage v2 failed")
        candidate_rows[task] = candidate
        fast_rows[task] = fast
    candidate_summary = summarize(candidate_rows)
    fast_summary = summarize(fast_rows)
    if not candidate_summary["per_task_accuracy_nonregression"]:
        raise ValueError("unified candidate regressed on at least one benchmark")
    if not candidate_summary["strict_aggregate_accuracy_gain"]:
        raise ValueError("unified candidate did not beat the audited baseline in aggregate")
    return {
        "schema": VERSION,
        "single_algorithm": True,
        "task_specific_routing": False,
        "selected_profile": spec["candidate_profile"],
        "selected_family": spec["candidate_family"],
        "selection_reason": spec["selection_reason"],
        "fast_arm_is_comparator_only": True,
        "paper_reported_results_are_not_local_arms": True,
        "leakage_audit_version": "generation_information_leakage_audit_v2",
        "offline_artifacts_verified": True,
        "candidate": candidate_summary,
        "candidate_wall_timing_basis": (
            "fresh same-task simultaneous baseline/candidate runs with task-balanced GPU assignment"
        ),
        "fast_comparator": fast_summary,
        "fast_wall_timing_is_diagnostic_only": True,
        "benchmark_reuse_status": (
            "confirmatory_on_reused_public_benchmarks_not_an_independent_clean_holdout"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = finalize(load(args.spec))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
