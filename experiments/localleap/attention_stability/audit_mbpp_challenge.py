#!/usr/bin/env python3
"""Post-hoc MBPP challenge-test audit; never used for candidate selection."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from audit_mbpp_assertions import read_lm_eval_samples, read_records
from differential_selector import evaluate_public_candidate, extract_python_code

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humaneval_execution import check_correctness


VERSION = "mbpp_hidden_challenge_posthoc_v1"


def independent_challenge_execution(generation, doc, task_id, timeout):
    tests = list(doc.get("challenge_test_list") or [])
    setup = str(doc.get("test_setup_code") or "")
    program = extract_python_code(generation)
    if setup:
        program += "\n" + setup
    if tests:
        program += "\n" + "\n".join(tests)
    result = check_correctness(program, timeout, task_id, 0)
    return {
        "test_count": len(tests),
        "passed": bool(tests) and bool(result["passed"]),
        "result": result["result"],
    }


def audit_record(sample, task_record, timeout=2.0):
    doc = sample.get("doc") or {}
    task_id = str(doc.get("task_id"))
    if str(task_record["stable_task_id"]) != task_id:
        raise ValueError(f"stable identity mismatch for {task_id}")
    if sample.get("prompt_hash") != task_record.get("prompt_hash"):
        raise ValueError(f"prompt hash mismatch for {task_id}")
    if sample.get("target_hash") != task_record.get("target_hash"):
        raise ValueError(f"target hash mismatch for {task_id}")
    generation = sample["resps"][0][0]
    if generation != task_record.get("decoded_generation"):
        raise ValueError(f"generation mismatch for {task_id}")
    challenge_tests = list(doc.get("challenge_test_list") or [])
    challenge_prompt = "\n".join(challenge_tests)
    primary_generation = extract_python_code(generation)
    if doc.get("test_setup_code"):
        primary_generation += "\n" + str(doc["test_setup_code"])
    primary = evaluate_public_candidate(
        primary_generation, challenge_prompt, None
    )
    primary_correct = bool(
        challenge_tests
        and primary["visible_check_count"] == len(challenge_tests)
        and primary["visible_checks_passed"] == len(challenge_tests)
    )
    independent = independent_challenge_execution(
        generation, doc, task_id, timeout
    )
    crosscheck_match = (
        int(primary["visible_check_count"]) == len(challenge_tests)
        and int(independent["test_count"]) == len(challenge_tests)
        and primary_correct == bool(independent["passed"])
    )
    return {
        "absolute_index": int(task_record["absolute_index"]),
        "stable_task_id": task_id,
        "task_id": task_id,
        "prompt_hash": sample["prompt_hash"],
        "target_hash": sample["target_hash"],
        "raw_gold": task_record.get("raw_gold"),
        "normalized_gold": task_record.get("raw_gold"),
        "decoded_generation": generation,
        "extracted_prediction": extract_python_code(generation),
        "correct": primary_correct,
        "nfe": int(task_record["nfe"]),
        "evaluator_version": VERSION,
        "challenge_diagnostics": {
            "challenge_test_count": len(challenge_tests),
            "challenge_tests_passed": int(primary["visible_checks_passed"]),
            "compile_valid": bool(primary["compile_valid"]),
            "independent_passed": bool(independent["passed"]),
            "independent_result": independent["result"],
            "crosscheck_match": crosscheck_match,
            "selection_used_challenge_tests": False,
            "posthoc_robustness_only": True,
        },
        "residual_mask_count": generation.count("[MASK]")
        + generation.count("<|mask|>"),
    }


def audit_files(samples_path, task_path, timeout):
    samples = read_lm_eval_samples(samples_path)
    tasks = read_records(task_path, "stable_task_id")
    if set(samples) != set(tasks):
        raise ValueError(
            f"identity mismatch samples_only={sorted(set(samples)-set(tasks))[:5]} "
            f"tasks_only={sorted(set(tasks)-set(samples))[:5]}"
        )
    records = [audit_record(samples[key], tasks[key], timeout) for key in samples]
    records.sort(key=lambda row: row["absolute_index"])
    mismatch = [
        row["task_id"]
        for row in records
        if not row["challenge_diagnostics"]["crosscheck_match"]
    ]
    summary = {
        "evaluator_version": VERSION,
        "metric_role": "posthoc_hidden_challenge_robustness_not_formal_mbpp",
        "total": len(records),
        "correct": sum(row["correct"] for row in records),
        "accuracy": sum(row["correct"] for row in records) / len(records),
        "crosscheck_mismatch_ids": mismatch,
        "all_crosschecks_pass": not mismatch,
        "selection_used_challenge_tests": False,
        "nfe_min": min(row["nfe"] for row in records),
        "nfe_max": max(row["nfe"] for row in records),
        "nfe_total": sum(row["nfe"] for row in records),
        "residual_mask_count": sum(row["residual_mask_count"] for row in records),
        "duplicate_ids": 0,
        "missing_ids": 0,
        "prompt_hash_mismatches": 0,
        "target_hash_mismatches": 0,
        "samples_sha256": hashlib.sha256(samples_path.read_bytes()).hexdigest(),
        "task_records_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
    }
    return records, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--task-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records, summary = audit_files(args.samples, args.task_records, args.timeout)
    with (args.output_dir / "audit_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    raise SystemExit(0 if summary["all_crosschecks_pass"] else 2)


if __name__ == "__main__":
    main()
