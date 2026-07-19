#!/usr/bin/env python3
"""Read-only health audit for an append-only MBPP generation trace.

Only assertions literally present in the current prompt segment are executed.
Reference implementations and hidden tests are never read.  The primary and
independent execution paths must agree before the health report passes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from audit_mbpp_assertions import independent_assertion_execution
from audit_partial_gsm8k_trace import load_append_only_jsonl, wilson_interval
from differential_selector import evaluate_public_candidate


VERSION = "mbpp_partial_prompt_assertion_health_v1"


def _valid_nfe(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and value > 0
    )


def audit_records(
    records: list[dict[str, Any]],
    *,
    expected_total: int | None = None,
    timeout: float = 1.5,
    trailing_partial: int = 0,
) -> dict[str, Any]:
    identities: list[str] = []
    absolute_ids: list[int] = []
    invalid_ids = 0
    duplicate_ids = 0
    correct = 0
    compile_valid = 0
    no_visible_checks = 0
    crosscheck_mismatches: list[str] = []
    residual_masks = 0
    nfes: list[float] = []
    invalid_nfe = 0
    missing_prompt_hash = 0

    seen_identity: set[str] = set()
    for record in records:
        task_id_value = record.get("task_id")
        identity = str(task_id_value) if task_id_value is not None else ""
        if not identity or identity in seen_identity:
            duplicate_ids += bool(identity in seen_identity)
            invalid_ids += not identity
        seen_identity.add(identity)
        identities.append(identity)

        absolute_index = record.get("absolute_index")
        if isinstance(absolute_index, int) and not isinstance(absolute_index, bool):
            absolute_ids.append(absolute_index)
        else:
            invalid_ids += 1

        prompt = record.get("prompt_text")
        generation = record.get("decoded_generation")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"missing prompt_text for task {identity!r}")
        if not isinstance(generation, str):
            raise ValueError(f"missing decoded_generation for task {identity!r}")
        if not record.get("prompt_hash"):
            missing_prompt_hash += 1

        primary = evaluate_public_candidate(generation, prompt, record.get("entry_point"))
        visible_count = int(primary["visible_check_count"])
        primary_pass = bool(
            visible_count and int(primary["visible_checks_passed"]) == visible_count
        )
        independent = independent_assertion_execution(
            generation,
            prompt,
            record.get("entry_point"),
            identity,
            timeout,
        )
        crosscheck_match = (
            visible_count == int(independent["assertion_count"])
            and primary_pass == bool(independent["passed"])
        )
        if not crosscheck_match:
            crosscheck_mismatches.append(identity)
        correct += primary_pass
        compile_valid += bool(primary["compile_valid"])
        no_visible_checks += visible_count == 0

        nfe = record.get("nfe")
        if _valid_nfe(nfe):
            nfes.append(float(nfe))
        else:
            invalid_nfe += 1
        residual_masks += generation.count("[MASK]") + generation.count("<|mask|>")

    duplicate_absolute_ids = len(absolute_ids) - len(set(absolute_ids))
    min_id = min(absolute_ids) if absolute_ids else None
    max_id = max(absolute_ids) if absolute_ids else None
    missing_prefix_ids = (
        sorted(set(range(max_id + 1)) - set(absolute_ids)) if max_id is not None else []
    )
    out_of_range_ids = (
        sorted(value for value in absolute_ids if value < 0 or value >= expected_total)
        if expected_total is not None
        else sorted(value for value in absolute_ids if value < 0)
    )
    low, high = wilson_interval(correct, len(records))

    anomalies: list[str] = []
    if invalid_ids:
        anomalies.append("invalid_task_or_absolute_ids")
    if duplicate_ids:
        anomalies.append("duplicate_task_ids")
    if duplicate_absolute_ids:
        anomalies.append("duplicate_absolute_ids")
    if missing_prefix_ids:
        anomalies.append("missing_prefix_ids")
    if out_of_range_ids:
        anomalies.append("out_of_range_ids")
    if invalid_nfe:
        anomalies.append("invalid_or_missing_nfe")
    if residual_masks:
        anomalies.append("residual_masks")
    if missing_prompt_hash:
        anomalies.append("missing_prompt_hash")
    if no_visible_checks:
        anomalies.append("missing_prompt_visible_checks")
    if crosscheck_mismatches:
        anomalies.append("independent_execution_mismatch")

    return {
        "evaluator_version": VERSION,
        "records": len(records),
        "expected_total": expected_total,
        "complete": expected_total is not None and len(records) == expected_total,
        "correct": correct,
        "accuracy": correct / len(records) if records else None,
        "wilson_95_low": low if records else None,
        "wilson_95_high": high if records else None,
        "compile_valid": compile_valid,
        "missing_visible_checks": no_visible_checks,
        "crosscheck_mismatch_ids": crosscheck_mismatches,
        "duplicate_task_ids": duplicate_ids,
        "invalid_ids": invalid_ids,
        "duplicate_absolute_ids": duplicate_absolute_ids,
        "min_id": min_id,
        "max_id": max_id,
        "missing_prefix_ids": missing_prefix_ids,
        "out_of_range_ids": out_of_range_ids,
        "nfe_count": len(nfes),
        "nfe_total": sum(nfes),
        "nfe_min": min(nfes) if nfes else None,
        "nfe_max": max(nfes) if nfes else None,
        "invalid_or_missing_nfe": invalid_nfe,
        "residual_masks": residual_masks,
        "missing_prompt_hash": missing_prompt_hash,
        "trailing_partial_lines": trailing_partial,
        "uses_hidden_tests": False,
        "uses_reference_solution": False,
        "anomalies": anomalies,
        "pass": not anomalies,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--expected-total", type=int)
    parser.add_argument("--timeout", type=float, default=1.5)
    args = parser.parse_args()

    records, trailing_partial = load_append_only_jsonl(args.trace)
    summary = audit_records(
        records,
        expected_total=args.expected_total,
        timeout=args.timeout,
        trailing_partial=trailing_partial,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if summary["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
