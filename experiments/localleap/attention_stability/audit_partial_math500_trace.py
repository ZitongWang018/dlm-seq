#!/usr/bin/env python3
"""Independent post-generation health audit for an append-only MATH-500 trace."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

from audit_partial_gsm8k_trace import load_append_only_jsonl, wilson_interval


VERSION = "math500_partial_prism_aligned_health_v1"
DEFAULT_UTILS = Path(__file__).parent / "tasks" / "localleap_math500" / "utils.py"


def load_utils(path: Path):
    spec = importlib.util.spec_from_file_location("partial_math500_frozen_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load evaluator utils from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_nfe(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and value > 0
    )


def audit_records(
    records: list[dict[str, Any]],
    evaluator,
    *,
    expected_total: int | None = None,
    trailing_partial: int = 0,
    evaluator_path: Path | None = None,
) -> dict[str, Any]:
    absolute_ids: list[int] = []
    task_ids: list[str] = []
    invalid_ids = 0
    correct = 0
    extraction_failures = 0
    explicit_extractions = 0
    residual_masks = 0
    nfes: list[float] = []
    invalid_nfe = 0
    missing_prompt_hash = 0
    missing_gold = 0
    non_null_correct = 0

    for record in records:
        absolute_index = record.get("absolute_index")
        if isinstance(absolute_index, int) and not isinstance(absolute_index, bool):
            absolute_ids.append(absolute_index)
        else:
            invalid_ids += 1
        task_id_value = record.get("task_id")
        task_id = str(task_id_value) if task_id_value is not None else ""
        if not task_id:
            invalid_ids += 1
        task_ids.append(task_id)

        generation = record.get("decoded_generation")
        gold = record.get("raw_gold")
        if not isinstance(generation, str):
            raise ValueError(f"missing decoded_generation for task {task_id!r}")
        if gold is None or str(gold) == "":
            missing_gold += 1
            gold = ""
        prediction, explicit = evaluator.extract_answer(generation)
        extraction_failures += not bool(prediction)
        explicit_extractions += bool(explicit)
        correct += bool(evaluator.is_equiv(prediction, str(gold)))

        if not record.get("prompt_hash"):
            missing_prompt_hash += 1
        non_null_correct += record.get("correct") is not None
        nfe = record.get("nfe")
        if _valid_nfe(nfe):
            nfes.append(float(nfe))
        else:
            invalid_nfe += 1
        residual_masks += generation.count("[MASK]") + generation.count("<|mask|>")

    duplicate_task_ids = len(task_ids) - len(set(task_ids))
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
    if duplicate_task_ids:
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
    if missing_gold:
        anomalies.append("missing_post_generation_gold")
    if non_null_correct:
        anomalies.append("generation_trace_contains_non_null_correct")

    evaluator_hash = (
        hashlib.sha256(evaluator_path.read_bytes()).hexdigest()
        if evaluator_path is not None
        else None
    )
    return {
        "evaluator_version": VERSION,
        "task_evaluator_version": evaluator.EVALUATOR_VERSION,
        "task_evaluator_sha256": evaluator_hash,
        "records": len(records),
        "expected_total": expected_total,
        "complete": expected_total is not None and len(records) == expected_total,
        "correct": correct,
        "accuracy": correct / len(records) if records else None,
        "wilson_95_low": low if records else None,
        "wilson_95_high": high if records else None,
        "extraction_failures": extraction_failures,
        "explicit_extractions": explicit_extractions,
        "duplicate_task_ids": duplicate_task_ids,
        "duplicate_absolute_ids": duplicate_absolute_ids,
        "invalid_ids": invalid_ids,
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
        "missing_post_generation_gold": missing_gold,
        "non_null_correct": non_null_correct,
        "trailing_partial_lines": trailing_partial,
        "post_generation_only": True,
        "generation_inputs_used": ["decoded_generation"],
        "evaluation_only_fields_used": ["raw_gold"],
        "uses_hidden_tests": False,
        "anomalies": anomalies,
        "pass": not anomalies,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--expected-total", type=int)
    parser.add_argument("--utils", type=Path, default=DEFAULT_UTILS)
    args = parser.parse_args()

    records, trailing_partial = load_append_only_jsonl(args.trace)
    evaluator = load_utils(args.utils)
    summary = audit_records(
        records,
        evaluator,
        expected_total=args.expected_total,
        trailing_partial=trailing_partial,
        evaluator_path=args.utils,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if summary["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
