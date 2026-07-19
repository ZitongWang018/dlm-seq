#!/usr/bin/env python3
"""Independently audit an append-only GSM8K generation trace.

This monitor intentionally mirrors the installed lm-eval GSM8K
``flexible-extract`` filter and exact-match normalization.  It is read-only and
does not import the active decoder, evaluator wrapper, or model runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


FLEXIBLE_PATTERN = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
IGNORE_PATTERNS = (r",", r"\$", r"(?s).*#### ", r"\.$")


def extract_flexible_answer(text: str) -> str:
    """Match lm-eval's regex(group_select=-1) plus take_first pipeline."""

    matches = FLEXIBLE_PATTERN.findall(text)
    if not matches:
        return "[invalid]"
    selected = matches[-1]
    if isinstance(selected, tuple):
        nonempty = [part for part in selected if part]
        return nonempty[0].strip() if nonempty else "[invalid]"
    return selected.strip()


def normalize_exact_match(text: str) -> str:
    """Apply GSM8K's configured regexes_to_ignore and ignore_case."""

    normalized = text
    for pattern in IGNORE_PATTERNS:
        normalized = re.sub(pattern, "", normalized)
    return normalized.lower()


def is_correct(generation: str, raw_gold: str) -> tuple[bool, str]:
    prediction = extract_flexible_answer(generation)
    return normalize_exact_match(prediction) == normalize_exact_match(raw_gold), prediction


def load_append_only_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Load complete JSON records, tolerating only one incomplete final line."""

    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    trailing_partial = 0
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            is_last = index == len(lines) - 1
            is_unterminated = not line.endswith((b"\n", b"\r"))
            if is_last and is_unterminated:
                trailing_partial += 1
                continue
            raise ValueError(f"malformed JSONL record at physical line {index + 1}")
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record at physical line {index + 1} is not an object")
        records.append(value)
    return records, trailing_partial


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius


def _finite_numbers(values: Iterable[Any]) -> tuple[list[float], int]:
    finite: list[float] = []
    invalid = 0
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            invalid += 1
        elif not math.isfinite(float(value)) or value <= 0:
            invalid += 1
        else:
            finite.append(float(value))
    return finite, invalid


def audit_records(
    records: list[dict[str, Any]],
    *,
    expected_total: int | None = None,
    trailing_partial: int = 0,
) -> dict[str, Any]:
    ids = [record.get("absolute_index") for record in records]
    integer_ids = [value for value in ids if isinstance(value, int) and not isinstance(value, bool)]
    invalid_ids = len(ids) - len(integer_ids)
    duplicate_ids = len(integer_ids) - len(set(integer_ids))
    minimum_id = min(integer_ids) if integer_ids else None
    maximum_id = max(integer_ids) if integer_ids else None
    missing_prefix_ids = (
        sorted(set(range(maximum_id + 1)) - set(integer_ids)) if maximum_id is not None else []
    )
    out_of_range_ids = (
        sorted(value for value in integer_ids if value < 0 or value >= expected_total)
        if expected_total is not None
        else sorted(value for value in integer_ids if value < 0)
    )

    correct = 0
    extraction_failures = 0
    for record in records:
        prediction = extract_flexible_answer(str(record.get("decoded_generation", "")))
        extraction_failures += prediction == "[invalid]"
        row_correct, _ = is_correct(
            str(record.get("decoded_generation", "")), str(record.get("raw_gold", ""))
        )
        correct += row_correct

    nfes, invalid_nfe = _finite_numbers(record.get("nfe") for record in records)
    residual_masks = sum(int(record.get("residual_mask_count", 0) or 0) for record in records)
    non_null_correct = sum(record.get("correct") is not None for record in records)
    lower, upper = wilson_interval(correct, len(records))

    anomalies: list[str] = []
    if invalid_ids:
        anomalies.append("invalid_absolute_ids")
    if duplicate_ids:
        anomalies.append("duplicate_absolute_ids")
    if missing_prefix_ids:
        anomalies.append("missing_prefix_ids")
    if out_of_range_ids:
        anomalies.append("out_of_range_ids")
    if invalid_nfe:
        anomalies.append("invalid_or_missing_nfe")
    if residual_masks:
        anomalies.append("residual_masks")
    if any(not record.get("prompt_hash") for record in records):
        anomalies.append("missing_prompt_hash")
    if non_null_correct:
        anomalies.append("generation_trace_contains_non_null_correct")

    return {
        "evaluator_version": "gsm8k_partial_trace_health_v1",
        "records": len(records),
        "expected_total": expected_total,
        "complete": expected_total is not None and len(records) == expected_total,
        "correct": correct,
        "accuracy": correct / len(records) if records else None,
        "wilson_95_low": lower if records else None,
        "wilson_95_high": upper if records else None,
        "extraction_failures": extraction_failures,
        "invalid_ids": invalid_ids,
        "duplicate_ids": duplicate_ids,
        "min_id": minimum_id,
        "max_id": maximum_id,
        "missing_prefix_ids": missing_prefix_ids,
        "out_of_range_ids": out_of_range_ids,
        "nfe_count": len(nfes),
        "nfe_total": sum(nfes),
        "nfe_min": min(nfes) if nfes else None,
        "nfe_max": max(nfes) if nfes else None,
        "invalid_or_missing_nfe": invalid_nfe,
        "residual_masks": residual_masks,
        "missing_prompt_hash": sum(not record.get("prompt_hash") for record in records),
        "missing_target_hash": sum(not record.get("target_hash") for record in records),
        "target_hash_phase": "post_generation_enrichment",
        "non_null_correct": non_null_correct,
        "trailing_partial_lines": trailing_partial,
        "anomalies": anomalies,
        "pass": not anomalies,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--expected-total", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records, trailing_partial = load_append_only_jsonl(args.trace)
    summary = audit_records(
        records, expected_total=args.expected_total, trailing_partial=trailing_partial
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if summary["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
