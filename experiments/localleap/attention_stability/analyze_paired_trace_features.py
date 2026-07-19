#!/usr/bin/env python3
"""Post-generation feature audit for stable-ID paired decoder results.

This tool never changes a selector. It joins an already-completed paired audit
to immutable trace records and summarizes decoder diagnostics by paired outcome.
Gold answers and generated text are deliberately not read.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Callable


def load_jsonl(path: Path, identity: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = str(record[identity])
            if key in records:
                raise ValueError(f"duplicate {identity}={key} in {path}:{line_number}")
            records[key] = record
    return records


def nested(record: dict[str, Any], path: str) -> Any:
    value: Any = record
    for component in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(component)
    return value


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def outcome(record: dict[str, Any]) -> str:
    baseline = bool(record["baseline_correct"])
    method = bool(record["method_correct"])
    if method and not baseline:
        return "method_only"
    if baseline and not method:
        return "baseline_only"
    if method:
        return "both_correct"
    return "both_incorrect"


NUMERIC_FEATURES = {
    "method_nfe": lambda pair, trace: pair.get("method_nfe"),
    "disagreement_token_count": lambda pair, trace: nested(
        trace, "decode_diagnostics.disagreement_token_count"
    ),
    "scored_disagreement_token_count": lambda pair, trace: nested(
        trace, "decode_diagnostics.scored_disagreement_token_count"
    ),
    "block_evidence_margin": lambda pair, trace: nested(
        trace, "decode_diagnostics.block_evidence_margin"
    ),
    "selected_score": lambda pair, trace: nested(
        trace, "decode_diagnostics.selected_score"
    ),
    "accuracy_minus_fast_score": lambda pair, trace: (
        finite_number(nested(trace, "decode_diagnostics.selection_candidate_scores.accuracy"))
        - finite_number(nested(trace, "decode_diagnostics.selection_candidate_scores.fast"))
        if finite_number(
            nested(trace, "decode_diagnostics.selection_candidate_scores.accuracy")
        )
        is not None
        and finite_number(
            nested(trace, "decode_diagnostics.selection_candidate_scores.fast")
        )
        is not None
        else None
    ),
    "accuracy_unstable_candidates": lambda pair, trace: nested(
        trace, "decode_diagnostics.candidate_summaries.accuracy.unstable_candidates"
    ),
    "fast_unstable_candidates": lambda pair, trace: nested(
        trace, "decode_diagnostics.candidate_summaries.fast.unstable_candidates"
    ),
    "accuracy_commit_logprob_mean": lambda pair, trace: nested(
        trace, "decode_diagnostics.candidate_summaries.accuracy.commit_logprob_mean"
    ),
    "fast_commit_logprob_mean": lambda pair, trace: nested(
        trace, "decode_diagnostics.candidate_summaries.fast.commit_logprob_mean"
    ),
}


def summarize_group(
    joined: list[tuple[dict[str, Any], dict[str, Any]]]
) -> dict[str, Any]:
    numeric: dict[str, Any] = {}
    for name, getter in NUMERIC_FEATURES.items():
        values = [
            number
            for pair, trace in joined
            if (number := finite_number(getter(pair, trace))) is not None
        ]
        numeric[name] = numeric_summary(values)

    selected = Counter(
        str(nested(trace, "decode_diagnostics.selected_name"))
        for _, trace in joined
    )
    public_status = Counter(
        str(nested(trace, "decode_diagnostics.public_example_guard.status"))
        for _, trace in joined
    )
    early_abort = sum(
        nested(trace, "decode_diagnostics.accuracy_early_abort.triggered") is True
        for _, trace in joined
    )
    baseline_generated = sum(
        nested(trace, "decode_diagnostics.public_example_guard.baseline_generated") is True
        for _, trace in joined
    )
    count = len(joined)
    return {
        "count": count,
        "selected_name_counts": dict(sorted(selected.items())),
        "public_guard_status_counts": dict(sorted(public_status.items())),
        "accuracy_early_abort_rate": early_abort / count if count else None,
        "baseline_generated_rate": baseline_generated / count if count else None,
        "numeric": numeric,
    }


def make_stratum(value: Any, *, positive_labels: tuple[str, str] | None = None) -> str:
    if positive_labels is not None:
        number = finite_number(value)
        if number is None:
            return "missing"
        return positive_labels[0] if number > 0 else positive_labels[1]
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def cross_tabs(
    joined: list[tuple[dict[str, Any], dict[str, Any]]]
) -> dict[str, dict[str, dict[str, int]]]:
    extractors: dict[str, Callable[[dict[str, Any], dict[str, Any]], str]] = {
        "selected_name": lambda pair, trace: make_stratum(
            nested(trace, "decode_diagnostics.selected_name")
        ),
        "accuracy_early_abort": lambda pair, trace: make_stratum(
            nested(trace, "decode_diagnostics.accuracy_early_abort.triggered")
        ),
        "baseline_generated": lambda pair, trace: make_stratum(
            nested(trace, "decode_diagnostics.public_example_guard.baseline_generated")
        ),
        "public_guard_status": lambda pair, trace: make_stratum(
            nested(trace, "decode_diagnostics.public_example_guard.status")
        ),
        "disagreement_tokens": lambda pair, trace: (
            "zero"
            if finite_number(
                nested(trace, "decode_diagnostics.disagreement_token_count")
            )
            == 0
            else "nonzero"
        ),
        "block_evidence_margin": lambda pair, trace: make_stratum(
            nested(trace, "decode_diagnostics.block_evidence_margin"),
            positive_labels=("positive", "nonpositive"),
        ),
        "accuracy_minus_fast_score": lambda pair, trace: make_stratum(
            (
                finite_number(
                    nested(
                        trace,
                        "decode_diagnostics.selection_candidate_scores.accuracy",
                    )
                )
                - finite_number(
                    nested(
                        trace,
                        "decode_diagnostics.selection_candidate_scores.fast",
                    )
                )
                if finite_number(
                    nested(
                        trace,
                        "decode_diagnostics.selection_candidate_scores.accuracy",
                    )
                )
                is not None
                and finite_number(
                    nested(
                        trace,
                        "decode_diagnostics.selection_candidate_scores.fast",
                    )
                )
                is not None
                else None
            ),
            positive_labels=("accuracy_positive", "fast_nonpositive"),
        ),
    }
    result: dict[str, dict[str, dict[str, int]]] = {}
    for feature, extractor in extractors.items():
        table: dict[str, Counter[str]] = {}
        for pair, trace in joined:
            stratum = extractor(pair, trace)
            table.setdefault(stratum, Counter())[outcome(pair)] += 1
        result[feature] = {}
        for stratum, counts in sorted(table.items()):
            row = {name: counts.get(name, 0) for name in (
                "method_only", "baseline_only", "both_correct", "both_incorrect"
            )}
            row["paired_balance"] = row["method_only"] - row["baseline_only"]
            row["total"] = sum(row[name] for name in (
                "method_only", "baseline_only", "both_correct", "both_incorrect"
            ))
            result[feature][stratum] = row
    return result


def build_report(pair_path: Path, trace_path: Path) -> dict[str, Any]:
    pairs = load_jsonl(pair_path, "stable_task_id")
    traces = load_jsonl(trace_path, "task_id")
    pair_ids = set(pairs)
    trace_ids = set(traces)
    if pair_ids != trace_ids:
        raise ValueError(
            f"identity mismatch: missing_trace={len(pair_ids - trace_ids)} "
            f"extra_trace={len(trace_ids - pair_ids)}"
        )

    prompt_mismatches = [
        task_id
        for task_id in sorted(pair_ids)
        if pairs[task_id]["prompt_hash"] != traces[task_id]["prompt_hash"]
    ]
    if prompt_mismatches:
        raise ValueError(f"prompt hash mismatches: {len(prompt_mismatches)}")

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
        "method_only": [],
        "baseline_only": [],
        "both_correct": [],
        "both_incorrect": [],
    }
    all_joined: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for task_id in sorted(pair_ids):
        pair = pairs[task_id]
        joined = (pair, traces[task_id])
        all_joined.append(joined)
        grouped[outcome(pair)].append(joined)

    return {
        "schema": "localleap_paired_trace_feature_audit_v1",
        "usage": "post_generation_diagnostic_only",
        "selection_or_generation_mutation": False,
        "gold_or_generation_text_read": False,
        "pair_records": len(pairs),
        "trace_records": len(traces),
        "duplicate_or_missing_ids": 0,
        "prompt_hash_mismatches": 0,
        "groups": {
            name: summarize_group(records) for name, records in grouped.items()
        },
        "outcome_cross_tabs": cross_tabs(all_joined),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-records", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = build_report(args.paired_records, args.trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
