#!/usr/bin/env python3
"""Test small training-free selectors using trajectory risk and decoded answer evidence."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load(path: str) -> list[dict]:
    return json.load(open(path))["records"]


def value(row: dict, key: str) -> float:
    raw = row.get(key, 0.0)
    return float(raw) if isinstance(raw, (int, float, bool)) else 0.0


def answer_score(row: dict) -> float:
    return (
        2.0 * value(row, "answer_pred_in_last_line")
        + value(row, "answer_marker_in_tail")
        + min(value(row, "answer_pred_count_tail"), 2.0) * 0.5
        - min(value(row, "answer_distinct_numbers_tail"), 10.0) * 0.15
    )


def majority_index(rows: list[dict], fallback: int = 0) -> int:
    predictions = [row.get("pred") for row in rows if row.get("pred") is not None]
    if not predictions:
        return fallback
    prediction, count = Counter(predictions).most_common(1)[0]
    if count < 2:
        return fallback
    matched = [i for i, row in enumerate(rows) if row.get("pred") == prediction]
    return max(matched, key=lambda i: answer_score(rows[i]))


def evaluate(matrix: list[list[dict]], name: str, choose):
    chosen = [rows[choose(rows)] for rows in matrix]
    return {
        "rule": name,
        "correct": sum(bool(row["correct"]) for row in chosen),
        "accuracy": sum(bool(row["correct"]) for row in chosen) / len(chosen),
        "switches": sum(choose(rows) != 0 for rows in matrix),
        "switch_rate": sum(choose(rows) != 0 for rows in matrix) / len(chosen),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    names = [name for name, _ in args.result]
    loaded = {name: load(path) for name, path in args.result}
    total = min(len(rows) for rows in loaded.values())
    matrix = [[loaded[name][i] for name in names] for i in range(total)]

    rules = [evaluate(matrix, "majority_answer", majority_index)]
    for risk_key in ("response_selected_delta_mean", "response_mean_delta", "answer_flip_mean"):
        for threshold in (0.20, 0.25, 0.30, 0.35, 0.40, 2.0, 3.0):
            def risk_then_vote(rows, key=risk_key, th=threshold):
                return majority_index(rows) if value(rows[0], key) >= th else 0
            rules.append(evaluate(matrix, f"risk_{risk_key}_ge_{threshold}_then_majority", risk_then_vote))
    for risk_threshold in (0.25, 0.30, 0.35, 0.40):
        for margin in (0.0, 0.5, 1.0):
            def risk_then_evidence(rows, th=risk_threshold, m=margin):
                if value(rows[0], "response_selected_delta_mean") < th:
                    return 0
                candidate = max(range(1, len(rows)), key=lambda i: answer_score(rows[i]))
                return candidate if answer_score(rows[candidate]) >= answer_score(rows[0]) + m else 0
            rules.append(evaluate(matrix, f"risk_selected_ge_{risk_threshold}_then_evidence_margin_{margin}", risk_then_evidence))

    base_correct = sum(bool(rows[0]["correct"]) for rows in matrix)
    oracle_correct = sum(any(bool(row["correct"]) for row in rows) for rows in matrix)
    rules.sort(key=lambda row: (row["accuracy"], -row["switch_rate"]), reverse=True)
    result = {"summary": {"total": total, "methods": names, "base_accuracy": base_correct / total, "oracle_accuracy": oracle_correct / total}, "best_rules": rules[:20], "rules": rules}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["summary"], indent=2))
    print(json.dumps(result["best_rules"][:10], indent=2))


if __name__ == "__main__":
    main()
