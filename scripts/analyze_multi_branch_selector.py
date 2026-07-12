#!/usr/bin/env python3
"""Evaluate simple training-free selectors over multiple decoded branches."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load(path: str) -> list[dict]:
    return json.load(open(path))["records"]


def _value(row: dict, key: str) -> float:
    value = row.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _majority_choice(rows: list[dict], fallback_idx: int = 0) -> int:
    counts = Counter(row.get("pred") for row in rows)
    pred, count = counts.most_common(1)[0]
    if count <= 1:
        return fallback_idx
    for idx, row in enumerate(rows):
        if row.get("pred") == pred:
            return idx
    return fallback_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    names = [name for name, _ in args.result]
    records = {name: _load(path) for name, path in args.result}
    total = min(len(rows) for rows in records.values())
    matrix = [[records[name][i] for name in names] for i in range(total)]

    rules = []
    base_idx = 0
    specs = []
    for key in ("response_selected_delta_mean", "response_mean_delta", "response_max_delta", "answer_flip_mean"):
        specs.append((f"max_{key}", key, "max"))
        specs.append((f"min_{key}", key, "min"))
    for threshold in (0.25, 0.30, 0.35, 0.40, 0.45):
        specs.append((f"base_else_max_selected_ge_{threshold}", "response_selected_delta_mean", threshold))
    for threshold in (0.20, 0.22, 0.24, 0.26):
        specs.append((f"base_else_max_mean_ge_{threshold}", "response_mean_delta", threshold))
    for base_th in (0.30, 0.35, 0.40):
        specs.append((f"risk_base_then_min_selected_base_ge_{base_th}", "response_selected_delta_mean", ("risk_min", base_th)))
    for base_th in (0.22, 0.24, 0.26):
        specs.append((f"risk_base_then_min_mean_base_ge_{base_th}", "response_mean_delta", ("risk_min", base_th)))
    for base_th in (2.0, 3.0, 4.0):
        specs.append((f"risk_base_then_min_answer_flip_base_ge_{base_th}", "answer_flip_mean", ("risk_min", base_th)))
    specs.append(("majority_else_base", "pred", "majority"))

    for rule_name, key, mode in specs:
        chosen = []
        switches = 0
        for rows in matrix:
            idx = base_idx
            if mode == "max":
                idx = max(range(len(rows)), key=lambda j: _value(rows[j], key))
            elif mode == "min":
                idx = min(range(len(rows)), key=lambda j: _value(rows[j], key))
            elif mode == "majority":
                idx = _majority_choice(rows, fallback_idx=base_idx)
            elif isinstance(mode, tuple) and mode[0] == "risk_min":
                base_threshold = float(mode[1])
                if _value(rows[base_idx], key) >= base_threshold:
                    idx = min(range(len(rows)), key=lambda j: _value(rows[j], key))
            else:
                candidates = [j for j in range(1, len(rows)) if _value(rows[j], key) >= float(mode)]
                if candidates:
                    idx = max(candidates, key=lambda j: _value(rows[j], key))
            switches += int(idx != base_idx)
            chosen.append(rows[idx])
        correct = sum(bool(row["correct"]) for row in chosen)
        rules.append(
            {
                "rule": rule_name,
                "accuracy": correct / total,
                "correct": correct,
                "switch_rate": switches / total,
                "switches": switches,
            }
        )

    base_correct = sum(bool(rows[base_idx]["correct"]) for rows in matrix)
    oracle_correct = sum(any(bool(row["correct"]) for row in rows) for rows in matrix)
    rules.sort(key=lambda row: (row["accuracy"], -row["switch_rate"]), reverse=True)
    result = {
        "summary": {
            "total": total,
            "methods": names,
            "base_accuracy": base_correct / total,
            "oracle_accuracy": oracle_correct / total,
        },
        "best_rules": rules[:20],
        "rules": rules,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["summary"], indent=2))
    print(json.dumps(result["best_rules"][:10], indent=2))


if __name__ == "__main__":
    main()
