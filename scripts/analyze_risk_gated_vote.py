#!/usr/bin/env python3
"""Evaluate answer voting only on samples selected by base trajectory risk."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load(path: str) -> list[dict]:
    return json.load(open(path))["records"]


def value(row: dict, key: str) -> float:
    raw = row.get(key, 0.0)
    return float(raw) if isinstance(raw, (int, float)) else 0.0


def vote_index(rows: list[dict]) -> int:
    valid = [(idx, row.get("pred")) for idx, row in enumerate(rows) if row.get("pred") is not None]
    if not valid:
        return 0
    prediction, count = Counter(pred for _, pred in valid).most_common(1)[0]
    if count < 2:
        return 0
    return next(idx for idx, pred in valid if pred == prediction)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    names = [name for name, _ in args.result]
    records = {name: load(path) for name, path in args.result}
    total = min(len(rows) for rows in records.values())
    matrix = [[records[name][i] for name in names] for i in range(total)]
    rules = []
    grids = {
        "response_selected_delta_mean": (0.20, 0.25, 0.30, 0.35, 0.40, 0.45),
        "response_mean_delta": (0.18, 0.20, 0.22, 0.24, 0.26, 0.28),
        "answer_flip_mean": (4.0, 5.0, 6.0, 7.0, 8.0),
    }
    for key, thresholds in grids.items():
        for threshold in thresholds:
            chosen = []
            triggered = 0
            switches = 0
            for rows in matrix:
                idx = 0
                if value(rows[0], key) >= threshold:
                    triggered += 1
                    idx = vote_index(rows)
                switches += int(idx != 0)
                chosen.append(rows[idx])
            correct = sum(bool(row["correct"]) for row in chosen)
            rules.append({"risk": key, "threshold": threshold, "accuracy": correct / total, "correct": correct, "trigger_rate": triggered / total, "switch_rate": switches / total})
    rules.sort(key=lambda row: (row["accuracy"], -row["trigger_rate"]), reverse=True)
    base = sum(bool(rows[0]["correct"]) for rows in matrix) / total
    output = {"summary": {"methods": names, "total": total, "base_accuracy": base}, "best_rules": rules[:20], "rules": rules}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(output, open(out, "w"), indent=2)
    print(json.dumps(output["summary"], indent=2))
    print(json.dumps(output["best_rules"][:10], indent=2))


if __name__ == "__main__":
    main()
