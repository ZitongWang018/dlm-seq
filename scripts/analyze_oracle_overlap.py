#!/usr/bin/env python3
"""Report overlap and oracle ceilings across result files."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path


def _load(path: str) -> list[dict]:
    return json.load(open(path))["records"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    records = {name: _load(path) for name, path in args.result}
    total = min(len(rows) for rows in records.values())
    records = {name: rows[:total] for name, rows in records.items()}

    methods = []
    for name, rows in records.items():
        correct = sum(bool(row["correct"]) for row in rows)
        methods.append({"name": name, "correct": correct, "accuracy": correct / total})

    combo_rows = []
    names = list(records)
    for size in range(2, len(names) + 1):
        for combo in combinations(names, size):
            oracle = sum(any(bool(records[name][i]["correct"]) for name in combo) for i in range(total))
            pred_diff = sum(
                len({records[name][i].get("pred") for name in combo}) > 1 for i in range(total)
            )
            combo_rows.append(
                {
                    "methods": list(combo),
                    "oracle_correct": oracle,
                    "oracle_accuracy": oracle / total,
                    "pred_diff_count": pred_diff,
                }
            )
    combo_rows.sort(key=lambda row: (row["oracle_accuracy"], row["pred_diff_count"]), reverse=True)
    result = {"total": total, "methods": methods, "combos": combo_rows}

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)

    print(json.dumps({"total": total, "methods": methods, "best_combos": combo_rows[:10]}, indent=2))


if __name__ == "__main__":
    main()
