#!/usr/bin/env python3
"""Compute paired accuracy changes and an exact McNemar test for result records."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_rows(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        data = json.load(open(path))
        rows.extend(data["records"])
    return rows


def two_sided_binomial_pvalue(a: int, b: int) -> float:
    total = a + b
    if total == 0:
        return 1.0
    lower = min(a, b)
    tail = sum(math.comb(total, k) for k in range(lower + 1)) / (2 ** total)
    return min(1.0, 2.0 * tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = load_rows(args.result)
    base_only = sum(bool(row["base_correct"]) and not bool(row["correct"]) for row in rows)
    method_only = sum(not bool(row["base_correct"]) and bool(row["correct"]) for row in rows)
    base = sum(bool(row["base_correct"]) for row in rows)
    method = sum(bool(row["correct"]) for row in rows)
    result = {
        "total": len(rows), "base_correct": base, "method_correct": method,
        "base_accuracy": base / len(rows), "method_accuracy": method / len(rows),
        "difference": (method - base) / len(rows), "base_only": base_only,
        "method_only": method_only, "mcnemar_exact_pvalue": two_sided_binomial_pvalue(base_only, method_only),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out, "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
