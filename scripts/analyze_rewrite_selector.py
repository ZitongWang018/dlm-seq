#!/usr/bin/env python3
"""Select a rewrite-branch rule on one split and report a held-out split."""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path


def value(record: dict, key: str) -> float:
    return float(record.get(key, 0.0) or 0.0)


def evaluate(base: list[dict], branch: list[dict], rule: dict) -> dict:
    chosen = []
    triggers = 0
    accepts = 0
    for b, r in zip(base, branch):
        trigger = value(b, "response_selected_delta_mean") >= rule["trigger_response"]
        accept = (
            trigger
            and value(r, "branch_changed_count") >= rule["min_changed"]
            and value(r, "branch_event_delta") >= rule["min_event_delta"]
        )
        triggers += int(trigger)
        accepts += int(accept)
        chosen.append(r if accept else b)
    total = len(chosen)
    return {
        "accuracy": sum(bool(r["correct"]) for r in chosen) / total,
        "correct": sum(bool(r["correct"]) for r in chosen),
        "trigger_rate": triggers / total,
        "accept_rate": accepts / total,
        "avg_cost_multiplier": 1.0 + triggers / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--split", type=int, default=50)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = json.load(open(args.base))["records"]
    branch = json.load(open(args.branch))["records"]
    total = min(len(base), len(branch))
    base, branch = base[:total], branch[:total]
    if any(b.get("idx") != r.get("idx") for b, r in zip(base, branch)):
        raise ValueError("Base and branch records are not index-aligned")

    split = min(args.split, total // 2)
    dev_base, test_base = base[:split], base[split:]
    dev_branch, test_branch = branch[:split], branch[split:]
    rules = [
        {
            "trigger_response": t,
            "min_changed": c,
            "min_event_delta": d,
        }
        for t, c, d in product(
            [0.25, 0.30, 0.35, 0.40],
            [4, 8, 12, 16, 20],
            [1.4, 1.6, 1.8],
        )
    ]
    scored = []
    for rule in rules:
        dev = evaluate(dev_base, dev_branch, rule)
        test = evaluate(test_base, test_branch, rule)
        scored.append({"rule": rule, "dev": dev, "test": test})
    scored.sort(key=lambda row: (row["dev"]["accuracy"], -row["dev"]["avg_cost_multiplier"]), reverse=True)

    def accuracy(records: list[dict]) -> float:
        return sum(bool(r["correct"]) for r in records) / len(records)

    result = {
        "summary": {
            "total": total,
            "dev_size": len(dev_base),
            "test_size": len(test_base),
            "dev_base_accuracy": accuracy(dev_base),
            "dev_branch_accuracy": accuracy(dev_branch),
            "test_base_accuracy": accuracy(test_base),
            "test_branch_accuracy": accuracy(test_branch),
            "test_oracle_accuracy": sum(
                bool(b["correct"] or r["correct"]) for b, r in zip(test_base, test_branch)
            ) / len(test_base),
        },
        "selected_rule": scored[0],
        "top_dev_rules": scored[:10],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["summary"], indent=2))
    print(json.dumps(result["selected_rule"], indent=2))


if __name__ == "__main__":
    main()
