#!/usr/bin/env python3
"""Analyze whether two decoded branches can be selected cheaply."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _trigger(record: dict, rule: str, threshold: float) -> bool:
    if rule == "response_selected_delta_mean":
        return record.get("response_selected_delta_mean", 0.0) >= threshold
    if rule == "response_mean_delta":
        return record.get("response_mean_delta", 0.0) >= threshold
    if rule == "response_max_delta":
        return record.get("response_max_delta", 0.0) >= threshold
    raise ValueError(rule)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trigger_source", choices=["base", "branch"], default="branch")
    args = parser.parse_args()

    base = json.load(open(args.base))
    branch = json.load(open(args.branch))
    b_records = base["records"]
    r_records = branch["records"]

    total = len(b_records)
    base_correct = sum(bool(r["correct"]) for r in b_records)
    branch_correct = sum(bool(r["correct"]) for r in r_records)
    oracle_correct = sum(bool(b["correct"] or r["correct"]) for b, r in zip(b_records, r_records))

    rules = []
    for rule, thresholds in {
        "response_selected_delta_mean": [0.25, 0.30, 0.35, 0.40, 0.45],
        "response_mean_delta": [0.18, 0.20, 0.22, 0.24],
        "response_max_delta": [1.5, 1.7, 1.9],
    }.items():
        for threshold in thresholds:
            chosen = []
            triggers = 0
            for b, r in zip(b_records, r_records):
                trigger_record = b if args.trigger_source == "base" else r
                use_branch = _trigger(trigger_record, rule, threshold)
                triggers += int(use_branch)
                chosen.append(r if use_branch else b)
            correct = sum(bool(r["correct"]) for r in chosen)
            rules.append(
                {
                    "rule": rule,
                    "threshold": threshold,
                    "accuracy": correct / total,
                    "correct": correct,
                    "trigger_rate": triggers / total,
                    "triggers": triggers,
                    "avg_cost_multiplier": 1.0 + triggers / total,
                }
            )
    rules.sort(key=lambda x: (x["accuracy"], -x["trigger_rate"]), reverse=True)

    result = {
        "summary": {
            "total": total,
            "base_accuracy": base_correct / total,
            "branch_accuracy": branch_correct / total,
            "oracle_accuracy": oracle_correct / total,
            "oracle_gain": (oracle_correct - base_correct) / total,
        },
        "rules": rules,
        "best_rules": rules[:10],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["summary"], indent=2))
    print(json.dumps(result["best_rules"][:5], indent=2))


if __name__ == "__main__":
    main()
