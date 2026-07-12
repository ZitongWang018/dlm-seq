#!/usr/bin/env python3
"""Analyze a deployable two-stage branch selector.

Stage 1 uses base/LCR-probe diagnostics to decide whether to run a branch.
Stage 2 uses branch diagnostics to decide whether to accept the branch output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _value(record: dict, key: str) -> float:
    return float(record.get(key, 0.0) or 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = json.load(open(args.base))
    branch = json.load(open(args.branch))
    b_records = base["records"]
    r_records = branch["records"]
    total = len(b_records)

    trigger_specs = [
        ("response_selected_delta_mean", 0.25),
        ("response_selected_delta_mean", 0.30),
        ("response_selected_delta_mean", 0.35),
        ("response_selected_delta_mean", 0.40),
        ("response_selected_delta_mean", 0.45),
        ("answer_flip_mean", 2.0),
        ("answer_flip_mean", 3.0),
        ("answer_flip_mean", 4.0),
        ("answer_or_response", (3.0, 0.35)),
        ("answer_or_response", (4.0, 0.40)),
    ]
    accept_specs = [
        ("response_selected_delta_mean", 0.25),
        ("response_selected_delta_mean", 0.30),
        ("response_selected_delta_mean", 0.35),
        ("response_selected_delta_mean", 0.40),
        ("response_selected_delta_mean", 0.45),
        ("branch_lower_answer_flip", 0.0),
        ("branch_much_lower_answer_flip", 1.0),
        ("branch_answer_flip_below", 2.0),
        ("branch_answer_flip_below", 3.0),
    ]

    rows = []
    for trigger_key, t_th in trigger_specs:
        for accept_key, a_th in accept_specs:
            chosen = []
            triggers = 0
            accepts = 0
            for b, r in zip(b_records, r_records):
                if trigger_key == "answer_or_response":
                    flip_th, response_th = t_th
                    trigger = (
                        _value(b, "answer_flip_mean") >= flip_th
                        or _value(b, "response_selected_delta_mean") >= response_th
                    )
                else:
                    trigger = _value(b, trigger_key) >= t_th
                if accept_key == "branch_lower_answer_flip":
                    accept = trigger and _value(r, "answer_flip_mean") < _value(b, "answer_flip_mean")
                elif accept_key == "branch_much_lower_answer_flip":
                    accept = trigger and _value(r, "answer_flip_mean") + a_th < _value(b, "answer_flip_mean")
                elif accept_key == "branch_answer_flip_below":
                    accept = trigger and _value(r, "answer_flip_mean") <= a_th
                else:
                    accept = trigger and _value(r, accept_key) >= a_th
                triggers += int(trigger)
                accepts += int(accept)
                chosen.append(r if accept else b)
            correct = sum(bool(x["correct"]) for x in chosen)
            rows.append(
                {
                    "trigger_key": trigger_key,
                    "trigger_threshold": t_th,
                    "accept_key": accept_key,
                    "accept_threshold": a_th,
                    "accuracy": correct / total,
                    "correct": correct,
                    "trigger_rate": triggers / total,
                    "accept_rate": accepts / total,
                    "avg_cost_multiplier": 1.0 + triggers / total,
                }
            )

    rows.sort(key=lambda x: (x["accuracy"], -x["avg_cost_multiplier"]), reverse=True)
    base_correct = sum(bool(x["correct"]) for x in b_records)
    branch_correct = sum(bool(x["correct"]) for x in r_records)
    oracle_correct = sum(bool(b["correct"] or r["correct"]) for b, r in zip(b_records, r_records))
    result = {
        "summary": {
            "total": total,
            "base_accuracy": base_correct / total,
            "branch_accuracy": branch_correct / total,
            "oracle_accuracy": oracle_correct / total,
        },
        "best_rules": rows[:20],
        "rules": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["summary"], indent=2))
    print(json.dumps(result["best_rules"][:10], indent=2))


if __name__ == "__main__":
    main()
