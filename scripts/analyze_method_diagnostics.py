#!/usr/bin/env python3
"""Compare two method result files with intermediate diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _mean(records: list[dict], key: str) -> float:
    vals = [r[key] for r in records if key in r and isinstance(r[key], (int, float))]
    return float(np.mean(vals)) if vals else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = json.load(open(args.base))
    cand = json.load(open(args.candidate))
    base_records = base["records"]
    cand_records = cand["records"]
    total = min(len(base_records), len(cand_records))
    base_records = base_records[:total]
    cand_records = cand_records[:total]
    if any(b.get("idx") != c.get("idx") for b, c in zip(base_records, cand_records)):
        raise ValueError("Base and candidate records are not index-aligned")

    only_base = []
    only_candidate = []
    pred_diff = []
    same_correct = []
    same_wrong = []

    for b, c in zip(base_records, cand_records):
        row = {
            "idx": b["idx"],
            "base_correct": b["correct"],
            "candidate_correct": c["correct"],
            "base_pred": b.get("pred"),
            "candidate_pred": c.get("pred"),
            "gold": b.get("gold"),
            "candidate_response_max_delta": c.get("response_max_delta", 0.0),
            "candidate_response_mean_delta": c.get("response_mean_delta", 0.0),
            "candidate_response_selected_delta_mean": c.get("response_selected_delta_mean", 0.0),
            "candidate_response_selected_count": c.get("response_selected_count", 0),
            "candidate_response_delay_candidate_count": c.get("response_delay_candidate_count", 0),
            "candidate_response_delay_selected_count": c.get("response_delay_selected_count", 0),
            "candidate_response_delayed_count": c.get("response_delayed_count", 0),
            "candidate_response_near_delta_mean": c.get("response_near_delta_mean", 0.0),
            "candidate_response_far_delta_mean": c.get("response_far_delta_mean", 0.0),
            "candidate_response_locality_gap": (
                c.get("response_near_delta_mean", 0.0) - c.get("response_far_delta_mean", 0.0)
            ),
        }
        if b.get("pred") != c.get("pred"):
            pred_diff.append(row)
        if b["correct"] and not c["correct"]:
            only_base.append(row)
        elif c["correct"] and not b["correct"]:
            only_candidate.append(row)
        elif b["correct"] and c["correct"]:
            same_correct.append(row)
        else:
            same_wrong.append(row)

    groups = {
        "only_base": only_base,
        "only_candidate": only_candidate,
        "pred_diff": pred_diff,
        "same_correct": same_correct,
        "same_wrong": same_wrong,
    }
    group_stats = {
        name: {
            "n": len(rows),
            "mean_response_max_delta": _mean(rows, "candidate_response_max_delta"),
            "mean_response_mean_delta": _mean(rows, "candidate_response_mean_delta"),
            "mean_response_selected_delta_mean": _mean(rows, "candidate_response_selected_delta_mean"),
            "mean_response_selected_count": _mean(rows, "candidate_response_selected_count"),
            "mean_response_delay_candidate_count": _mean(rows, "candidate_response_delay_candidate_count"),
            "mean_response_delay_selected_count": _mean(rows, "candidate_response_delay_selected_count"),
            "mean_response_delayed_count": _mean(rows, "candidate_response_delayed_count"),
            "mean_response_near_delta": _mean(rows, "candidate_response_near_delta_mean"),
            "mean_response_far_delta": _mean(rows, "candidate_response_far_delta_mean"),
            "mean_response_locality_gap": _mean(rows, "candidate_response_locality_gap"),
        }
        for name, rows in groups.items()
    }

    result = {
        "base": {
            "path": args.base,
            "accuracy": sum(bool(r["correct"]) for r in base_records) / total,
            "correct": sum(bool(r["correct"]) for r in base_records),
            "total": total,
        },
        "candidate": {
            "path": args.candidate,
            "accuracy": sum(bool(r["correct"]) for r in cand_records) / total,
            "correct": sum(bool(r["correct"]) for r in cand_records),
            "total": total,
        },
        "summary": {
            "num_pred_diff": len(pred_diff),
            "only_base": len(only_base),
            "only_candidate": len(only_candidate),
            "net_gain": len(only_candidate) - len(only_base),
        },
        "candidate_diagnostics": {
            key: _mean(cand_records, key)
            for key in (
                "nfe", "rewrite_event_count", "refine_masked_count", "refine_changed_count",
                "response_confirmed_count", "response_delayed_count",
                "response_budget_steps", "response_budget_deferred_tokens",
                "response_persistence_candidates", "response_persistence_confirmed",
                "response_alignment_steps", "response_alignment_margin_sum",
                "response_alignment_winner_sum", "response_alignment_runner_up_sum",
                "wavefront_expansion_steps", "wavefront_narrow_steps",
                "effective_gen_length", "branch_seed_relative", "branch_event_delta",
                "branch_changed_count",
            )
        },
        "group_stats": group_stats,
        "examples": {
            "only_base": only_base[:30],
            "only_candidate": only_candidate[:30],
            "pred_diff": pred_diff[:50],
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["summary"], indent=2))
    print(json.dumps(result["candidate_diagnostics"], indent=2))
    print(json.dumps(group_stats, indent=2))


if __name__ == "__main__":
    main()
