#!/usr/bin/env python3
"""Visualize why trajectory-risk and answer-evidence branch selection succeeds or fails."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def answer_score(row: dict) -> float:
    return (
        2.0 * float(row.get("answer_pred_in_last_line", False))
        + float(row.get("answer_marker_in_tail", False))
        + min(float(row.get("answer_pred_count_tail", 0.0)), 2.0) * 0.5
        - min(float(row.get("answer_distinct_numbers_tail", 0.0)), 10.0) * 0.15
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base = json.load(open(args.base))["records"]
    branch = json.load(open(args.branch))["records"]
    n = min(len(base), len(branch))
    base, branch = base[:n], branch[:n]

    groups = {
        "Base correct": [i for i in range(n) if base[i]["correct"]],
        "Branch-only correct": [i for i in range(n) if not base[i]["correct"] and branch[i]["correct"]],
        "Both wrong": [i for i in range(n) if not base[i]["correct"] and not branch[i]["correct"]],
    }
    colors = {"Base correct": "#2a9d8f", "Branch-only correct": "#e76f51", "Both wrong": "#6c757d"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for name, indices in groups.items():
        axes[0].scatter(
            [float(base[i].get("response_selected_delta_mean", 0.0)) for i in indices],
            [answer_score(branch[i]) - answer_score(base[i]) for i in indices],
            label=f"{name} (n={len(indices)})", alpha=0.75, s=28, color=colors[name],
        )
    axes[0].axvline(0.30, color="black", linewidth=1, linestyle="--", label="Fixed risk threshold")
    axes[0].axhline(0.0, color="black", linewidth=1, linestyle=":")
    axes[0].set_xlabel("Base selected-response delta")
    axes[0].set_ylabel("Branch answer-evidence advantage")
    axes[0].set_title("Selector features by outcome")
    axes[0].legend(fontsize=8)

    labels = list(groups)
    data = [[float(base[i].get("response_selected_delta_mean", 0.0)) for i in groups[label]] for label in labels]
    axes[1].boxplot(data, tick_labels=labels, showfliers=False)
    axes[1].set_ylabel("Base selected-response delta")
    axes[1].set_title("Risk overlap across outcome groups")
    axes[1].tick_params(axis="x", rotation=18)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(out)


if __name__ == "__main__":
    main()
