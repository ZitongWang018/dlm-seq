#!/usr/bin/env python3
"""Plot focused diagnostics for the response-delay hypothesis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data = json.load(open(args.analysis))
    stats = data["group_stats"]
    groups = ["only_candidate", "only_base", "same_correct", "same_wrong"]
    labels = ["Fixed by delay", "Broken by delay", "Both correct", "Both wrong"]

    near = [stats[g]["mean_response_near_delta"] for g in groups]
    far = [stats[g]["mean_response_far_delta"] for g in groups]
    delayed = [stats[g]["mean_response_delayed_count"] for g in groups]

    x = np.arange(len(groups))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.36
    axes[0].bar(x - width / 2, near, width, label="Near recent commits")
    axes[0].bar(x + width / 2, far, width, label="Far from recent commits")
    axes[0].set_ylabel("Mean distribution change (top-k L1)")
    axes[0].set_title("Lateral response by outcome group")
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].legend(frameon=False)

    axes[1].bar(x, delayed, color="#4472C4")
    axes[1].set_ylabel("Mean delayed candidates per sample")
    axes[1].set_title("Actual timing changes by outcome group")
    axes[1].set_xticks(x, labels, rotation=18, ha="right")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
