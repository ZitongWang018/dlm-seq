#!/usr/bin/env python3
"""Paired evaluation and English-labeled plot for region-scale decoding."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def exact_pvalue(base_only: int, method_only: int) -> float:
    total = base_only + method_only
    if total == 0:
        return 1.0
    lower = min(base_only, method_only)
    tail = sum(math.comb(total, k) for k in range(lower + 1)) / (2 ** total)
    return min(1.0, 2.0 * tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--base_label", default="Single block (128 tokens)")
    parser.add_argument("--method_label", default="Four blocks (32 tokens)")
    args = parser.parse_args()

    base = json.load(open(args.base))["records"]
    method = json.load(open(args.method))["records"]
    if len(base) != len(method):
        raise ValueError("Result files have different sample counts")

    base_only = sum(bool(a["correct"]) and not bool(b["correct"]) for a, b in zip(base, method))
    method_only = sum(not bool(a["correct"]) and bool(b["correct"]) for a, b in zip(base, method))
    both_correct = sum(bool(a["correct"]) and bool(b["correct"]) for a, b in zip(base, method))
    both_wrong = len(base) - base_only - method_only - both_correct
    base_accuracy = sum(bool(row["correct"]) for row in base) / len(base)
    method_accuracy = sum(bool(row["correct"]) for row in method) / len(method)
    result = {
        "total": len(base),
        "base_accuracy": base_accuracy,
        "method_accuracy": method_accuracy,
        "difference": method_accuracy - base_accuracy,
        "base_only": base_only,
        "method_only": method_only,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "mcnemar_exact_pvalue": exact_pvalue(base_only, method_only),
        "base_avg_nfe": sum(row["nfe"] for row in base) / len(base),
        "method_avg_nfe": sum(row["nfe"] for row in method) / len(method),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "block_decoding_analysis.json", "w") as handle:
        json.dump(result, handle, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1))
    fig.patch.set_facecolor("white")
    for axis in axes:
        axis.set_facecolor("white")
    accuracy_bars = axes[0].bar([args.base_label, args.method_label], [base_accuracy, method_accuracy], color=["#4C78A8", "#59A14F"])
    axes[0].bar_label(accuracy_bars, labels=[f"{base_accuracy:.1%}", f"{method_accuracy:.1%}"], padding=3)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("GSM8K accuracy at equal NFE")
    axes[0].tick_params(axis="x", rotation=12)

    outcome_bars = axes[1].bar(["Recovered", "Lost"], [method_only, base_only], color=["#59A14F", "#E15759"])
    axes[1].bar_label(outcome_bars, padding=3)
    axes[1].set_ylim(0, max(method_only, base_only, 1) * 1.2)
    axes[1].set_ylabel("Number of samples")
    axes[1].set_title("Paired outcome changes")
    fig.suptitle("Region-scale sequential decoding")
    fig.tight_layout()
    fig.savefig(out_dir / "block_decoding_evaluation.png", dpi=180, facecolor="white")
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
