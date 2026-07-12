#!/usr/bin/env python3
"""Plot fixed-policy accuracy and NFE across independent evaluation slices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    labels, base, method, nfe = [], [], [], []
    for label, path in args.result:
        data = json.load(open(path))
        labels.append(label)
        base.append(float(data["base_accuracy"]) * 100)
        method.append(float(data["accuracy"]) * 100)
        nfe.append(float(data.get("avg_nfe", 64.0)))

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    width = 0.36
    axes[0].bar(x - width / 2, base, width, label="LCR", color="#457b9d")
    axes[0].bar(x + width / 2, method, width, label="Risk-gated agreement", color="#2a9d8f")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("GSM8K accuracy (%)")
    axes[0].set_title("Fixed-policy accuracy by evaluation slice")
    axes[0].legend()
    axes[0].set_ylim(0, 75)

    axes[1].bar(x - width / 2, [64] * len(labels), width, label="LCR", color="#457b9d")
    axes[1].bar(x + width / 2, nfe, width, label="Risk-gated agreement", color="#2a9d8f")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Average NFE")
    axes[1].set_title("Inference cost by evaluation slice")
    axes[1].legend()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(out)


if __name__ == "__main__":
    main()
