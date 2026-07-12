#!/usr/bin/env python3
"""Check whether lateral and longitudinal trajectory signals replicate across slices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SIGNALS = {
    "Lateral response": "response_selected_delta_mean",
    "Longitudinal flips": "answer_flip_mean",
}


def load_rows(path: str) -> list[dict]:
    return json.load(open(path))["records"]


def value(row: dict, key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def error_auc(rows: list[dict], key: str) -> float:
    wrong = [value(row, key) for row in rows if not row["correct"]]
    correct = [value(row, key) for row in rows if row["correct"]]
    comparisons = [(a > b) + 0.5 * (a == b) for a in wrong for b in correct]
    return float(np.mean(comparisons))


def threshold(rows: list[dict], key: str, quantile: float) -> float:
    return float(np.quantile([value(row, key) for row in rows], quantile))


def group_summary(rows: list[dict], key: str, cutoff: float) -> dict:
    selected = [row for row in rows if value(row, key) >= cutoff]
    return {
        "count": len(selected),
        "rate": len(selected) / len(rows),
        "accuracy": sum(row["correct"] for row in selected) / len(selected),
        "error_rate": 1.0 - sum(row["correct"] for row in selected) / len(selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--quantile", type=float, default=0.8)
    args = parser.parse_args()

    calibration = load_rows(args.calibration)
    test = load_rows(args.test)
    output = {
        "calibration_size": len(calibration),
        "test_size": len(test),
        "quantile": args.quantile,
        "signals": {},
    }
    for label, key in SIGNALS.items():
        cutoff = threshold(calibration, key, args.quantile)
        output["signals"][label] = {
            "key": key,
            "calibration_threshold": cutoff,
            "calibration_error_auc": error_auc(calibration, key),
            "test_error_auc": error_auc(test, key),
            "calibration_high_signal": group_summary(calibration, key, cutoff),
            "test_high_signal": group_summary(test, key, cutoff),
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "trajectory_risk_replication.json", "w") as handle:
        json.dump(output, handle, indent=2)

    labels = list(SIGNALS)
    x = np.arange(len(labels))
    width = 0.34
    cal_auc = [output["signals"][label]["calibration_error_auc"] for label in labels]
    test_auc = [output["signals"][label]["test_error_auc"] for label in labels]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.bar(x - width / 2, cal_auc, width, label="Calibration slice", color="#4C78A8")
    ax.bar(x + width / 2, test_auc, width, label="Test slice", color="#E45756")
    ax.axhline(0.5, color="#444444", linestyle="--", linewidth=1, label="Random ranking")
    ax.set_ylabel("Error-ranking AUC")
    ax.set_title("Replication of trajectory-based risk signals")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.4, 0.75)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_risk_replication.png", dpi=180)
    plt.close(fig)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
