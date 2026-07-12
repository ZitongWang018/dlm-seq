#!/usr/bin/env python3
"""Repair decoded-answer fields in an existing GSM8K result file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import answer_evidence, extract_number


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    path = Path(args.path)
    with open(path) as f:
        result = json.load(f)

    correct = 0
    for row in result["records"]:
        text = row.get("answer_text_tail", "")
        pred = extract_number(text)
        row["pred"] = pred
        row["correct"] = pred is not None and pred == row["gold"]
        row.update(answer_evidence(text, pred))
        correct += int(row["correct"])
    result["correct"] = correct
    result["accuracy"] = correct / len(result["records"])
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"{path}: {correct}/{len(result['records'])} = {result['accuracy']:.4f}")


if __name__ == "__main__":
    main()
