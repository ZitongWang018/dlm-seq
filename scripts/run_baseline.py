#!/usr/bin/env python3
"""Run baseline LCR on GSM8K and MBPP."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runner import run_experiment

if __name__ == "__main__":
    run_experiment(
        config_path=str(ROOT / "configs" / "default.yaml"),
        methods=["lcr"],
        datasets=["gsm8k", "mbpp"],
        run_tag="round0_baseline",
    )
