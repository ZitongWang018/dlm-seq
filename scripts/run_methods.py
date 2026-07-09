#!/usr/bin/env python3
"""Run full method comparison: LCR, RCR, Traj, Traj+Lateral."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runner import run_experiment

if __name__ == "__main__":
    run_tag = sys.argv[1] if len(sys.argv) > 1 else "round2_methods"
    run_experiment(
        config_path=str(ROOT / "configs" / "default.yaml"),
        methods=["lcr", "rcr", "traj", "traj_lateral"],
        datasets=["gsm8k", "mbpp"],
        run_tag=run_tag,
    )
