#!/usr/bin/env bash
# Full experiment pipeline for dlm-seq-flow
set -euo pipefail
cd /root/autodl-tmp/dlm-seq-flow
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export PYTHONPATH=/root/autodl-tmp/dlm-seq-flow

mkdir -p results

echo "[1/3] Observation study (10 samples)..."
python3 scripts/run_observation.py 2>&1 | tee results/round1_observation.log

echo "[2/3] Method comparison..."
python3 scripts/run_methods.py round2_methods 2>&1 | tee results/round2_methods.log

echo "[3/3] MBPP baseline rerun with fixed steps..."
python3 - <<'PY'
import sys
sys.path.insert(0, ".")
from src.runner import run_experiment
run_experiment("configs/default.yaml", ["lcr"], ["mbpp"], "round0_baseline_mbpp_fixed")
PY

echo "Done."
