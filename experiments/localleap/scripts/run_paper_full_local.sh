#!/bin/bash
set -euo pipefail
ROOT=/root/autodl-tmp/LocalLeap
cd "$ROOT"
mkdir -p logs
echo "PAPER FULL RUN start $(date)" | tee logs/paper_full_run.meta
# official scripts/llada/run.sh order
bash scripts/llada/baseline_paper_local.sh 256 humaneval mbpp minerva_math
bash scripts/llada/baseline_paper_local.sh 512 gsm8k ifeval
bash scripts/llada/localleap_paper_local.sh 256 0.9 0.75 4 humaneval mbpp minerva_math
bash scripts/llada/localleap_paper_local.sh 512 0.9 0.75 4 gsm8k ifeval
echo "PAPER FULL RUN done $(date)" | tee -a logs/paper_full_run.meta
