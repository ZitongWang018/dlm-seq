#!/bin/bash
set -euo pipefail
export PATH=/root/miniconda3/bin:$PATH
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

# AutoDL 翻墙（HF / datasets）
if [ -f /etc/network_turbo ]; then
  source /etc/network_turbo
  echo "[NET] network_turbo enabled"
else
  echo "[NET] WARNING: /etc/network_turbo missing"
fi

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HUB_DISABLE_XET=True
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/.cache/huggingface/hub
unset TRANSFORMERS_CACHE
# 有翻墙后走官方 HF，避免镜像/缓存 config 名不一致
unset HF_ENDPOINT || true

cd /root/autodl-tmp/LocalLeap/llada
cp -f /root/autodl-tmp/model/LLaDA/instruct/configuration_llada.py ./model/
touch ./model/__init__.py

MODEL=/root/autodl-tmp/model/LLaDA/instruct
LOGROOT=/root/autodl-tmp/LocalLeap/logs/smoke
mkdir -p "$LOGROOT"

echo "[SMOKE] ensure deps + gsm8k via HF"
python - <<'PY'
import importlib.util, subprocess, sys, os
mods = {
  "lm_eval": "lm_eval==0.4.8",
  "antlr4": "antlr4-python3-runtime==4.11",
  "einops": "einops",
  "math_verify": "math_verify",
  "immutabledict": "immutabledict",
  "langdetect": "langdetect",
  "torchmetrics": "torchmetrics",
  "sympy": "sympy",
}
miss=[pkg for mod,pkg in mods.items() if importlib.util.find_spec(mod) is None]
print("missing", miss, flush=True)
if miss:
    subprocess.check_call([sys.executable,"-m","pip","install","-q",*miss])
from datasets import load_dataset
ds=load_dataset("gsm8k","main",split="test[:2]")
print("gsm8k ok", len(ds), ds[0]["question"][:80], flush=True)
from model.modeling_llada import LLaDAModelLM
from generate import generate_localleap
print("imports ok", flush=True)
PY

run_one() {
  local name="$1"
  local args="$2"
  echo "[SMOKE] START $name $(date)"
  set +e
  accelerate launch --num_processes 1 --num_machines 1 --mixed_precision no eval_llada.py \
    --model llada_dist \
    --model_args "$args" \
    --tasks gsm8k --num_fewshot 5 --limit 2 --confirm_run_unsafe_code \
    2>&1 | tee "$LOGROOT/${name}.log"
  local ec=${PIPESTATUS[0]}
  echo "[SMOKE] ${name}_EXIT=$ec $(date)"
  set -e
  return $ec
}

BASE_ARGS="model_path=${MODEL},gen_length=256,steps=256,block_length=32,early_stop=False,show_speed=True,integrate_speed=False"
run_one baseline_gsm8k_limit2 "$BASE_ARGS" || true

STEPS=$((256/32))
LL_ARGS="model_path=${MODEL},gen_length=256,steps=${STEPS},block_length=32,threshold=0.9,relaxed_threshold=0.75,radius=4,early_stop=False,show_speed=True,integrate_speed=True"
run_one localleap_gsm8k_limit2 "$LL_ARGS" || true

echo "[SMOKE] DONE $(date)"
