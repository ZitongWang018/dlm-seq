#!/bin/bash
set -x
source /etc/network_turbo
export PATH=/root/miniconda3/bin:$PATH
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HUB_DISABLE_XET=True
export HF_HOME=/root/autodl-tmp/.cache/huggingface
unset TRANSFORMERS_CACHE
unset HF_ENDPOINT || true

cd /root/autodl-tmp/LocalLeap/llada
cp -f /root/autodl-tmp/model/LLaDA/instruct/configuration_llada.py ./model/ || true
touch ./model/__init__.py

model_path=/root/autodl-tmp/model/LLaDA/instruct
# Paper localleap.sh for HumanEval:
# length=256, kappa=0.9, tau=0.75, radius=4, block=32, steps=length/block=8, fewshot=0
length=256
block_length=32
steps=$((length / block_length))
threshold=0.9
relaxed_threshold=0.75
radius=4
num_fewshot=0
task=humaneval

log_dir=./logs/eval/llada_ins/localleap/anchor${threshold}_relax${relaxed_threshold}_radius${radius}
out_dir=./results/localleap/anchor${threshold}_relax${relaxed_threshold}_radius${radius}
mkdir -p "$log_dir" "$out_dir"
ts=$(date +%Y%m%d_%H%M%S)
RUN_NAME=${task}_len${length}_blen${block_length}_${num_fewshot}shot
LOG_FILE=${log_dir}/log_${ts}_${RUN_NAME}.log
MODEL_ARGS=model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},threshold=${threshold},relaxed_threshold=${relaxed_threshold},radius=${radius},early_stop=False,show_speed=True,integrate_speed=True

{
  echo "=================================================="
  echo "[LLaDA LocalLeap] HumanEval PAPER CONFIG ONLY"
  echo "MODEL_ARGS=$MODEL_ARGS"
  echo "NO --limit (full 164)"
  echo "Start: $(date)"
  echo "=================================================="

  accelerate launch --num_processes 1 --num_machines 1 --mixed_precision no eval_llada.py \
    --model llada_dist \
    --model_args "$MODEL_ARGS" \
    --tasks humaneval \
    --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code \
    --output_path ${out_dir}/${RUN_NAME} \
    --log_samples

  echo "[POSTPROCESS] start $(date)"
  SAMP=$(find ${out_dir}/${RUN_NAME} -name 'samples_humaneval_*.jsonl' | head -1)
  echo "samples=$SAMP"
  python postprocess_code.py "$SAMP" | tee ${LOG_FILE}.postprocess
  echo "[POSTPROCESS] done $(date)"
  echo "Finished LocalLeap HumanEval $(date)"
} 2>&1 | tee "$LOG_FILE"

echo DONE > ${log_dir}/DONE_humaneval_${ts}.txt
