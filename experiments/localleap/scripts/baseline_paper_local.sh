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

declare -A TASK_SHOT_MAP=(
  [gsm8k]=5
  [minerva_math]=4
  [humaneval]=0
  [mbpp]=3
  [ifeval]=0
)

model_path=/root/autodl-tmp/model/LLaDA/instruct
block_length=32
length=$1
shift 1
SELECTED_TASKS=("$@")

log_dir=./logs/eval/llada_ins/baseline
out_dir=./results/baseline
mkdir -p $log_dir $out_dir

for task in "${SELECTED_TASKS[@]}"; do
  num_fewshot=${TASK_SHOT_MAP[$task]}
  # official baseline.sh computes steps=length/block but MODEL_ARGS uses steps=${length}
  ts=$(date +%Y%m%d_%H%M%S)
  LOG_FILE=${log_dir}/log_${ts}_${task}_len${length}_blen${block_length}_${num_fewshot}shot.log
  RUN_NAME=${task}_len${length}_blen${block_length}_${num_fewshot}shot
  MODEL_ARGS=model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},early_stop=False,show_speed=True,integrate_speed=False

  if [[ "$task" == "humaneval" ]]; then
    ADDITION_ARG="--output_path ${out_dir}/${RUN_NAME} --log_samples"
  else
    ADDITION_ARG=""
  fi

  {
    echo "=================================================="
    echo "[LLaDA Baseline] PAPER CONFIG"
    echo "MODEL: $model_path"
    echo "Running task: $task | num_fewshot: $num_fewshot"
    echo "MODEL_ARGS: $MODEL_ARGS"
    echo "NO --limit (full benchmark)"
    echo "Start time: $(date)"
    echo "=================================================="

    accelerate launch --num_processes 1 --num_machines 1 --mixed_precision no eval_llada.py \
      --model llada_dist \
      --model_args $MODEL_ARGS \
      --tasks ${task} \
      --num_fewshot ${num_fewshot} \
      --confirm_run_unsafe_code \
      $ADDITION_ARG

    if [[ "$task" == "humaneval" ]]; then
      echo "[POSTPROCESS] HumanEval sanitize+pass@1"
      SAMP=$(ls -t ${out_dir}/${RUN_NAME}/**/samples_humaneval_*.jsonl 2>/dev/null | head -1)
      if [[ -z "$SAMP" ]]; then SAMP=$(find ${out_dir}/${RUN_NAME} -name 'samples_humaneval_*.jsonl' | head -1); fi
      echo "samples=$SAMP"
      python postprocess_code.py "$SAMP" | tee -a ${LOG_FILE}.postprocess
    fi

    echo "Finished task: $task | End time: $(date)"
  } 2>&1 | tee $LOG_FILE

  sleep 60
done
