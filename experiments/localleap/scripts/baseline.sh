#!/bin/bash
set -x
# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HUB_DISABLE_XET=True
# ======================================================================
cd llada
# ======================================================================
declare -A TASK_SHOT_MAP=(
    [gsm8k]=5
    [minerva_math]=4
    [humaneval]=0
    [mbpp]=3
    [ifeval]=0
)
# ======================================================================
model_path=GSAI-ML/LLaDA-8B-Instruct

block_length=32
length=$1

shift 1
SELECTED_TASKS=("$@")
# ======================================================================
log_dir=./logs/eval/llada_ins/baseline
out_dir=./results/baseline
mkdir -p $log_dir
# ======================================================================
for task in "${SELECTED_TASKS[@]}"; do
    num_fewshot=${TASK_SHOT_MAP[$task]}
    steps=$((length / block_length))

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
        echo "[LLaDA Baseline] Inference"
        echo "MODEL: $model_path"
        echo "Running task: $task | num_fewshot: $num_fewshot"
        echo "MODEL_ARGS: $MODEL_ARGS"
        echo "Start time: $(date)"
        echo "=================================================="

        accelerate launch eval_llada.py \
            --model llada_dist \
            --model_args $MODEL_ARGS \
            --tasks ${task} \
            --num_fewshot ${num_fewshot} \
            --confirm_run_unsafe_code \
            $ADDITION_ARG

        echo "--------------------------------------------------"
        echo "Finished task: $task | End time: $(date)"
        echo "--------------------------------------------------"
    } 2>&1 | tee $LOG_FILE

    sleep 60
done


