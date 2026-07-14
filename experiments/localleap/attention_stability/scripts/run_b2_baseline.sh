#!/usr/bin/env bash
set -euo pipefail

task=$1
shots=$2
run_tag=$3
case "${task}" in
  mbpp) expected_records=500 ;;
  minerva_math_counting_and_prob) expected_records=474 ;;
  *) echo "unsupported b2 task: ${task}" >&2; exit 2 ;;
esac

queue_id=stcc_overnight_20260715_v1
queue_root=/root/autodl-tmp/LocalLeap/llada/results/experiment_queues/${queue_id}
run_root=/root/autodl-tmp/LocalLeap/llada/results/b2_confirmatory/${task}/baseline/${run_tag}
log_file=${queue_root}/${run_tag}.log
mkdir -p "${queue_root}" "${run_root}"
if [[ -e "${run_root}/DONE" ]]; then
  echo "already complete: ${run_root}"
  exit 0
fi

export PATH=/root/miniconda3/bin:${PATH}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
unset TRANSFORMERS_CACHE || true
unset HF_ENDPOINT || true
cd /root/autodl-tmp/LocalLeap/llada
model_args="model_path=/root/autodl-tmp/model/LLaDA/instruct,gen_length=256,steps=128,block_length=32,remasking=low_confidence,early_stop=False,show_speed=True,integrate_speed=False"

{
  echo "schema=stcc_b2_baseline_v1"
  echo "run_tag=${run_tag}"
  echo "task=${task}"
  echo "expected_records=${expected_records}"
  echo "num_fewshot=${shots}"
  echo "seeds=0,1234,1234,1234"
  echo "expected_nfe_per_task=128"
  echo "expected_budget_per_step=2"
  echo "model_args=${model_args}"
  echo "start=$(date --iso-8601=seconds)"
  git rev-parse HEAD
  sha256sum generate.py eval_llada.py model/modeling_llada.py
} > "${run_root}/run_config.txt"

set +e
accelerate launch --num_processes 1 --num_machines 1 --mixed_precision no eval_llada.py \
  --model llada_dist \
  --model_args "${model_args}" \
  --tasks "${task}" \
  --num_fewshot "${shots}" \
  --seed 0,1234,1234,1234 \
  --confirm_run_unsafe_code \
  --output_path "${run_root}/lm_eval" \
  --log_samples \
  2>&1 | tee "${log_file}"
rc=${PIPESTATUS[0]}
set -e
echo "${rc}" > "${run_root}/EXITCODE"
echo "finish=$(date --iso-8601=seconds)" >> "${run_root}/run_config.txt"
if [[ ${rc} -eq 0 ]]; then
  samples=$(find "${run_root}/lm_eval" -type f -name "samples_${task}_*.jsonl" | sort | tail -1)
  results_json=$(find "${run_root}/lm_eval" -type f -name 'results_*.json' | sort | tail -1)
  if [[ "${task}" == "mbpp" ]]; then primary_metric=pass_at_1; else primary_metric=math_verify; fi
  /root/miniconda3/bin/python audit_lm_eval_task.py \
    "${samples}" "${results_json}" \
    --task "${task}" \
    --primary-metric "${primary_metric}" \
    --expected-records "${expected_records}" \
    --output-dir "${run_root}/audit"
  touch "${run_root}/DONE"
else
  touch "${run_root}/FAILED"
fi
exit "${rc}"
