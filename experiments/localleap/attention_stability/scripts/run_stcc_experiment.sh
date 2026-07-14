#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 10 ]]; then
  echo "usage: $0 TASK SHOTS STEPS MODE JSD_TAU ATTN_TAU EXTRA_MULT STREAK DIAG_MODE RUN_TAG [LIMIT]" >&2
  exit 2
fi

task=$1
shots=$2
steps=$3
mode=$4
jsd_tau=$5
attention_tau=$6
extra_multiplier=$7
streak=$8
diagnostics_mode=$9
run_tag=${10}
limit=${11:-full}

case "${task}" in
  humaneval) expected_full=164 ;;
  mbpp) expected_full=500 ;;
  minerva_math_counting_and_prob) expected_full=474 ;;
  *) echo "unsupported task: ${task}" >&2; exit 2 ;;
esac
expected_records=${expected_full}
limit_args=()
if [[ "${limit}" != "full" ]]; then
  expected_records=${limit}
  limit_args=(--limit "${limit}")
fi

queue_id=stcc_overnight_20260715_v1
queue_root=/root/autodl-tmp/LocalLeap/llada/results/experiment_queues/${queue_id}
run_root=/root/autodl-tmp/LocalLeap/llada/results/stcc/${task}/${run_tag}
trace_dir=${run_root}/trace
diagnostics_dir=${run_root}/step_diagnostics
audit_dir=${run_root}/audit
output_dir=${run_root}/lm_eval
log_file=${queue_root}/${run_tag}.log
mkdir -p "${queue_root}" "${run_root}" "${trace_dir}" "${audit_dir}"
if [[ -e "${run_root}/DONE" ]]; then
  echo "refusing to overwrite completed run: ${run_root}" >&2
  exit 3
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

diagnostics_arg=""
if [[ "${diagnostics_mode}" == "full" ]]; then
  mkdir -p "${diagnostics_dir}"
  diagnostics_arg=",stcc_diagnostics_dir=${diagnostics_dir}"
elif [[ "${diagnostics_mode}" != "trace" ]]; then
  echo "diagnostics mode must be full or trace" >&2
  exit 2
fi

model_args="model_path=/root/autodl-tmp/model/LLaDA/instruct,gen_length=256,steps=${steps},block_length=32,remasking=low_confidence,early_stop=False,show_speed=True,integrate_speed=False,stcc_mode=${mode},stcc_topk=8,stcc_jsd_threshold=${jsd_tau},stcc_attention_threshold=${attention_tau},stcc_extra_multiplier=${extra_multiplier},stcc_extra_jsd_threshold=${jsd_tau},stcc_min_topk_overlap=7,stcc_min_stability_streak=${streak},stcc_trace_dir=${trace_dir}${diagnostics_arg}"

{
  echo "schema=stcc_experiment_run_v1"
  echo "queue_id=${queue_id}"
  echo "run_tag=${run_tag}"
  echo "task=${task}"
  echo "expected_records=${expected_records}"
  echo "num_fewshot=${shots}"
  echo "seeds=0,1234,1234,1234"
  echo "steps=${steps}"
  echo "baseline_budget_per_step=$((256 / steps))"
  echo "mode=${mode}"
  echo "jsd_threshold=${jsd_tau}"
  echo "attention_threshold=${attention_tau}"
  echo "extra_multiplier=${extra_multiplier}"
  echo "min_topk_overlap=7"
  echo "min_stability_streak=${streak}"
  echo "diagnostics_mode=${diagnostics_mode}"
  echo "model_args=${model_args}"
  echo "start=$(date --iso-8601=seconds)"
  git rev-parse HEAD
  sha256sum generate.py stcc_generate.py eval_llada.py model/modeling_llada.py \
    validate_stcc_trace.py validate_stcc_diagnostics.py audit_attention_stability.py \
    postprocess_code.py sanitize.py
} > "${run_root}/run_config.txt"

set +e
accelerate launch --num_processes 1 --num_machines 1 --mixed_precision no eval_llada.py \
  --model llada_dist \
  --model_args "${model_args}" \
  --tasks "${task}" \
  --num_fewshot "${shots}" \
  --seed 0,1234,1234,1234 \
  --confirm_run_unsafe_code \
  --output_path "${output_dir}" \
  --log_samples \
  "${limit_args[@]}" \
  2>&1 | tee "${log_file}"
rc=${PIPESTATUS[0]}
set -e
if [[ ${rc} -ne 0 ]]; then
  echo "${rc}" > "${run_root}/EXITCODE"
  touch "${run_root}/FAILED"
  exit "${rc}"
fi

samples=$(find "${output_dir}" -type f -name "samples_${task}_*.jsonl" | sort | tail -1)
if [[ -z "${samples}" ]]; then
  echo "sample file not found" >&2
  touch "${run_root}/FAILED"
  exit 4
fi
trace=${trace_dir}/rank_0.jsonl
quality_args=()
if [[ "${extra_multiplier}" == "1" ]]; then
  quality_args=(--quality-nfe "${steps}")
fi
/root/miniconda3/bin/python validate_stcc_trace.py "${trace}" \
  --expected-records "${expected_records}" \
  "${quality_args[@]}" \
  --output "${audit_dir}/trace_health.json"

if [[ "${diagnostics_mode}" == "full" ]]; then
  /root/miniconda3/bin/python validate_stcc_diagnostics.py "${diagnostics_dir}" \
    --expected-files "${expected_records}" \
    --output "${audit_dir}/step_diagnostics_health.json"
fi

if [[ "${task}" == "humaneval" ]]; then
  /root/miniconda3/bin/python postprocess_code.py "${samples}" | tee "${run_root}/postprocess.txt"
  /root/miniconda3/bin/python audit_attention_stability.py \
    "${samples}" "${trace}" \
    --postprocess "${run_root}/postprocess.txt" \
    --output-dir "${audit_dir}"
  baseline_samples=/root/autodl-tmp/LocalLeap/llada/results/baseline/humaneval_len256_blen32_0shot/__root__autodl-tmp__model__LLaDA__instruct/samples_humaneval_2026-07-13T15-29-41.470324.jsonl
  /root/miniconda3/bin/python analyze_paired_humaneval.py \
    "${baseline_samples}" "${samples}" "${trace}" \
    --output-dir "${run_root}/paired_vs_baseline"
else
  results_json=$(find "${output_dir}" -type f -name 'results_*.json' | sort | tail -1)
  if [[ "${task}" == "mbpp" ]]; then
    primary_metric=pass_at_1
  else
    primary_metric=math_verify
  fi
  /root/miniconda3/bin/python audit_lm_eval_task.py \
    "${samples}" "${results_json}" \
    --task "${task}" \
    --primary-metric "${primary_metric}" \
    --trace "${trace}" \
    --expected-records "${expected_records}" \
    --output-dir "${audit_dir}"
fi

echo "0" > "${run_root}/EXITCODE"
echo "finish=$(date --iso-8601=seconds)" >> "${run_root}/run_config.txt"
touch "${run_root}/DONE"
echo "run_root=${run_root}"
