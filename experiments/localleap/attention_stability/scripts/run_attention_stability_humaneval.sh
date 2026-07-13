#!/bin/bash
set -euo pipefail

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
tau=${1:?dependency threshold required}
limit=${2:-full}
run_tag=${3:-run_$(date +%Y%m%d_%H%M%S)}

model_path=/root/autodl-tmp/model/LLaDA/instruct
gen_length=256
steps=256
block_length=32
num_fewshot=0
run_root=results/attention_stability/tau${tau}/${run_tag}
trace_dir=${run_root}/trace
diagnostics_dir=${run_root}/step_diagnostics
output_dir=${run_root}/humaneval_len256_blen32_0shot
mkdir -p "${trace_dir}" "${diagnostics_dir}" "${output_dir}"

model_args=model_path=${model_path},gen_length=${gen_length},steps=${steps},block_length=${block_length},early_stop=False,show_speed=True,integrate_speed=False,dependency_threshold=${tau},dependency_trace_dir=${trace_dir},dependency_diagnostics_dir=${diagnostics_dir}
limit_args=()
if [[ "${limit}" != "full" ]]; then
  limit_args=(--limit "${limit}")
fi

echo "decoder=attention_stability_v1"
echo "model_args=${model_args}"
echo "task=humaneval num_fewshot=${num_fewshot} limit=${limit}"
echo "seeds=lm_eval:0,numpy:1234,torch:1234,fewshot:1234"
echo "start=$(date --iso-8601=seconds)"

accelerate launch --num_processes 1 --num_machines 1 --mixed_precision no eval_llada.py \
  --model llada_dist \
  --model_args "${model_args}" \
  --tasks humaneval \
  --num_fewshot "${num_fewshot}" \
  --confirm_run_unsafe_code \
  --output_path "${output_dir}" \
  --log_samples \
  "${limit_args[@]}"

samples=$(find "${output_dir}" -name 'samples_humaneval_*.jsonl' | sort | tail -1)
if [[ -z "${samples}" ]]; then
  echo "ERROR: no HumanEval samples file" >&2
  exit 2
fi
echo "samples=${samples}"
python postprocess_code.py "${samples}" | tee "${run_root}/postprocess.txt"
python audit_attention_stability.py \
  "${samples}" \
  "${trace_dir}/rank_0.jsonl" \
  --postprocess "${run_root}/postprocess.txt" \
  --output-dir "${run_root}/audit"
trace_count=$(wc -l < "${trace_dir}/rank_0.jsonl")
diagnostics_count=$(find "${diagnostics_dir}" -maxdepth 1 -type f -name '*.pt' | wc -l)
if [[ "${trace_count}" -ne "${diagnostics_count}" ]]; then
  echo "ERROR: trace/step-diagnostics count mismatch: ${trace_count} vs ${diagnostics_count}" >&2
  exit 3
fi
python validate_step_diagnostics.py \
  "${diagnostics_dir}" \
  --expected-count "${trace_count}" \
  --output "${run_root}/audit/step_diagnostics_summary.json"
echo "step_diagnostics_count=${diagnostics_count}"
echo "finish=$(date --iso-8601=seconds)"
touch "${run_root}/DONE"
