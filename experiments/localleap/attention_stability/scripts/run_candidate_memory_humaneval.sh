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
selection_mode=${1:?selection mode required: confidence, frontier, or impact}
candidate_topk=${2:-8}
confidence_threshold=${3:-0}
limit=${4:-full}
run_tag=${5:-run_$(date +%Y%m%d_%H%M%S)}

case "${selection_mode}" in
  confidence|frontier|impact) ;;
  *) echo "ERROR: invalid selection mode ${selection_mode}" >&2; exit 2 ;;
esac

# A full run first proves the exact source hash on one HumanEval task through
# generation, sanitize/code_eval, record audit, and the step validator.
if [[ "${limit}" == "full" && "${CANDIDATE_PREFLIGHT_DONE:-0}" != "1" ]]; then
  preflight_tag=${run_tag}_preflight1
  preflight_root=results/candidate_memory/${selection_mode}/k${candidate_topk}_delta${confidence_threshold//./p}/${preflight_tag}
  if [[ ! -e "${preflight_root}/DONE" ]]; then
    echo "[PREFLIGHT] starting ${preflight_tag}"
    CANDIDATE_PREFLIGHT_DONE=1 "$0" \
      "${selection_mode}" "${candidate_topk}" "${confidence_threshold}" 1 "${preflight_tag}"
  else
    echo "[PREFLIGHT] reusing completed ${preflight_tag}"
  fi
fi

model_path=/root/autodl-tmp/model/LLaDA/instruct
gen_length=256
steps=256
block_length=32
num_fewshot=0
delta_tag=${confidence_threshold//./p}
run_root=results/candidate_memory/${selection_mode}/k${candidate_topk}_delta${delta_tag}/${run_tag}
trace_dir=${run_root}/trace
diagnostics_dir=${run_root}/step_diagnostics
output_dir=${run_root}/humaneval_len256_blen32_0shot
mkdir -p "${trace_dir}" "${diagnostics_dir}" "${output_dir}"

if [[ -e "${run_root}/DONE" ]]; then
  echo "ERROR: refusing to overwrite completed run ${run_root}" >&2
  exit 3
fi
if [[ -s "${trace_dir}/rank_0.jsonl" ]] || find "${diagnostics_dir}" -name '*.pt' -print -quit | grep -q .; then
  echo "ERROR: refusing to overwrite partial run ${run_root}" >&2
  exit 4
fi

model_args=model_path=${model_path},gen_length=${gen_length},steps=${steps},block_length=${block_length},early_stop=False,show_speed=True,integrate_speed=False,candidate_memory_topk=${candidate_topk},candidate_memory_confidence_threshold=${confidence_threshold},candidate_memory_fallback=${selection_mode},candidate_memory_exact_jsd=False,candidate_memory_trace_dir=${trace_dir},candidate_memory_diagnostics_dir=${diagnostics_dir}
limit_args=()
if [[ "${limit}" != "full" ]]; then
  limit_args=(--limit "${limit}")
fi

{
  echo "decoder=candidate_memory_stability_v2"
  echo "candidate_memory_exact_jsd=False"
  echo "selection_mode=${selection_mode}"
  echo "candidate_topk=${candidate_topk}"
  echo "confidence_threshold=${confidence_threshold}"
  echo "model_args=${model_args}"
  echo "task=humaneval num_fewshot=${num_fewshot} limit=${limit}"
  echo "seeds=lm_eval:0,numpy:1234,torch:1234,fewshot:1234"
  echo "start=$(date --iso-8601=seconds)"
  if git -C /root/autodl-tmp/LocalLeap rev-parse HEAD >/dev/null 2>&1; then
    echo "localleap_git_sha=$(git -C /root/autodl-tmp/LocalLeap rev-parse HEAD)"
    echo "localleap_git_status_begin"
    git -C /root/autodl-tmp/LocalLeap status --short
    echo "localleap_git_status_end"
  fi
  sha256sum generate.py eval_llada.py model/modeling_llada.py \
    validate_candidate_memory_diagnostics.py audit_attention_stability.py \
    postprocess_code.py sanitize.py \
    /root/autodl-tmp/LocalLeap/scripts/llada/run_candidate_memory_humaneval.sh
} | tee "${run_root}/run_config.txt"

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
  exit 5
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
  exit 6
fi
python validate_candidate_memory_diagnostics.py \
  "${diagnostics_dir}" \
  --expected-count "${trace_count}" \
  --output "${run_root}/audit/step_diagnostics_summary.json"
echo "step_diagnostics_count=${diagnostics_count}"
echo "finish=$(date --iso-8601=seconds)"
touch "${run_root}/DONE"
