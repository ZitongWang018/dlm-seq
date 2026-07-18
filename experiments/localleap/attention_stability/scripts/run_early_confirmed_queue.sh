#!/usr/bin/env bash
set -Eeuo pipefail

source_root=${SOURCE_ROOT:-/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability_dev_early_confirmed}
llada_root=${LLADA_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_early_confirmed}
parent_queue=${PARENT_QUEUE:-/root/autodl-tmp/LocalLeap/llada/results/experiment_queues/confirmed_bidirectional_rapid_20260719_v1}
parent_run_base=${PARENT_RUN_BASE:-/root/autodl-tmp/LocalLeap/llada_slot_confirmed_block/results/best_symmetric_benchmarks/confirmed_bidirectional_rapid_20260719_v1}
queue_id=${ATTENTION_QUEUE_ID:-early_confirmed_acceleration_20260719_v1}
queue_root=${llada_root}/results/experiment_queues/${queue_id}
run_base=${llada_root}/results/best_symmetric_benchmarks/${queue_id}
runner=${llada_root}/run_best_symmetric_benchmark.sh
controller=${source_root}/scripts/run_early_confirmed_queue.sh
auditor=${source_root}/compare_exact_output_runs.py
manifest=${queue_root}/formal_manifest.tsv
frozen=${queue_root}/frozen_sources.sha256

mkdir -p "${queue_root}"
printf '%s\n' "$$" >"${queue_root}/controller.pid"
exec > >(tee -a "${queue_root}/formal_controller.log") 2>&1
trap 'rc=$?; echo "controller_error rc=${rc} line=${LINENO}"; touch "${queue_root}/FAILED"; exit "${rc}"' ERR

append_manifest() { printf '%s\n' "$1" >>"${manifest}"; }
verify() { (cd "${llada_root}" && sha256sum -c "${frozen}"); }
run_stage() {
  local label=$1 start finish rc
  shift
  verify
  start=$(date --iso-8601=seconds)
  append_manifest "$(printf '%s\tSTARTED\t%s\t\t' "${label}" "${start}")"
  set +e
  timeout --kill-after=5m 12h "$@"
  rc=$?
  set -e
  finish=$(date --iso-8601=seconds)
  verify
  if [[ ${rc} -eq 0 ]]; then
    append_manifest "$(printf '%s\tDONE\t%s\t%s\t0' "${label}" "${start}" "${finish}")"
  else
    append_manifest "$(printf '%s\tFAILED\t%s\t%s\t%s' "${label}" "${start}" "${finish}" "${rc}")"
  fi
  return "${rc}"
}
run_gpu() {
  local label=$1 gpu=$2
  shift 2
  run_stage "${label}" env CUDA_VISIBLE_DEVICES="${gpu}" LLADA_ROOT="${llada_root}" \
    ATTENTION_QUEUE_ID="${queue_id}" "$@"
}
records() {
  if [[ -s "$1/audit/audit_records.jsonl" ]]; then
    printf '%s\n' "$1/audit/audit_records.jsonl"
  elif [[ -s "$1/audit/task_audit_records.jsonl" ]]; then
    printf '%s\n' "$1/audit/task_audit_records.jsonl"
  else
    return 1
  fi
}
field() {
  /root/miniconda3/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"
}
exact_pair() {
  local label=$1 parent=$2 candidate=$3 output=${queue_root}/exact/$1
  run_stage "exact_${label}" /root/miniconda3/bin/python "${auditor}" \
    "$(records "${parent}")" "$(records "${candidate}")" --output-dir "${output}"
}
require_speedup() {
  /root/miniconda3/bin/python - "$1" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
assert summary["all_invariants_pass"]
assert summary["nfe_reduction"] > 0
PY
}

export PATH=/root/miniconda3/bin:${PATH}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_ALLOW_CODE_EVAL=1
cd "${llada_root}"
[[ "${run_base}" == */"${queue_id}" ]] || exit 20
[[ -s "${manifest}" ]] || printf 'stage\tstatus\tstart\tfinish\texit_code_or_note\n' >"${manifest}"

echo "waiting_for_parent=${parent_queue}"
deadline=$(( $(date +%s) + 18 * 60 * 60 ))
while [[ ! -e "${parent_queue}/DONE" ]]; do
  if (( $(date +%s) >= deadline )); then
    touch "${queue_root}/FAILED_PARENT_TIMEOUT" "${queue_root}/FAILED"
    exit 21
  fi
  sleep 20
done

hfast=${parent_queue}/paired/he_holdout_vs_fast/paired_summary.json
hbase=${parent_queue}/paired/he_holdout_vs_base/paired_summary.json
if [[ ! -s "${hfast}" || ! -s "${hbase}" ]]; then
  echo "parent has no formal HumanEval holdout"
  touch "${queue_root}/DONE_SKIPPED_PARENT" "${queue_root}/DONE"
  exit 0
fi
hm=$(field "${hfast}" method_correct)
hf=$(field "${hfast}" baseline_correct)
hb=$(field "${hbase}" baseline_correct)
hr=$(field "${hfast}" method_only)
if ! (( hm > hf && hm > hb && hr > 0 )); then
  echo "parent formal gate failed method=${hm} fast=${hf} base=${hb} recoveries=${hr}"
  touch "${queue_root}/DONE_SKIPPED_PARENT" "${queue_root}/DONE"
  exit 0
fi

if [[ ! -s "${frozen}" ]]; then
  sha256sum generate.py eval_llada.py run_best_symmetric_benchmark.sh \
    "${auditor}" "${controller}" \
    "${source_root}/tests/test_attention_stability.py" \
    "${source_root}/tests/test_candidate_generation_trace.py" \
    "${source_root}/test_compare_exact_output_runs.py" \
    "${source_root}/early_confirmed_preregistration_20260719.json" >"${frozen}"
fi
verify
/root/miniconda3/bin/python -m py_compile generate.py eval_llada.py "${auditor}"
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_attention_stability.py"
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_candidate_generation_trace.py"
PYTHONPATH="${source_root}" /root/miniconda3/bin/python "${source_root}/test_compare_exact_output_runs.py"
bash -n "${runner}" "${controller}"

echo "method=trajectory_early_confirmed_bidirectional_block_tau_0.004"
echo "claim=exact_v9_output_with_safe_nfe_reduction"

# Small real-model equivalence gate before the full speed audit.
run_gpu he_early_dev32 0 "${runner}" humaneval 0 128 \
  trajectory_early_confirmed_bidirectional_block 0.004 trace he_early_dev32 32 256
exact_pair he_dev32 "${parent_run_base}/humaneval/he_cbv_dev32" \
  "${run_base}/humaneval/he_early_dev32"
require_speedup "${queue_root}/exact/he_dev32/exact_summary.json"

# Once the prefix is exact, use both GPUs on HumanEval and MATH speed validation.
run_gpu he_early_full164 0 "${runner}" humaneval 0 128 \
  trajectory_early_confirmed_bidirectional_block 0.004 trace he_early_full164 164 256 & p0=$!
math_parent=${parent_run_base}/localleap_math500/math_cbv_n50
if [[ -e "${math_parent}/DONE" ]]; then
  run_gpu math_early_n50 1 "${runner}" localleap_math500 0 128 \
    trajectory_early_confirmed_bidirectional_block 0.004 trace math_early_n50 50 256 & p1=$!
else
  p1=
fi
wait "${p0}"
[[ -z "${p1}" ]] || wait "${p1}"
exact_pair he_full164 "${parent_run_base}/humaneval/he_cbv_full164" \
  "${run_base}/humaneval/he_early_full164"
require_speedup "${queue_root}/exact/he_full164/exact_summary.json"
if [[ -n "${p1}" ]]; then
  exact_pair math_n50 "${math_parent}" "${run_base}/localleap_math500/math_early_n50"
  require_speedup "${queue_root}/exact/math_n50/exact_summary.json"
fi

# Recheck exactness on every generalization arm that the parent unlocked.
mbpp_parent=${parent_run_base}/mbpp/mbpp_cbv_n50
gsm_parent=${parent_run_base}/gsm8k/gsm_cbv_n64
if [[ -e "${mbpp_parent}/DONE" || -e "${gsm_parent}/DONE" ]]; then
  if [[ -e "${mbpp_parent}/DONE" ]]; then
    run_gpu mbpp_early_n50 0 "${runner}" mbpp 0 128 \
      trajectory_early_confirmed_bidirectional_block 0.004 trace mbpp_early_n50 50 256 & p0=$!
  else
    p0=
  fi
  if [[ -e "${gsm_parent}/DONE" ]]; then
    run_gpu gsm_early_n64 1 "${runner}" gsm8k 0 128 \
      trajectory_early_confirmed_bidirectional_block 0.004 trace gsm_early_n64 64 256 & p1=$!
  else
    p1=
  fi
  [[ -z "${p0}" ]] || wait "${p0}"
  [[ -z "${p1}" ]] || wait "${p1}"
  if [[ -n "${p0}" ]]; then
    exact_pair mbpp_n50 "${mbpp_parent}" "${run_base}/mbpp/mbpp_early_n50"
    require_speedup "${queue_root}/exact/mbpp_n50/exact_summary.json"
  fi
  if [[ -n "${p1}" ]]; then
    exact_pair gsm_n64 "${gsm_parent}" "${run_base}/gsm8k/gsm_early_n64"
    require_speedup "${queue_root}/exact/gsm_n64/exact_summary.json"
  fi
fi

touch "${queue_root}/DONE"
echo "queue_complete"
