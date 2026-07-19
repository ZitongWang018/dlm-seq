#!/usr/bin/env bash
set -Eeuo pipefail

source_root=${SOURCE_ROOT:-/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability_dev_lazy_public_guard}
llada_root=${LLADA_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_lazy_public_guard}
parent_queue=${PARENT_QUEUE:-/root/autodl-tmp/LocalLeap/llada_slot_public_guard/results/experiment_queues/public_example_guard_20260719_v1}
parent_run_base=${PARENT_RUN_BASE:-/root/autodl-tmp/LocalLeap/llada_slot_public_guard/results/best_symmetric_benchmarks/public_example_guard_20260719_v1}
queue_id=${ATTENTION_QUEUE_ID:-lazy_public_guard_exact_20260719_v1}
queue_root=${llada_root}/results/experiment_queues/${queue_id}
run_base=${llada_root}/results/best_symmetric_benchmarks/${queue_id}
runner=${llada_root}/run_best_symmetric_benchmark.sh
controller=${source_root}/scripts/run_lazy_public_guard_queue.sh
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
wait_for_file() {
  local path=$1 deadline=$(( $(date +%s) + 18 * 60 * 60 ))
  while [[ ! -e "${path}" ]]; do
    if (( $(date +%s) >= deadline )); then
      touch "${queue_root}/FAILED_PARENT_TIMEOUT" "${queue_root}/FAILED"
      exit 21
    fi
    sleep 15
  done
}
field() {
  /root/miniconda3/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"
}

export PATH=/root/miniconda3/bin:${PATH}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_ALLOW_CODE_EVAL=1
cd "${llada_root}"
[[ "${run_base}" == */"${queue_id}" ]] || exit 20
[[ -s "${manifest}" ]] || printf 'stage\tstatus\tstart\tfinish\texit_code_or_note\n' >"${manifest}"

echo "waiting_for_v11_integrated_parent"
wait_for_file "${parent_queue}/DONE"
wait_for_file "${parent_run_base}/humaneval/he_public_guard_smoke4/DONE"
wait_for_file "${parent_run_base}/mbpp/mbpp_public_guard_n100/DONE"

if [[ ! -s "${frozen}" ]]; then
  sha256sum generate.py eval_llada.py differential_selector.py \
    run_best_symmetric_benchmark.sh compare_exact_output_runs.py \
    postprocess_code.py humaneval_execution.py sanitize.py \
    "${controller}" "${source_root}/test_public_example_guard.py" \
    "${source_root}/tests/test_attention_stability.py" \
    "${source_root}/tests/test_candidate_generation_trace.py" \
    "${source_root}/lazy_public_guard_preregistration_20260719.json" >"${frozen}"
fi
verify
/root/miniconda3/bin/python -m py_compile generate.py eval_llada.py \
  differential_selector.py compare_exact_output_runs.py
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/test_public_example_guard.py"
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_attention_stability.py"
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_candidate_generation_trace.py"
bash -n "${runner}" "${controller}"

run_gpu he_lazy_guard_smoke4 0 "${runner}" humaneval 0 128 \
  trajectory_lazy_confirmed_public_guard 0.004 trace he_lazy_guard_smoke4 4 256 & p0=$!
run_gpu mbpp_lazy_guard_n100 1 "${runner}" mbpp 0 128 \
  trajectory_lazy_confirmed_public_guard 0.004 trace mbpp_lazy_guard_n100 100 256 & p1=$!
wait "${p0}"
wait "${p1}"

he_parent=${parent_run_base}/humaneval/he_public_guard_smoke4
he_lazy=${run_base}/humaneval/he_lazy_guard_smoke4
mbpp_parent=${parent_run_base}/mbpp/mbpp_public_guard_n100
mbpp_lazy=${run_base}/mbpp/mbpp_lazy_guard_n100
run_stage he_exact_output /root/miniconda3/bin/python compare_exact_output_runs.py \
  "$(records "${he_parent}")" "$(records "${he_lazy}")" \
  --output-dir "${queue_root}/exact/he_smoke4"
run_stage mbpp_exact_output /root/miniconda3/bin/python compare_exact_output_runs.py \
  "$(records "${mbpp_parent}")" "$(records "${mbpp_lazy}")" \
  --output-dir "${queue_root}/exact/mbpp_n100"

mbpp_summary=${queue_root}/exact/mbpp_n100/exact_summary.json
mbpp_saved=$(field "${mbpp_summary}" nfe_reduction)
if (( mbpp_saved > 0 )); then
  touch "${queue_root}/SPEED_PASS"
else
  touch "${queue_root}/SPEED_FAIL" "${queue_root}/FAILED"
  exit 22
fi
touch "${queue_root}/DONE"
echo "lazy_public_guard_exact_queue_complete nfe_saved=${mbpp_saved}"
