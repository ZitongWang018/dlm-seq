#!/usr/bin/env bash
set -Eeuo pipefail

source_root=${SOURCE_ROOT:-/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability}
queue_root=${QUEUE_ROOT:-/root/autodl-tmp/LocalLeap/llada/results/experiment_queues/confirmed_bidirectional_rapid_20260719_v1}
run_base=${RUN_BASE:-/root/autodl-tmp/LocalLeap/llada_slot_confirmed_block/results/best_symmetric_benchmarks/confirmed_bidirectional_rapid_20260719_v1}
output_root=${OUTPUT_ROOT:-${queue_root}/evaluator_fixes/mbpp_assertions_v2}
method_root=${run_base}/mbpp/mbpp_cbv_n50
fast_root=${run_base}/mbpp/mbpp_fast_n50
base_root=${run_base}/mbpp/mbpp_base_n50
manifest=${output_root}/formal_manifest.tsv

mkdir -p "${output_root}"
printf '%s\n' "$$" >"${output_root}/controller.pid"
exec > >(tee -a "${output_root}/controller.log") 2>&1
trap 'rc=$?; echo "reaudit_error rc=${rc} line=${LINENO}"; touch "${output_root}/FAILED"; exit "${rc}"' ERR
[[ -s "${manifest}" ]] || printf 'stage\tstatus\n' >"${manifest}"

wait_for_file() {
  local path=$1 deadline=$(( $(date +%s) + 8 * 60 * 60 ))
  while [[ ! -e "${path}" ]]; do
    if (( $(date +%s) >= deadline )); then
      touch "${output_root}/FAILED_TIMEOUT" "${output_root}/FAILED"
      exit 21
    fi
    sleep 15
  done
}
records() {
  printf '%s\n' "$1/audit/task_audit_records.jsonl"
}
samples() {
  local run_root=$1 matches=()
  mapfile -t matches < <(find "${run_root}/lm_eval" -type f -name 'samples_mbpp_*.jsonl' | sort)
  [[ ${#matches[@]} -eq 1 ]] || {
    echo "expected exactly one MBPP sample file for ${run_root}; found ${#matches[@]}"
    return 22
  }
  printf '%s\n' "${matches[0]}"
}
reaudit() {
  local label=$1 run_root=$2 output=${output_root}/$1
  if [[ ! -s "${output}/audit_summary.json" ]]; then
    test ! -e "${output}"
    PYTHONPATH="${source_root}" /root/miniconda3/bin/python \
      "${source_root}/audit_mbpp_assertions.py" \
      --samples "$(samples "${run_root}")" \
      --task-records "$(records "${run_root}")" --output-dir "${output}"
  fi
  printf '%s\tDONE\n' "${label}" >>"${manifest}"
}

wait_for_file "${method_root}/DONE"
wait_for_file "${fast_root}/DONE"
wait_for_file "${base_root}/DONE"
reaudit method_n50 "${method_root}"
reaudit fast_n50 "${fast_root}"
reaudit base_n50 "${base_root}"

if [[ ! -d "${output_root}/paired_vs_fast" ]]; then
  /root/miniconda3/bin/python "${source_root}/compare_paired_task_runs.py" \
    "${output_root}/fast_n50/audit_records.jsonl" \
    "${output_root}/method_n50/audit_records.jsonl" \
    --baseline-config "${fast_root}/run_config.txt" \
    --method-config "${method_root}/run_config.txt" \
    --output-dir "${output_root}/paired_vs_fast"
fi
if [[ ! -d "${output_root}/paired_vs_base" ]]; then
  /root/miniconda3/bin/python "${source_root}/compare_paired_task_runs.py" \
    "${output_root}/base_n50/audit_records.jsonl" \
    "${output_root}/method_n50/audit_records.jsonl" \
    --baseline-config "${base_root}/run_config.txt" \
    --method-config "${method_root}/run_config.txt" \
    --output-dir "${output_root}/paired_vs_base"
fi
printf 'paired\tDONE\n' >>"${manifest}"
touch "${output_root}/DONE"
echo "mbpp_assertion_reaudit_complete"
