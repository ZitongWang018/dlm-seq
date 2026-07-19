#!/usr/bin/env bash
set -Eeuo pipefail

canonical=${SOURCE_ROOT:-/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability}
output_root=${OUTPUT_ROOT:-/root/autodl-tmp/LocalLeap/llada/results/experiment_queues/public_guard_mbpp_reaudit_20260719_v1}
parent_eval=${PARENT_EVAL:-/root/autodl-tmp/LocalLeap/llada/results/experiment_queues/confirmed_bidirectional_rapid_20260719_v1/evaluator_fixes/mbpp_assertions_v2}
parent_runs=${PARENT_RUNS:-/root/autodl-tmp/LocalLeap/llada_slot_confirmed_block/results/best_symmetric_benchmarks/confirmed_bidirectional_rapid_20260719_v1}
v11_queue=${V11_QUEUE:-/root/autodl-tmp/LocalLeap/llada_slot_public_guard/results/experiment_queues/public_example_guard_20260719_v1}
v11_runs=${V11_RUNS:-/root/autodl-tmp/LocalLeap/llada_slot_public_guard/results/best_symmetric_benchmarks/public_example_guard_20260719_v1}
v12_queue=${V12_QUEUE:-/root/autodl-tmp/LocalLeap/llada_slot_lazy_public_guard/results/experiment_queues/lazy_public_guard_exact_20260719_v1}
v12_runs=${V12_RUNS:-/root/autodl-tmp/LocalLeap/llada_slot_lazy_public_guard/results/best_symmetric_benchmarks/lazy_public_guard_exact_20260719_v1}
snapshot=${output_root}/source_snapshot

test ! -e "${output_root}/controller.pid"
mkdir -p "${snapshot}"
printf '%s\n' "$$" >"${output_root}/controller.pid"
exec > >(tee -a "${output_root}/controller.log") 2>&1
trap 'rc=$?; echo "reaudit_error rc=${rc} line=${LINENO}"; touch "${output_root}/FAILED"; exit "${rc}"' ERR

for file in audit_mbpp_assertions.py differential_selector.py compare_paired_task_runs.py slice_matching_task_records.py; do
  cp "${canonical}/${file}" "${snapshot}/${file}"
done
cp /root/autodl-tmp/dlm-seq-flow/experiments/localleap/humaneval_execution.py "${snapshot}/humaneval_execution.py"
(cd "${snapshot}" && sha256sum *.py >"${output_root}/frozen_evaluator_sources.sha256")

wait_for_file() {
  local path=$1 deadline=$(( $(date +%s) + 12 * 60 * 60 ))
  while [[ ! -e "${path}" ]]; do
    if (( $(date +%s) >= deadline )); then
      touch "${output_root}/FAILED_TIMEOUT" "${output_root}/FAILED"
      exit 21
    fi
    sleep 15
  done
}

sample_file() {
  local run_root=$1 matches=()
  mapfile -t matches < <(find "${run_root}/lm_eval" -type f -name 'samples_mbpp_*.jsonl' | sort)
  [[ ${#matches[@]} -eq 1 ]] || {
    echo "expected one MBPP sample log for ${run_root}; found ${#matches[@]}"
    return 22
  }
  printf '%s\n' "${matches[0]}"
}

audit_run() {
  local run_root=$1 output=$2
  test ! -e "${output}"
  PYTHONPATH="${snapshot}" /root/miniconda3/bin/python \
    "${snapshot}/audit_mbpp_assertions.py" \
    --samples "$(sample_file "${run_root}")" \
    --task-records "${run_root}/audit/task_audit_records.jsonl" \
    --output-dir "${output}"
}

pair_subset() {
  local reference_records=$1 method_records=$2 baseline_run=$3 method_run=$4 output=$5
  local subset="${output_root}/subsets/$(basename "${output}").jsonl"
  mkdir -p "$(dirname "${subset}")"
  /root/miniconda3/bin/python "${snapshot}/slice_matching_task_records.py" \
    "${method_records}" "${reference_records}" "${subset}"
  /root/miniconda3/bin/python "${snapshot}/compare_paired_task_runs.py" \
    "${reference_records}" "${subset}" \
    --baseline-config "${baseline_run}/run_config.txt" \
    --method-config "${method_run}/run_config.txt" \
    --allow-source-drift --output-dir "${output}"
}

wait_for_file "${parent_eval}/DONE"
v11_run=${v11_runs}/mbpp/mbpp_public_guard_n100
wait_for_file "${v11_run}/DONE"
audit_run "${v11_run}" "${output_root}/v11_n100"
pair_subset "${parent_eval}/method_n50/audit_records.jsonl" \
  "${output_root}/v11_n100/audit_records.jsonl" \
  "${parent_runs}/mbpp/mbpp_cbv_n50" "${v11_run}" \
  "${output_root}/paired/v11_vs_v9_first50"
pair_subset "${parent_eval}/base_n50/audit_records.jsonl" \
  "${output_root}/v11_n100/audit_records.jsonl" \
  "${parent_runs}/mbpp/mbpp_base_n50" "${v11_run}" \
  "${output_root}/paired/v11_vs_original_first50"

v12_run=${v12_runs}/mbpp/mbpp_lazy_guard_n100
wait_for_file "${v12_run}/DONE"
audit_run "${v12_run}" "${output_root}/v12_n100"
/root/miniconda3/bin/python "${snapshot}/compare_paired_task_runs.py" \
  "${output_root}/v11_n100/audit_records.jsonl" \
  "${output_root}/v12_n100/audit_records.jsonl" \
  --baseline-config "${v11_run}/run_config.txt" --method-config "${v12_run}/run_config.txt" \
  --allow-source-drift --output-dir "${output_root}/paired/v12_vs_v11_n100"

touch "${output_root}/DONE"
echo "public_guard_mbpp_reaudit_complete"
