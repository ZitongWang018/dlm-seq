#!/usr/bin/env bash
set -Eeuo pipefail

source_root=${SOURCE_ROOT:-/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability_dev_outcome_arbiter}
llada_root=${LLADA_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_outcome_arbiter}
predecessor_queue=${PREDECESSOR_QUEUE:-/root/autodl-tmp/LocalLeap/llada_slot_public_verifier/results/experiment_queues/public_full_draft_rapid_20260719_v1}
parent_queue=${PARENT_QUEUE:-/root/autodl-tmp/LocalLeap/llada/results/experiment_queues/confirmed_bidirectional_rapid_20260719_v1}
parent_run_base=${PARENT_RUN_BASE:-/root/autodl-tmp/LocalLeap/llada_slot_confirmed_block/results/best_symmetric_benchmarks/confirmed_bidirectional_rapid_20260719_v1}
queue_id=${ATTENTION_QUEUE_ID:-outcome_arbiter_rapid_20260719_v1}
queue_root=${llada_root}/results/experiment_queues/${queue_id}
run_base=${llada_root}/results/best_symmetric_benchmarks/${queue_id}
runner=${llada_root}/run_best_symmetric_benchmark.sh
controller=${source_root}/scripts/run_outcome_arbiter_queue.sh
manifest=${queue_root}/formal_manifest.tsv
frozen=${queue_root}/frozen_sources.sha256
parent_frozen=${parent_queue}/frozen_sources.sha256

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
  local path=$1 deadline=$(( $(date +%s) + 30 * 60 * 60 ))
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

echo "waiting_for_humaneval_accuracy_queue"
wait_for_file "${predecessor_queue}/DONE"
wait_for_file "${parent_queue}/DONE"
math_parent=${parent_run_base}/localleap_math500/math_cbv_n50
gsm_parent=${parent_run_base}/gsm8k/gsm_cbv_n64
wait_for_file "${math_parent}/DONE"
wait_for_file "${gsm_parent}/DONE"

if [[ ! -s "${frozen}" ]]; then
  sha256sum generate.py eval_llada.py differential_selector.py outcome_arbiter.py \
    run_best_symmetric_benchmark.sh compare_paired_task_runs.py \
    slice_audit_by_index.py postprocess_code.py humaneval_execution.py sanitize.py \
    tasks/localleap_math500/utils.py "${controller}" \
    "${source_root}/test_outcome_arbiter.py" \
    "${source_root}/tests/test_attention_stability.py" \
    "${source_root}/tests/test_candidate_generation_trace.py" \
    "${source_root}/tests/test_compare_paired_task_runs.py" \
    "${source_root}/outcome_arbiter_preregistration_20260719.json" >"${frozen}"
fi
verify
/root/miniconda3/bin/python -m py_compile generate.py eval_llada.py \
  differential_selector.py outcome_arbiter.py compare_paired_task_runs.py
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/test_outcome_arbiter.py"
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_attention_stability.py"
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_candidate_generation_trace.py"
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_compare_paired_task_runs.py"
bash -n "${runner}" "${controller}"

mkdir -p "${queue_root}/slices"
/root/miniconda3/bin/python slice_audit_by_index.py "$(records "${math_parent}")" \
  "${queue_root}/slices/math_parent_0_16.jsonl" --start 0 --end 16
/root/miniconda3/bin/python slice_audit_by_index.py "$(records "${gsm_parent}")" \
  "${queue_root}/slices/gsm_parent_0_16.jsonl" --start 0 --end 16

run_gpu math_dev_n16 0 "${runner}" localleap_math500 0 128 \
  trajectory_outcome_arbiter 0.004 trace math_dev_n16 16 256 & p0=$!
run_gpu gsm_dev_n16 1 "${runner}" gsm8k 0 128 \
  trajectory_outcome_arbiter 0.004 trace gsm_dev_n16 16 256 & p1=$!
wait "${p0}"
wait "${p1}"

math_dev=${run_base}/localleap_math500/math_dev_n16
gsm_dev=${run_base}/gsm8k/gsm_dev_n16
run_stage compare_math_dev /root/miniconda3/bin/python compare_paired_task_runs.py \
  "${queue_root}/slices/math_parent_0_16.jsonl" "$(records "${math_dev}")" \
  --baseline-config "${parent_frozen}" --method-config "${math_dev}/run_config.txt" \
  --method-log "${math_dev}/run.log" --allow-source-drift \
  --output-dir "${queue_root}/paired/math_dev"
run_stage compare_gsm_dev /root/miniconda3/bin/python compare_paired_task_runs.py \
  "${queue_root}/slices/gsm_parent_0_16.jsonl" "$(records "${gsm_dev}")" \
  --baseline-config "${parent_frozen}" --method-config "${gsm_dev}/run_config.txt" \
  --method-log "${gsm_dev}/run.log" --allow-source-drift \
  --output-dir "${queue_root}/paired/gsm_dev"

eligible=()
for task in math gsm; do
  summary=${queue_root}/paired/${task}_dev/paired_summary.json
  parent_correct=$(field "${summary}" baseline_correct)
  method_correct=$(field "${summary}" method_correct)
  parent_only=$(field "${summary}" baseline_only)
  method_only=$(field "${summary}" method_only)
  echo "${task}_dev parent=${parent_correct} method=${method_correct} method_only=${method_only} parent_only=${parent_only}"
  if (( method_correct > parent_correct && method_only > parent_only )); then
    eligible+=("${task}")
    touch "${queue_root}/DEV_PASS_${task}"
  else
    touch "${queue_root}/DEV_REJECT_${task}"
  fi
done
if (( ${#eligible[@]} == 0 )); then
  touch "${queue_root}/DEV_REJECTED" "${queue_root}/DONE"
  echo "outcome_arbiter_no_dev_improvement"
  exit 0
fi

pids=()
for task in "${eligible[@]}"; do
  if [[ "${task}" == math ]]; then
    run_gpu math_formal_n50 0 "${runner}" localleap_math500 0 128 \
      trajectory_outcome_arbiter 0.004 trace math_formal_n50 50 256 &
  else
    run_gpu gsm_formal_n64 1 "${runner}" gsm8k 0 128 \
      trajectory_outcome_arbiter 0.004 trace gsm_formal_n64 64 256 &
  fi
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done

formal_pass=0
for task in "${eligible[@]}"; do
  if [[ "${task}" == math ]]; then
    parent_root=${math_parent}
    method_root=${run_base}/localleap_math500/math_formal_n50
    start=25
    end=50
  else
    parent_root=${gsm_parent}
    method_root=${run_base}/gsm8k/gsm_formal_n64
    start=32
    end=64
  fi
  /root/miniconda3/bin/python slice_audit_by_index.py "$(records "${parent_root}")" \
    "${queue_root}/slices/${task}_parent_holdout.jsonl" --start "${start}" --end "${end}"
  /root/miniconda3/bin/python slice_audit_by_index.py "$(records "${method_root}")" \
    "${queue_root}/slices/${task}_method_holdout.jsonl" --start "${start}" --end "${end}"
  run_stage "compare_${task}_holdout" /root/miniconda3/bin/python compare_paired_task_runs.py \
    "${queue_root}/slices/${task}_parent_holdout.jsonl" \
    "${queue_root}/slices/${task}_method_holdout.jsonl" \
    --baseline-config "${parent_frozen}" --method-config "${method_root}/run_config.txt" \
    --method-log "${method_root}/run.log" --allow-source-drift \
    --output-dir "${queue_root}/paired/${task}_holdout"
  summary=${queue_root}/paired/${task}_holdout/paired_summary.json
  parent_correct=$(field "${summary}" baseline_correct)
  method_correct=$(field "${summary}" method_correct)
  parent_only=$(field "${summary}" baseline_only)
  method_only=$(field "${summary}" method_only)
  echo "${task}_holdout parent=${parent_correct} method=${method_correct} method_only=${method_only} parent_only=${parent_only}"
  if (( method_correct > parent_correct && method_only > 0 && parent_only <= method_only )); then
    touch "${queue_root}/FORMAL_PASS_${task}"
    formal_pass=1
  else
    touch "${queue_root}/FORMAL_REJECT_${task}"
  fi
done
if (( formal_pass )); then
  touch "${queue_root}/FORMAL_PASS"
else
  touch "${queue_root}/FORMAL_REJECTED"
fi
touch "${queue_root}/DONE"
echo "outcome_arbiter_queue_complete formal_pass=${formal_pass}"
