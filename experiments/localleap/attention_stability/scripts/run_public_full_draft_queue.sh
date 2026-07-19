#!/usr/bin/env bash
set -Eeuo pipefail

source_root=${SOURCE_ROOT:-/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability_dev_public_verifier}
llada_root=${LLADA_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_public_verifier}
speed_queue=${SPEED_QUEUE:-/root/autodl-tmp/LocalLeap/llada_slot_lazy_public_guard/results/experiment_queues/lazy_public_guard_exact_20260719_v1}
parent_queue=${PARENT_QUEUE:-/root/autodl-tmp/LocalLeap/llada_slot_public_guard/results/experiment_queues/public_example_guard_20260719_v1}
parent_dev_records=${PARENT_DEV_RECORDS:-/root/autodl-tmp/LocalLeap/llada/results/experiment_queues/confirmed_bidirectional_rapid_20260719_v1/public_guard_dev32_preregistered_v2/audit_records.jsonl}
parent_full_records=${PARENT_FULL_RECORDS:-${parent_queue}/replay/full164/audit_records.jsonl}
parent_frozen=${PARENT_FROZEN:-${parent_queue}/frozen_sources.sha256}
queue_id=${ATTENTION_QUEUE_ID:-public_full_draft_rapid_20260719_v1}
queue_root=${llada_root}/results/experiment_queues/${queue_id}
run_base=${llada_root}/results/best_symmetric_benchmarks/${queue_id}
runner=${llada_root}/run_best_symmetric_benchmark.sh
controller=${source_root}/scripts/run_public_full_draft_queue.sh
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
  local path=$1 deadline=$(( $(date +%s) + 24 * 60 * 60 ))
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

echo "waiting_for_exact_speed_parent"
wait_for_file "${speed_queue}/DONE"
wait_for_file "${parent_queue}/DONE"
wait_for_file "${parent_dev_records}"
wait_for_file "${parent_full_records}"

if [[ ! -s "${frozen}" ]]; then
  sha256sum generate.py eval_llada.py differential_selector.py \
    run_best_symmetric_benchmark.sh compare_paired_task_runs.py \
    slice_audit_by_index.py postprocess_code.py humaneval_execution.py sanitize.py \
    "${controller}" "${source_root}/tests/test_attention_stability.py" \
    "${source_root}/tests/test_candidate_generation_trace.py" \
    "${source_root}/tests/test_compare_paired_task_runs.py" \
    "${source_root}/test_public_example_guard.py" \
    "${source_root}/public_full_draft_preregistration_20260719.json" >"${frozen}"
fi
verify
/root/miniconda3/bin/python -m py_compile generate.py eval_llada.py \
  differential_selector.py compare_paired_task_runs.py slice_audit_by_index.py
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_attention_stability.py"
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_candidate_generation_trace.py"
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_compare_paired_task_runs.py"
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/test_public_example_guard.py"
bash -n "${runner}" "${controller}"

run_gpu dev_mean_n32 0 "${runner}" humaneval 0 128 \
  trajectory_public_full_draft_verifier 0.004 trace dev_mean_n32 32 256 & p0=$!
run_gpu dev_pareto_n32 1 "${runner}" humaneval 0 128 \
  trajectory_public_pareto_verifier 0.004 trace dev_pareto_n32 32 256 & p1=$!
wait "${p0}"
wait "${p1}"

mean_root=${run_base}/humaneval/dev_mean_n32
pareto_root=${run_base}/humaneval/dev_pareto_n32
run_stage compare_dev_mean /root/miniconda3/bin/python compare_paired_task_runs.py \
  "${parent_dev_records}" "$(records "${mean_root}")" \
  --baseline-config "${parent_frozen}" --method-config "${mean_root}/run_config.txt" \
  --method-log "${mean_root}/run.log" --allow-source-drift \
  --output-dir "${queue_root}/paired/dev_mean"
run_stage compare_dev_pareto /root/miniconda3/bin/python compare_paired_task_runs.py \
  "${parent_dev_records}" "$(records "${pareto_root}")" \
  --baseline-config "${parent_frozen}" --method-config "${pareto_root}/run_config.txt" \
  --method-log "${pareto_root}/run.log" --allow-source-drift \
  --output-dir "${queue_root}/paired/dev_pareto"

eligible=()
for variant in mean pareto; do
  summary=${queue_root}/paired/dev_${variant}/paired_summary.json
  parent_correct=$(field "${summary}" baseline_correct)
  method_correct=$(field "${summary}" method_correct)
  parent_only=$(field "${summary}" baseline_only)
  method_only=$(field "${summary}" method_only)
  echo "dev_${variant} parent=${parent_correct} method=${method_correct} method_only=${method_only} parent_only=${parent_only}"
  if (( method_correct > parent_correct && method_only > parent_only )); then
    eligible+=("${variant}")
    touch "${queue_root}/DEV_PASS_${variant}"
  else
    touch "${queue_root}/DEV_REJECT_${variant}"
  fi
done
if (( ${#eligible[@]} == 0 )); then
  touch "${queue_root}/DEV_REJECTED" "${queue_root}/DONE"
  echo "public_full_draft_no_dev_improvement"
  exit 0
fi

declare -A profile=(
  [mean]=trajectory_public_full_draft_verifier
  [pareto]=trajectory_public_pareto_verifier
)
declare -A gpu=([mean]=0 [pareto]=1)
pids=()
for variant in "${eligible[@]}"; do
  run_gpu "formal_${variant}_full164" "${gpu[$variant]}" "${runner}" humaneval 0 128 \
    "${profile[$variant]}" 0.004 trace "formal_${variant}_full164" 164 256 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done

mkdir -p "${queue_root}/holdout"
/root/miniconda3/bin/python slice_audit_by_index.py "${parent_full_records}" \
  "${queue_root}/holdout/parent_96_164.jsonl" --start 96 --end 164
formal_pass=0
for variant in "${eligible[@]}"; do
  formal_root=${run_base}/humaneval/formal_${variant}_full164
  /root/miniconda3/bin/python slice_audit_by_index.py "$(records "${formal_root}")" \
    "${queue_root}/holdout/${variant}_96_164.jsonl" --start 96 --end 164
  run_stage "compare_holdout_${variant}" /root/miniconda3/bin/python compare_paired_task_runs.py \
    "${queue_root}/holdout/parent_96_164.jsonl" \
    "${queue_root}/holdout/${variant}_96_164.jsonl" \
    --baseline-config "${parent_frozen}" --method-config "${formal_root}/run_config.txt" \
    --method-log "${formal_root}/run.log" --allow-source-drift \
    --output-dir "${queue_root}/paired/holdout_${variant}"
  summary=${queue_root}/paired/holdout_${variant}/paired_summary.json
  parent_correct=$(field "${summary}" baseline_correct)
  method_correct=$(field "${summary}" method_correct)
  parent_only=$(field "${summary}" baseline_only)
  method_only=$(field "${summary}" method_only)
  echo "holdout_${variant} parent=${parent_correct} method=${method_correct} method_only=${method_only} parent_only=${parent_only}"
  if (( method_correct > parent_correct && method_only > 0 && parent_only <= method_only )); then
    touch "${queue_root}/FORMAL_PASS_${variant}"
    formal_pass=1
  else
    touch "${queue_root}/FORMAL_REJECT_${variant}"
  fi
done
if (( formal_pass )); then
  touch "${queue_root}/FORMAL_PASS"
else
  touch "${queue_root}/FORMAL_REJECTED"
fi
touch "${queue_root}/DONE"
echo "public_full_draft_queue_complete formal_pass=${formal_pass}"
