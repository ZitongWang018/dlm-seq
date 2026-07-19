#!/usr/bin/env bash
set -Eeuo pipefail

source_root=${SOURCE_ROOT:-/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability_dev_admissible_lazy_guard}
llada_root=${LLADA_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_admissible_lazy_guard}
wait_queue=${WAIT_QUEUE:-/root/autodl-tmp/LocalLeap/llada_slot_outcome_arbiter/results/experiment_queues/outcome_arbiter_rapid_20260719_v1}
v9_runs=${V9_RUNS:-/root/autodl-tmp/LocalLeap/llada_slot_confirmed_block/results/best_symmetric_benchmarks/confirmed_bidirectional_rapid_20260719_v1}
v11_queue=${V11_QUEUE:-/root/autodl-tmp/LocalLeap/llada_slot_public_guard/results/experiment_queues/public_example_guard_20260719_v1}
v11_runs=${V11_RUNS:-/root/autodl-tmp/LocalLeap/llada_slot_public_guard/results/best_symmetric_benchmarks/public_example_guard_20260719_v1}
v11_mbpp_eval=${V11_MBPP_EVAL:-/root/autodl-tmp/LocalLeap/llada/results/experiment_queues/public_guard_mbpp_reaudit_20260719_v1/v11_n100}
queue_id=${ATTENTION_QUEUE_ID:-admissible_lazy_guard_20260719_v1}
queue_root=${llada_root}/results/experiment_queues/${queue_id}
run_base=${llada_root}/results/best_symmetric_benchmarks/${queue_id}
runner=${llada_root}/run_best_symmetric_benchmark.sh
controller=${source_root}/scripts/run_admissible_lazy_guard_queue.sh
exact_auditor=${source_root}/compare_exact_output_runs.py
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
  else
    printf '%s\n' "$1/audit/task_audit_records.jsonl"
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
exact_pair() {
  local label=$1 parent_records=$2 candidate_run=$3
  run_stage "exact_${label}" /root/miniconda3/bin/python "${exact_auditor}" \
    "${parent_records}" "$(records "${candidate_run}")" \
    --output-dir "${queue_root}/exact/${label}"
}
require_exact() {
  /root/miniconda3/bin/python - "$1" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x["all_invariants_pass"], x
assert x["nfe_reduction"] >= 0, x
PY
}
sample_file() {
  local run_root=$1 matches=()
  mapfile -t matches < <(find "${run_root}/lm_eval" -type f -name 'samples_mbpp_*.jsonl' | sort)
  [[ ${#matches[@]} -eq 1 ]] || return 22
  printf '%s\n' "${matches[0]}"
}

export PATH=/root/miniconda3/bin:${PATH}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_ALLOW_CODE_EVAL=1
cd "${llada_root}"
[[ "${run_base}" == */"${queue_id}" ]] || exit 20
[[ -s "${manifest}" ]] || printf 'stage\tstatus\tstart\tfinish\texit_code_or_note\n' >"${manifest}"

echo "waiting_for_accuracy_queue=${wait_queue}"
wait_for_file "${wait_queue}/DONE"
wait_for_file "${v11_queue}/DONE"

if [[ ! -s "${frozen}" ]]; then
  sha256sum generate.py eval_llada.py run_best_symmetric_benchmark.sh \
    "${exact_auditor}" "${controller}" \
    "${source_root}/audit_mbpp_assertions.py" \
    "${source_root}/slice_audit_records.py" \
    "${source_root}/tests/test_attention_stability.py" \
    "${source_root}/tests/test_candidate_generation_trace.py" \
    "${source_root}/test_audit_mbpp_assertions.py" \
    "${source_root}/test_v15_runner_contract.py" \
    "${source_root}/admissible_lazy_guard_preregistration_20260719.json" >"${frozen}"
fi
verify
/root/miniconda3/bin/python -m py_compile generate.py eval_llada.py \
  "${exact_auditor}" "${source_root}/audit_mbpp_assertions.py"
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_attention_stability.py"
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_candidate_generation_trace.py"
PYTHONPATH="${source_root}" /root/miniconda3/bin/python "${source_root}/test_audit_mbpp_assertions.py"
bash -n "${runner}" "${controller}"
/root/miniconda3/bin/python "${source_root}/test_v15_runner_contract.py"

echo "method=trajectory_early_lazy_confirmed_public_guard_tau_0.004"
echo "claim=exact_v11_output_with_admissible_vertical_early_abort"

mkdir -p "${queue_root}/parents"
/root/miniconda3/bin/python "${source_root}/slice_audit_records.py" \
  "${v11_queue}/replay/full164/audit_records.jsonl" \
  "${queue_root}/parents/he_v11_dev32.jsonl" --start 0 --end 32

run_gpu he_admissible_dev32 0 "${runner}" humaneval 0 128 \
  trajectory_early_lazy_confirmed_public_guard 0.004 trace he_admissible_dev32 32 256 & p0=$!
run_gpu gsm_admissible_n64 1 "${runner}" gsm8k 0 128 \
  trajectory_early_lazy_confirmed_public_guard 0.004 trace gsm_admissible_n64 64 256 & p1=$!
wait "${p0}"
wait "${p1}"

he_dev=${run_base}/humaneval/he_admissible_dev32
gsm_run=${run_base}/gsm8k/gsm_admissible_n64
exact_pair he_dev32 "${queue_root}/parents/he_v11_dev32.jsonl" "${he_dev}"
exact_pair gsm_n64 "$(records "${v9_runs}/gsm8k/gsm_cbv_n64")" "${gsm_run}"
require_exact "${queue_root}/exact/he_dev32/exact_summary.json"
require_exact "${queue_root}/exact/gsm_n64/exact_summary.json"
/root/miniconda3/bin/python - \
  "${queue_root}/exact/he_dev32/exact_summary.json" \
  "${queue_root}/exact/gsm_n64/exact_summary.json" <<'PY'
import json,sys
saved=sum(json.load(open(p))["nfe_reduction"] for p in sys.argv[1:])
assert saved > 0, saved
PY
touch "${queue_root}/QUICK_GATE_PASS"

run_gpu he_admissible_full164 0 "${runner}" humaneval 0 128 \
  trajectory_early_lazy_confirmed_public_guard 0.004 trace he_admissible_full164 164 256 & p0=$!
(
  run_gpu math_admissible_n50 1 "${runner}" localleap_math500 0 128 \
    trajectory_early_lazy_confirmed_public_guard 0.004 trace math_admissible_n50 50 256
  run_gpu mbpp_admissible_n100 1 "${runner}" mbpp 0 128 \
    trajectory_early_lazy_confirmed_public_guard 0.004 trace mbpp_admissible_n100 100 256
) & p1=$!
wait "${p0}"
wait "${p1}"

he_full=${run_base}/humaneval/he_admissible_full164
math_run=${run_base}/localleap_math500/math_admissible_n50
mbpp_run=${run_base}/mbpp/mbpp_admissible_n100
exact_pair he_full164 "${v11_queue}/replay/full164/audit_records.jsonl" "${he_full}"
exact_pair math_n50 "$(records "${v9_runs}/localleap_math500/math_cbv_n50")" "${math_run}"
exact_pair mbpp_n100 "$(records "${v11_runs}/mbpp/mbpp_public_guard_n100")" "${mbpp_run}"
require_exact "${queue_root}/exact/he_full164/exact_summary.json"
require_exact "${queue_root}/exact/math_n50/exact_summary.json"
require_exact "${queue_root}/exact/mbpp_n100/exact_summary.json"

wait_for_file "${v11_mbpp_eval}/audit_records.jsonl"
run_stage mbpp_assertion_v2 /root/miniconda3/bin/python \
  "${source_root}/audit_mbpp_assertions.py" \
  --samples "$(sample_file "${mbpp_run}")" \
  --task-records "$(records "${mbpp_run}")" \
  --output-dir "${queue_root}/mbpp_assertion_v2"
run_stage mbpp_vs_v11 /root/miniconda3/bin/python \
  "${source_root}/compare_paired_task_runs.py" \
  "${v11_mbpp_eval}/audit_records.jsonl" \
  "${queue_root}/mbpp_assertion_v2/audit_records.jsonl" \
  --baseline-config "${v11_runs}/mbpp/mbpp_public_guard_n100/run_config.txt" \
  --method-config "${mbpp_run}/run_config.txt" --allow-source-drift \
  --output-dir "${queue_root}/paired/mbpp_vs_v11_n100"

touch "${queue_root}/DONE"
echo "admissible_lazy_guard_queue_complete"
