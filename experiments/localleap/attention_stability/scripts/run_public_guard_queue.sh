#!/usr/bin/env bash
set -Eeuo pipefail

source_root=${SOURCE_ROOT:-/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability_dev_public_guard}
llada_root=${LLADA_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_public_guard}
parent_queue=${PARENT_QUEUE:-/root/autodl-tmp/LocalLeap/llada/results/experiment_queues/confirmed_bidirectional_rapid_20260719_v1}
parent_run_base=${PARENT_RUN_BASE:-/root/autodl-tmp/LocalLeap/llada_slot_confirmed_block/results/best_symmetric_benchmarks/confirmed_bidirectional_rapid_20260719_v1}
queue_id=${ATTENTION_QUEUE_ID:-public_example_guard_20260719_v1}
queue_root=${llada_root}/results/experiment_queues/${queue_id}
run_base=${llada_root}/results/best_symmetric_benchmarks/${queue_id}
runner=${llada_root}/run_best_symmetric_benchmark.sh
controller=${source_root}/scripts/run_public_guard_queue.sh
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

export PATH=/root/miniconda3/bin:${PATH}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_ALLOW_CODE_EVAL=1
cd "${llada_root}"
[[ "${run_base}" == */"${queue_id}" ]] || exit 20
[[ -s "${manifest}" ]] || printf 'stage\tstatus\tstart\tfinish\texit_code_or_note\n' >"${manifest}"

parent_method=${parent_run_base}/humaneval/he_cbv_full164
parent_base=${parent_run_base}/humaneval/he_base_full164
echo "waiting_for_full_candidates"
wait_for_file "${parent_method}/DONE"
wait_for_file "${parent_base}/DONE"

if [[ ! -s "${frozen}" ]]; then
  sha256sum generate.py eval_llada.py differential_selector.py \
    run_best_symmetric_benchmark.sh compare_paired_task_runs.py \
    postprocess_code.py humaneval_execution.py sanitize.py "${controller}" \
    "${source_root}/apply_public_example_guard.py" \
    "${source_root}/build_selected_sample_view.py" \
    "${source_root}/verify_selected_execution.py" \
    "${source_root}/slice_audit_records.py" \
    "${source_root}/tests/test_attention_stability.py" \
    "${source_root}/tests/test_candidate_generation_trace.py" \
    "${source_root}/test_public_example_guard.py" \
    "${source_root}/test_apply_public_example_guard.py" \
    "${source_root}/test_build_selected_sample_view.py" \
    "${source_root}/test_verify_selected_execution.py" \
    "${source_root}/public_example_guard_preregistration_20260719.json" >"${frozen}"
fi
verify
/root/miniconda3/bin/python -m py_compile generate.py eval_llada.py differential_selector.py \
  "${source_root}/apply_public_example_guard.py" \
  "${source_root}/build_selected_sample_view.py" \
  "${source_root}/verify_selected_execution.py"
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_attention_stability.py"
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_candidate_generation_trace.py"
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/test_public_example_guard.py"
PYTHONPATH="${source_root}:." /root/miniconda3/bin/python "${source_root}/test_apply_public_example_guard.py"
PYTHONPATH="${source_root}:." /root/miniconda3/bin/python "${source_root}/test_build_selected_sample_view.py"
PYTHONPATH="${source_root}:." /root/miniconda3/bin/python "${source_root}/test_verify_selected_execution.py"
bash -n "${runner}" "${controller}"

parent_records=$(records "${parent_method}")
base_records=$(records "${parent_base}")
full_dir=${queue_root}/replay/full164
run_stage replay_full164 env PYTHONPATH="${source_root}:${llada_root}" \
  /root/miniconda3/bin/python "${source_root}/apply_public_example_guard.py" \
  "${parent_records}" "${base_records}" --output-dir "${full_dir}"

# Independently execute the selected programs through the official checker.
samples=$(find "${parent_method}/lm_eval" -type f -name 'samples_humaneval_*.jsonl' ! -name '*.cleaned' | head -1)
run_stage build_selected_view /root/miniconda3/bin/python \
  "${source_root}/build_selected_sample_view.py" --samples "${samples}" \
  --selected-records "${full_dir}/audit_records.jsonl" \
  --output "${full_dir}/selected_samples.jsonl"
run_stage execute_selected /root/miniconda3/bin/python postprocess_code.py \
  "${full_dir}/selected_samples.jsonl"
run_stage verify_selected_execution /root/miniconda3/bin/python \
  "${source_root}/verify_selected_execution.py" \
  "${full_dir}/audit_records.jsonl" \
  "${full_dir}/selected_samples.jsonl.cleaned" \
  --output "${full_dir}/execution_crosscheck.json"

mkdir -p "${queue_root}/holdout"
/root/miniconda3/bin/python "${source_root}/slice_audit_records.py" \
  "${parent_records}" "${queue_root}/holdout/parent_96_164.jsonl" --start 96 --end 164
/root/miniconda3/bin/python "${source_root}/slice_audit_records.py" \
  "${base_records}" "${queue_root}/holdout/base_96_164.jsonl" --start 96 --end 164
run_stage replay_holdout env PYTHONPATH="${source_root}:${llada_root}" \
  /root/miniconda3/bin/python "${source_root}/apply_public_example_guard.py" \
  "${queue_root}/holdout/parent_96_164.jsonl" \
  "${queue_root}/holdout/base_96_164.jsonl" \
  --output-dir "${queue_root}/replay/holdout_96_164"

formal=${queue_root}/replay/holdout_96_164/audit_summary.json
fm=$(field "${formal}" method_correct)
fp=$(field "${formal}" parent_correct)
fb=$(field "${formal}" baseline_correct)
fo=$(field "${formal}" method_only_vs_parent)
fl=$(field "${formal}" parent_only_vs_method)
echo "formal_holdout method=${fm} parent=${fp} baseline=${fb} method_only=${fo} parent_only=${fl}"
if (( fm > fp && fm > fb && fo > 0 && fl <= fo )); then
  touch "${queue_root}/FORMAL_PASS"
else
  touch "${queue_root}/FORMAL_FAIL" "${queue_root}/DONE"
  exit 0
fi

# Confirm the integrated path on real model output after the accuracy gate.
wait_for_file "${parent_queue}/DONE"
run_gpu he_public_guard_smoke4 0 "${runner}" humaneval 0 128 \
  trajectory_confirmed_public_guard 0.004 trace he_public_guard_smoke4 4 256

# MATH has no public function examples, so v11 is exactly v9 there.  Reuse the
# parent's audited arm when present; otherwise run the same frozen v9 decoder.
parent_math=${parent_run_base}/localleap_math500/math_cbv_n50
if [[ -e "${parent_math}/DONE" ]]; then
  touch "${queue_root}/REUSED_PARENT_MATH"
else
  run_gpu math_v9_fallback_n50 0 "${runner}" localleap_math500 0 128 \
    trajectory_confirmed_bidirectional_block 0.004 trace math_v9_fallback_n50 50 256 & p0=$!
  (
    run_gpu math_fast_n50 1 "${runner}" localleap_math500 0 128 \
      symmetric_fast 0.004 trace math_fast_n50 50 256
    run_gpu math_base_n50 1 "${runner}" localleap_math500 0 128 \
      baseline 0 trace math_base_n50 50 256
  ) & p1=$!
  wait "${p0}"
  wait "${p1}"
  method_math=${run_base}/localleap_math500/math_v9_fallback_n50
  fast_math=${run_base}/localleap_math500/math_fast_n50
  base_math=${run_base}/localleap_math500/math_base_n50
  run_stage math_vs_fast /root/miniconda3/bin/python compare_paired_task_runs.py \
    "$(records "${fast_math}")" "$(records "${method_math}")" \
    --baseline-config "${fast_math}/run_config.txt" \
    --method-config "${method_math}/run_config.txt" \
    --output-dir "${queue_root}/paired/math_vs_fast"
  run_stage math_vs_base /root/miniconda3/bin/python compare_paired_task_runs.py \
    "$(records "${base_math}")" "$(records "${method_math}")" \
    --baseline-config "${base_math}/run_config.txt" \
    --method-config "${method_math}/run_config.txt" \
    --output-dir "${queue_root}/paired/math_vs_base"
fi

touch "${queue_root}/DONE"
echo "public_guard_queue_complete"
