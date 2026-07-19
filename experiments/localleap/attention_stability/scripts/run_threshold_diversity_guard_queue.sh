#!/usr/bin/env bash
set -Eeuo pipefail

source_root=${SOURCE_ROOT:-/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability}
llada_root=${LLADA_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_public_guard}
queue_id=${ATTENTION_QUEUE_ID:-threshold_diversity_guard_20260719_v1}
queue_root=${llada_root}/results/experiment_queues/${queue_id}
run_base=${llada_root}/results/best_symmetric_benchmarks/${queue_id}
runner=${llada_root}/run_best_symmetric_benchmark.sh
controller=${source_root}/scripts/run_threshold_diversity_guard_queue.sh
parent=/root/autodl-tmp/LocalLeap/llada/results/experiment_queues/public_example_guard_20260719_v1/replay/full164/audit_records.jsonl
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
field() { /root/miniconda3/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }

export PATH=/root/miniconda3/bin:${PATH}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_ALLOW_CODE_EVAL=1
cd "${llada_root}"
[[ "${run_base}" == */"${queue_id}" ]] || exit 20
[[ -s "${manifest}" ]] || printf 'stage\tstatus\tstart\tfinish\texit_code_or_note\n' >"${manifest}"

if [[ ! -s "${frozen}" ]]; then
  sha256sum generate.py eval_llada.py differential_selector.py \
    run_best_symmetric_benchmark.sh postprocess_code.py humaneval_execution.py sanitize.py \
    "${source_root}/apply_public_example_guard.py" \
    "${source_root}/build_selected_sample_view.py" \
    "${source_root}/verify_selected_execution.py" \
    "${source_root}/enrich_audit_prompts.py" \
    "${source_root}/slice_audit_by_index.py" \
    "${source_root}/compare_paired_task_runs.py" \
    "${source_root}/threshold_diversity_guard_preregistration_20260719.json" \
    "${controller}" >"${frozen}"
fi
verify
/root/miniconda3/bin/python -m py_compile generate.py eval_llada.py differential_selector.py \
  "${source_root}/apply_public_example_guard.py" \
  "${source_root}/build_selected_sample_view.py" \
  "${source_root}/verify_selected_execution.py" \
  "${source_root}/enrich_audit_prompts.py" \
  "${source_root}/slice_audit_by_index.py" \
  "${source_root}/compare_paired_task_runs.py"
bash -n "${runner}" "${controller}"

run_stage tau002_full164 env CUDA_VISIBLE_DEVICES=0 LLADA_ROOT="${llada_root}" \
  ATTENTION_QUEUE_ID="${queue_id}" "${runner}" humaneval 0 128 \
  trajectory_confirmed_public_guard 0.002 trace tau002_full164 164 256

candidate=${run_base}/humaneval/tau002_full164
candidate_records=${candidate}/audit/audit_records.jsonl
samples=$(find "${candidate}/lm_eval" -type f -name 'samples_humaneval_*.jsonl' ! -name '*.cleaned' | head -1)
[[ -s "${candidate_records}" && -s "${samples}" ]]

run_stage enrich_parent /root/miniconda3/bin/python \
  "${source_root}/enrich_audit_prompts.py" "${parent}" "${samples}" \
  "${queue_root}/parent_full164_enriched.jsonl"
run_stage select_full env PYTHONPATH="${source_root}:${llada_root}" \
  /root/miniconda3/bin/python "${source_root}/apply_public_example_guard.py" \
  "${queue_root}/parent_full164_enriched.jsonl" "${candidate_records}" \
  --output-dir "${queue_root}/hybrid_full164"
run_stage build_selected_view /root/miniconda3/bin/python \
  "${source_root}/build_selected_sample_view.py" --samples "${samples}" \
  --selected-records "${queue_root}/hybrid_full164/audit_records.jsonl" \
  --output "${queue_root}/hybrid_full164/selected_samples.jsonl"
run_stage execute_selected /root/miniconda3/bin/python postprocess_code.py \
  "${queue_root}/hybrid_full164/selected_samples.jsonl"
run_stage verify_selected /root/miniconda3/bin/python \
  "${source_root}/verify_selected_execution.py" \
  "${queue_root}/hybrid_full164/audit_records.jsonl" \
  "${queue_root}/hybrid_full164/selected_samples.jsonl.cleaned" \
  --output "${queue_root}/hybrid_full164/execution_crosscheck.json"

mkdir -p "${queue_root}/holdout"
/root/miniconda3/bin/python "${source_root}/slice_audit_by_index.py" \
  "${queue_root}/parent_full164_enriched.jsonl" "${queue_root}/holdout/parent_96_164.jsonl" \
  --start 96 --end 164
/root/miniconda3/bin/python "${source_root}/slice_audit_by_index.py" \
  "${queue_root}/hybrid_full164/audit_records.jsonl" "${queue_root}/holdout/method_96_164.jsonl" \
  --start 96 --end 164
run_stage compare_holdout /root/miniconda3/bin/python \
  "${source_root}/compare_paired_task_runs.py" \
  "${queue_root}/holdout/parent_96_164.jsonl" "${queue_root}/holdout/method_96_164.jsonl" \
  --baseline-config "${frozen}" --method-config "${candidate}/run_config.txt" \
  --method-log "${queue_root}/formal_controller.log" --allow-source-drift \
  --output-dir "${queue_root}/paired_holdout"

summary=${queue_root}/paired_holdout/paired_summary.json
parent_correct=$(field "${summary}" baseline_correct)
method_correct=$(field "${summary}" method_correct)
parent_only=$(field "${summary}" baseline_only)
method_only=$(field "${summary}" method_only)
echo "holdout method=${method_correct} parent=${parent_correct} method_only=${method_only} parent_only=${parent_only}"
if (( method_correct > parent_correct && method_only > parent_only )); then
  touch "${queue_root}/FORMAL_HOLDOUT_PASS"
else
  touch "${queue_root}/FORMAL_HOLDOUT_FAIL"
fi
touch "${queue_root}/DONE"
