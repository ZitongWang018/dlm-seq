#!/usr/bin/env bash
set -uo pipefail

queue_id=${ATTENTION_QUEUE_ID:-trajectory_convergent_consensus_20260719_v1}
llada_root=${LLADA_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_convergent_consensus}
source_root=${ATTENTION_SOURCE_ROOT:-/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability_dev_convergent_consensus}
runner=${source_root}/scripts/run_best_symmetric_benchmark.sh
controller=${source_root}/scripts/run_convergent_consensus_queue.sh
queue_root=${llada_root}/results/experiment_queues/${queue_id}
run_base=${llada_root}/results/best_symmetric_benchmarks/${queue_id}
manifest=${queue_root}/formal_manifest.tsv
frozen=${queue_root}/FROZEN_SOURCE_SHA256
parent_queue=/root/autodl-tmp/LocalLeap/llada/results/experiment_queues/trajectory_consensus_formal_20260719_v3
mkdir -p "${queue_root}" "${queue_root}/holdout" "${queue_root}/paired"
exec > >(tee -a "${queue_root}/formal_controller.log") 2>&1

append_manifest() { { flock 9; printf '%s\n' "$1" >&9; } 9>>"${manifest}"; }
done_stage() { [[ -s "${manifest}" ]] && awk -F '\t' -v x="$1" '$1==x && $2=="DONE"{ok=1} END{exit !ok}' "${manifest}"; }
verify() { (cd "${llada_root}" && sha256sum -c "${frozen}" >/dev/null) || { touch "${queue_root}/FAILED_SOURCE_DRIFT"; exit 21; }; }
run_stage() {
  local label=$1; shift
  done_stage "${label}" && return 0
  verify
  local start finish rc
  start=$(date --iso-8601=seconds)
  append_manifest "$(printf '%s\tSTARTED\t%s\t\t' "${label}" "${start}")"
  set +e
  timeout --kill-after=5m 12h "$@"
  rc=$?
  set -e
  finish=$(date --iso-8601=seconds)
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
audit_records() {
  [[ -s "$1/audit/audit_records.jsonl" ]] && { echo "$1/audit/audit_records.jsonl"; return; }
  [[ -s "$1/audit/task_audit_records.jsonl" ]] && echo "$1/audit/task_audit_records.jsonl"
}
pair_runs() {
  local label=$1 baseline=$2 method=$3 br mr
  br=$(audit_records "${baseline}") || return 1
  mr=$(audit_records "${method}") || return 1
  run_stage "${label}" /root/miniconda3/bin/python "${llada_root}/compare_paired_task_runs.py" \
    "${br}" "${mr}" --baseline-config "${baseline}/run_config.txt" \
    --method-config "${method}/run_config.txt" \
    --method-log "${queue_root}/$(basename "${method}").log" \
    --output-dir "${queue_root}/paired/${label}"
}
field() { /root/miniconda3/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }
require_done() {
  local run
  for run in "$@"; do
    [[ -e "${run}/DONE" ]] || { touch "${queue_root}/FAILED_REQUIRED_RUN"; exit 23; }
  done
}
wait_for_parent_boundary() {
  local pid
  pid=$(cat "${parent_queue}/controller.pid" 2>/dev/null || true)
  while [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; do sleep 10; done
}
export PATH=/root/miniconda3/bin:${PATH}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=/root/autodl-tmp/.cache/huggingface HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_ALLOW_CODE_EVAL=1
cd "${llada_root}"
[[ "${run_base}" == */"${queue_id}" ]] || { echo "run_base queue mismatch: ${run_base}"; exit 20; }
[[ -s "${manifest}" ]] || printf 'stage\tstatus\tstart\tfinish\texit_code_or_note\n' >"${manifest}"
if [[ ! -s "${frozen}" ]]; then
  sha256sum generate.py eval_llada.py compare_paired_task_runs.py audit_attention_stability.py \
    audit_lm_eval_task.py postprocess_code.py humaneval_execution.py sanitize.py \
    "${runner}" "${controller}" "${source_root}/tests/test_attention_stability.py" \
    "${source_root}/convergent_coverage_preregistration_20260719.json" >"${frozen}"
fi
verify
/root/miniconda3/bin/python -m py_compile generate.py eval_llada.py
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_attention_stability.py"
bash -n "${runner}" "${controller}"

echo "formal_baseline=original_llada_low_confidence"
echo "method=trajectory_convergent_coverage_consensus_block_tau_0.004"
echo "selection=convergent_one_nat_or_full_revision_coverage_plus_original_schedule_consensus"
echo "fitted_hyperparameters=none"
echo "phase=versioned_formal_validation_before_unseen_64_96_results"

# This descendant was committed and preregistered before the parent generated
# HumanEval/64.  Launch only if the parent fails its unseen-slice promotion
# gate; otherwise preserve the stronger branch and avoid redundant compute.
wait_for_parent_boundary
if [[ ! -e "${parent_queue}/DONE_NO_MATH" ]]; then
  touch "${queue_root}/DONE_DEFERRED_PARENT_PROMOTED" "${queue_root}/DONE"
  exit 0
fi

# Fresh comparators under the convergent-consensus source make 64..95 formal;
# earlier comparator runs are not reused for the final claim.
run_gpu he_convergent_n96 0 "${runner}" humaneval 0 128 trajectory_convergent_coverage_consensus_block 0.004 trace he_convergent_n96 96 256 & p0=$!
(
  run_gpu he_fast_n96 1 "${runner}" humaneval 0 128 symmetric_fast 0.004 trace he_fast_n96 96 256
  run_gpu he_base_n96 1 "${runner}" humaneval 0 128 baseline 0 trace he_base_n96 96 256
) & p1=$!
wait "${p0}" || true; wait "${p1}" || true
require_done "${run_base}/humaneval/he_convergent_n96" "${run_base}/humaneval/he_fast_n96" "${run_base}/humaneval/he_base_n96"
pair_runs he_convergent_vs_fast_n96 "${run_base}/humaneval/he_fast_n96" "${run_base}/humaneval/he_convergent_n96"
pair_runs he_convergent_vs_base_n96 "${run_base}/humaneval/he_base_n96" "${run_base}/humaneval/he_convergent_n96"
for name in fast base convergent; do
  /root/miniconda3/bin/python "${source_root}/slice_audit_records.py" \
    "$(audit_records "${run_base}/humaneval/he_${name}_n96")" \
    "${queue_root}/holdout/he_${name}_64_96.jsonl" --start 64 --end 96
done
run_stage he_holdout_vs_fast /root/miniconda3/bin/python "${llada_root}/compare_paired_task_runs.py" \
  "${queue_root}/holdout/he_fast_64_96.jsonl" "${queue_root}/holdout/he_convergent_64_96.jsonl" \
  --baseline-config "${run_base}/humaneval/he_fast_n96/run_config.txt" \
  --method-config "${run_base}/humaneval/he_convergent_n96/run_config.txt" \
  --method-log "${queue_root}/he_convergent_n96.log" --output-dir "${queue_root}/paired/he_holdout_vs_fast"
run_stage he_holdout_vs_base /root/miniconda3/bin/python "${llada_root}/compare_paired_task_runs.py" \
  "${queue_root}/holdout/he_base_64_96.jsonl" "${queue_root}/holdout/he_convergent_64_96.jsonl" \
  --baseline-config "${run_base}/humaneval/he_base_n96/run_config.txt" \
  --method-config "${run_base}/humaneval/he_convergent_n96/run_config.txt" \
  --method-log "${queue_root}/he_convergent_n96.log" --output-dir "${queue_root}/paired/he_holdout_vs_base"
hf="${queue_root}/paired/he_holdout_vs_fast/paired_summary.json"; hb="${queue_root}/paired/he_holdout_vs_base/paired_summary.json"
hm=$(field "${hf}" method_correct); hfast=$(field "${hf}" baseline_correct); hbase=$(field "${hb}" baseline_correct); hr=$(field "${hf}" method_only)
echo "holdout_gate method=${hm} fast=${hfast} base=${hbase} recoveries=${hr}"
if ! (( hm > hfast && hm > hbase && hr > 0 )); then
  touch "${queue_root}/DONE_NO_MATH" "${queue_root}/DONE"
  exit 0
fi

(
  run_gpu math_convergent_n50 0 "${runner}" localleap_math500 0 128 trajectory_convergent_coverage_consensus_block 0.004 trace math_convergent_n50 50 256
  run_gpu math_base_n50 0 "${runner}" localleap_math500 0 128 baseline 0 trace math_base_n50 50 256
) & p0=$!
run_gpu math_fast_n50 1 "${runner}" localleap_math500 0 128 symmetric_fast 0.004 trace math_fast_n50 50 256 & p1=$!
wait "${p0}" || true; wait "${p1}" || true
require_done "${run_base}/localleap_math500/math_convergent_n50" "${run_base}/localleap_math500/math_fast_n50" "${run_base}/localleap_math500/math_base_n50"
pair_runs math_convergent_vs_fast_n50 "${run_base}/localleap_math500/math_fast_n50" "${run_base}/localleap_math500/math_convergent_n50"
pair_runs math_convergent_vs_base_n50 "${run_base}/localleap_math500/math_base_n50" "${run_base}/localleap_math500/math_convergent_n50"
mf="${queue_root}/paired/math_convergent_vs_fast_n50/paired_summary.json"; mb="${queue_root}/paired/math_convergent_vs_base_n50/paired_summary.json"
mm=$(field "${mf}" method_correct); mfast=$(field "${mf}" baseline_correct); mbase=$(field "${mb}" baseline_correct)
if ! (( mm >= mfast && mm >= mbase )); then
  touch "${queue_root}/DONE_NO_GENERALIZATION" "${queue_root}/DONE"
  exit 0
fi

(
  run_gpu mbpp_base_n50 0 "${runner}" mbpp 0 128 baseline 0 trace mbpp_base_n50 50 256
  run_gpu mbpp_fast_n50 0 "${runner}" mbpp 0 128 symmetric_fast 0.004 trace mbpp_fast_n50 50 256
  run_gpu mbpp_convergent_n50 0 "${runner}" mbpp 0 128 trajectory_convergent_coverage_consensus_block 0.004 trace mbpp_convergent_n50 50 256
  pair_runs mbpp_convergent_vs_base_n50 "${run_base}/mbpp/mbpp_base_n50" "${run_base}/mbpp/mbpp_convergent_n50"
  pair_runs mbpp_convergent_vs_fast_n50 "${run_base}/mbpp/mbpp_fast_n50" "${run_base}/mbpp/mbpp_convergent_n50"
) & p0=$!
(
  run_gpu gsm_base_n64 1 "${runner}" gsm8k 0 128 baseline 0 trace gsm_base_n64 64 256
  run_gpu gsm_fast_n64 1 "${runner}" gsm8k 0 128 symmetric_fast 0.004 trace gsm_fast_n64 64 256
  run_gpu gsm_convergent_n64 1 "${runner}" gsm8k 0 128 trajectory_convergent_coverage_consensus_block 0.004 trace gsm_convergent_n64 64 256
  pair_runs gsm_convergent_vs_base_n64 "${run_base}/gsm8k/gsm_base_n64" "${run_base}/gsm8k/gsm_convergent_n64"
  pair_runs gsm_convergent_vs_fast_n64 "${run_base}/gsm8k/gsm_fast_n64" "${run_base}/gsm8k/gsm_convergent_n64"
) & p1=$!
wait "${p0}" || true; wait "${p1}" || true
touch "${queue_root}/DONE"
