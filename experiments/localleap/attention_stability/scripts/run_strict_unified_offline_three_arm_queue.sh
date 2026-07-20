#!/usr/bin/env bash
set -Eeuo pipefail

strict_root=${STRICT_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_strict_unified_v20_v10}
v15_root=${V15_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_admissible_lazy_guard}
v18_root=${V18_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_early_localized_conflict_repair_v3}
v19_root=${V19_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_sparse_context_repair_v19_direct_v4}
v20_root=${V20_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_original_anchor_pareto_v20_direct_v4}
v18_id=${V18_ID:-early_localized_evidence_conflict_repair_20260720_v3}
v19_id=${V19_ID:-sparse_context_repair_direct_20260720_v4}
v20_id=${V20_ID:-original_anchor_pareto_direct_20260720_v4}
queue_id=${ATTENTION_QUEUE_ID:-strict_unified_offline_three_arm_20260720_v10}
queue_root=${strict_root}/results/experiment_queues/${queue_id}
run_base=${strict_root}/results/best_symmetric_benchmarks/${queue_id}
v18_queue=${v18_root}/results/experiment_queues/${v18_id}
v19_queue=${v19_root}/results/experiment_queues/${v19_id}
v20_queue=${v20_root}/results/experiment_queues/${v20_id}
model_path=${MODEL_PATH:-/root/autodl-tmp/model/LLaDA/instruct}

controller=${strict_root}/scripts/run_strict_unified_offline_three_arm_queue.sh
runner=${strict_root}/scripts/run_best_symmetric_benchmark.sh
pairer=${strict_root}/compare_paired_task_runs.py
runtime_auditor=${strict_root}/audit_runtime_model_inputs.py
config_auditor=${strict_root}/audit_run_config_fairness.py
offline_freezer=${strict_root}/freeze_offline_eval_protocol.py
offline_preflight=${strict_root}/offline_dataset_preflight.py
leakage_auditor=${strict_root}/audit_generation_leakage_v2.py
mbpp_auditor=${strict_root}/audit_mbpp_assertions.py
finalizer=${strict_root}/finalize_unified_offline_protocol.py
preregistration=${strict_root}/strict_unified_preregistration_20260720_v2.json
manifest=${queue_root}/formal_manifest.tsv
source_manifest=${queue_root}/frozen_sources.sha256
offline_manifest=${queue_root}/protocol/offline_artifacts.json
weight_manifest=${queue_root}/protocol/model_weights.sha256

mkdir -p "${queue_root}/protocol"
printf '%s\n' "$$" >"${queue_root}/controller.pid"
exec > >(tee -a "${queue_root}/formal_controller.log") 2>&1
trap 'rc=$?; echo "strict_queue_error rc=${rc} line=${LINENO}"; touch "${queue_root}/FAILED"; exit "${rc}"' ERR

export PATH=/root/miniconda3/bin:${PATH}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true
unset HF_ENDPOINT TRANSFORMERS_CACHE || true

if [[ ! -s "${source_manifest}" ]]; then
  ( cd "${strict_root}" && sha256sum \
      generate.py eval_llada.py differential_selector.py outcome_arbiter.py \
      compare_paired_task_runs.py audit_runtime_model_inputs.py \
      audit_run_config_fairness.py freeze_offline_eval_protocol.py \
      offline_dataset_preflight.py \
      audit_generation_leakage_v2.py audit_mbpp_assertions.py \
      finalize_unified_offline_protocol.py postprocess_code.py \
      humaneval_execution.py sanitize.py model/modeling_llada.py \
      scripts/run_best_symmetric_benchmark.sh \
      scripts/run_strict_unified_offline_three_arm_queue.sh \
      test_strict_unified_offline_protocol.py \
      strict_unified_preregistration_20260720_v2.json ) >"${source_manifest}"
fi
verify_sources() { ( cd "${strict_root}" && sha256sum -c "${source_manifest}" >/dev/null ); }
append_manifest() { printf '%s\n' "$1" >>"${manifest}"; }
run_stage() {
  local label=$1 start finish rc
  shift
  verify_sources
  start=$(date --iso-8601=seconds)
  append_manifest "$(printf '%s\tSTARTED\t%s\t\t' "${label}" "${start}")"
  set +e
  timeout --kill-after=5m 96h "$@"
  rc=$?
  set -e
  finish=$(date --iso-8601=seconds)
  verify_sources
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
  run_stage "${label}" env CUDA_VISIBLE_DEVICES="${gpu}" LLADA_ROOT="${strict_root}" \
    ATTENTION_QUEUE_ID="${queue_id}" "$@"
}
wait_terminal() {
  local root=$1 deadline=$(( $(date +%s) + 14 * 24 * 60 * 60 ))
  while [[ ! -e "${root}/DONE" && ! -e "${root}/FAILED" ]]; do
    verify_sources
    (( $(date +%s) < deadline )) || return 21
    sleep 20
  done
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
sample_file() {
  local task=$1 root=$2 matches=()
  mapfile -t matches < <(find "${root}/lm_eval" -type f -name "samples_${task}_*.jsonl" | sort)
  [[ ${#matches[@]} -eq 1 ]] || return 24
  printf '%s\n' "${matches[0]}"
}
run_pair() {
  local task=$1 baseline_tag=$2 candidate_tag=$3 baseline_gpu=$4 candidate_gpu=$5 p0 p1 rc0 rc1
  ( run_gpu "${baseline_tag}" "${baseline_gpu}" "${runner}" "${task}" 0 128 \
      baseline 0 trace "${baseline_tag}" full 256 ) & p0=$!
  ( run_gpu "${candidate_tag}" "${candidate_gpu}" "${runner}" "${task}" 0 128 \
      "${candidate_profile}" 0.004 trace "${candidate_tag}" full 256 ) & p1=$!
  set +e
  wait "${p0}"; rc0=$?
  wait "${p1}"; rc1=$?
  set -e
  [[ ${rc0} -eq 0 && ${rc1} -eq 0 ]]
}
run_fast_pair() {
  local task0=$1 tag0=$2 task1=$3 tag1=$4 p0 p1 rc0 rc1
  ( run_gpu "${tag0}" 0 "${runner}" "${task0}" 0 128 \
      symmetric_fast 0.004 trace "${tag0}" full 256 ) & p0=$!
  ( run_gpu "${tag1}" 1 "${runner}" "${task1}" 0 128 \
      symmetric_fast 0.004 trace "${tag1}" full 256 ) & p1=$!
  set +e
  wait "${p0}"; rc0=$?
  wait "${p1}"; rc1=$?
  set -e
  [[ ${rc0} -eq 0 && ${rc1} -eq 0 ]]
}

[[ -s "${manifest}" ]] || printf 'stage\tstatus\tstart\tfinish\texit_code_or_note\n' >"${manifest}"
verify_sources
run_stage source_tests env PYTHONPATH="${strict_root}" /root/miniconda3/bin/python \
  "${strict_root}/test_strict_unified_offline_protocol.py"
run_stage shell_contract bash -n "${controller}" "${runner}"
run_stage preregistration /root/miniconda3/bin/python -m json.tool "${preregistration}"
run_stage offline_dataset_preflight /root/miniconda3/bin/python "${offline_preflight}" \
  --model-path "${model_path}" --output "${queue_root}/protocol/offline_dataset_preflight.json"

# Freeze every locally used dataset view, task/evaluator source, tokenizer and
# model metadata before any strict-arm generation. Model weights are hashed
# separately because they are large and must be reverified after all arms.
run_stage offline_artifact_manifest /root/miniconda3/bin/python "${offline_freezer}" build \
  --model-path "${model_path}" --minimum-arrow-files 4 --output "${offline_manifest}" \
  --root "${model_path}" \
  --root /root/autodl-tmp/.cache/huggingface/datasets/openai___openai_humaneval \
  --root /root/autodl-tmp/.cache/huggingface/datasets/google-research-datasets___mbpp \
  --root /root/autodl-tmp/.cache/huggingface/datasets/openai___gsm8k \
  --root /root/autodl-tmp/.cache/huggingface/datasets/HuggingFaceH4___math-500 \
  --root "${strict_root}/tasks" \
  --root /root/miniconda3/lib/python3.12/site-packages/lm_eval/tasks/humaneval \
  --root /root/miniconda3/lib/python3.12/site-packages/lm_eval/tasks/mbpp \
  --root /root/miniconda3/lib/python3.12/site-packages/lm_eval/tasks/gsm8k \
  --root /root/autodl-tmp/reference_repos/Prism/LLaDA/LLaDA_Baseline/scripts
run_stage model_weight_manifest bash -c \
  'sha256sum "$1"/*.safetensors | sort -k2 >"$2"' _ "${model_path}" "${weight_manifest}"
{
  echo "captured=$(date --iso-8601=seconds)"
  echo "canonical_commit=$(git -C /root/autodl-tmp/dlm-seq-flow rev-parse HEAD)"
  echo "strict_root=${strict_root}"
  /root/miniconda3/bin/python -m pip freeze
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
} >"${queue_root}/protocol/environment_and_packages.txt"
run_stage leakage_static /root/miniconda3/bin/python "${leakage_auditor}" \
  --source-root "${strict_root}" --output "${queue_root}/protocol/leakage_static.json"

echo "waiting_for_v20_handoff_terminal=${v20_queue}"
wait_terminal "${v20_queue}"
[[ -e "${v20_queue}/DONE" && ! -e "${v20_queue}/FAILED" ]] || {
  touch "${queue_root}/BLOCKED_V20_PIPELINE_FAILURE"; exit 22; }

if [[ ! -e "${v20_queue}/ACCEPTED" ]]; then
  echo "waiting_for_repair_chain_terminal=${v19_queue}"
  wait_terminal "${v19_queue}"
  [[ -e "${v19_queue}/DONE" && ! -e "${v19_queue}/FAILED" ]] || {
    touch "${queue_root}/BLOCKED_REPAIR_CHAIN_FAILURE"; exit 23; }
fi

if [[ -e "${v19_queue}/ACCEPTED" ]]; then
  candidate_profile=trajectory_early_sparse_context_repair
  candidate_family=v19_sparse_context_repair
  selection_reason=v19_passed_preregistered_unified_full4_gate
elif [[ -e "${v18_queue}/ACCEPTED" ]]; then
  candidate_profile=trajectory_early_localized_evidence_conflict_repair
  candidate_family=v18_early_localized_evidence_conflict_repair
  selection_reason=v18_passed_preregistered_unified_full4_gate
elif [[ -e "${v20_queue}/ACCEPTED" ]]; then
  candidate_profile=trajectory_original_anchor_pareto
  candidate_family=v20_original_anchor_pareto
  selection_reason=v20_passed_preregistered_four_task_rapid_gate
else
  printf 'reason=no_candidate_passed_cross_task_gate\nfinished=%s\n' \
    "$(date --iso-8601=seconds)" >"${queue_root}/NO_UNIFIED_CANDIDATE"
  touch "${queue_root}/DONE"
  exit 0
fi
printf 'candidate_profile\t%s\ncandidate_family\t%s\nselection_reason\t%s\nselected_at\t%s\n' \
  "${candidate_profile}" "${candidate_family}" "${selection_reason}" \
  "$(date --iso-8601=seconds)" >"${queue_root}/candidate_selection.tsv"

# Fresh baseline and the one globally selected candidate are simultaneous for
# every task. Candidate GPU assignment alternates to balance the two cards.
run_pair humaneval he_strict_baseline_full164 he_strict_candidate_full164 0 1
run_pair localleap_math500 math_strict_baseline_full500 math_strict_candidate_full500 1 0
run_pair gsm8k gsm_strict_baseline_full1319 gsm_strict_candidate_full1319 0 1
run_pair mbpp mbpp_strict_baseline_full500 mbpp_strict_candidate_full500 1 0

# The fast arm is a separately reported comparator, never a routing component.
run_fast_pair humaneval he_strict_fast_full164 localleap_math500 math_strict_fast_full500
run_fast_pair gsm8k gsm_strict_fast_full1319 mbpp mbpp_strict_fast_full500

he_base=${run_base}/humaneval/he_strict_baseline_full164
he_candidate=${run_base}/humaneval/he_strict_candidate_full164
he_fast=${run_base}/humaneval/he_strict_fast_full164
math_base=${run_base}/localleap_math500/math_strict_baseline_full500
math_candidate=${run_base}/localleap_math500/math_strict_candidate_full500
math_fast=${run_base}/localleap_math500/math_strict_fast_full500
gsm_base=${run_base}/gsm8k/gsm_strict_baseline_full1319
gsm_candidate=${run_base}/gsm8k/gsm_strict_candidate_full1319
gsm_fast=${run_base}/gsm8k/gsm_strict_fast_full1319
mbpp_base=${run_base}/mbpp/mbpp_strict_baseline_full500
mbpp_candidate=${run_base}/mbpp/mbpp_strict_candidate_full500
mbpp_fast=${run_base}/mbpp/mbpp_strict_fast_full500

mkdir -p "${queue_root}/mbpp_assertion"
for item in "baseline:${mbpp_base}" "candidate:${mbpp_candidate}" "fast:${mbpp_fast}"; do
  arm=${item%%:*}; root=${item#*:}
  run_stage "mbpp_assertion_${arm}" /root/miniconda3/bin/python "${mbpp_auditor}" \
    --samples "$(sample_file mbpp "${root}")" --task-records "$(records "${root}")" \
    --output-dir "${queue_root}/mbpp_assertion/${arm}"
done

mkdir -p "${queue_root}/paired" "${queue_root}/runtime_inputs" \
  "${queue_root}/config_fairness" "${queue_root}/leakage_v2"
pair_task() {
  local task=$1 baseline_root=$2 candidate_root=$3 fast_root=$4 baseline_records=$5 candidate_records=$6 fast_records=$7
  run_stage "pair_${task}_candidate" /root/miniconda3/bin/python "${pairer}" \
    "${baseline_records}" "${candidate_records}" \
    --baseline-config "${baseline_root}/run_config.txt" --method-config "${candidate_root}/run_config.txt" \
    --baseline-log "${queue_root}/$(basename "${baseline_root}").log" \
    --method-log "${queue_root}/$(basename "${candidate_root}").log" \
    --output-dir "${queue_root}/paired/${task}_candidate"
  run_stage "pair_${task}_fast" /root/miniconda3/bin/python "${pairer}" \
    "${baseline_records}" "${fast_records}" \
    --baseline-config "${baseline_root}/run_config.txt" --method-config "${fast_root}/run_config.txt" \
    --baseline-log "${queue_root}/$(basename "${baseline_root}").log" \
    --method-log "${queue_root}/$(basename "${fast_root}").log" \
    --output-dir "${queue_root}/paired/${task}_fast"
}
pair_task humaneval "${he_base}" "${he_candidate}" "${he_fast}" \
  "$(records "${he_base}")" "$(records "${he_candidate}")" "$(records "${he_fast}")"
pair_task math500 "${math_base}" "${math_candidate}" "${math_fast}" \
  "$(records "${math_base}")" "$(records "${math_candidate}")" "$(records "${math_fast}")"
pair_task gsm8k "${gsm_base}" "${gsm_candidate}" "${gsm_fast}" \
  "$(records "${gsm_base}")" "$(records "${gsm_candidate}")" "$(records "${gsm_fast}")"
pair_task mbpp "${mbpp_base}" "${mbpp_candidate}" "${mbpp_fast}" \
  "${queue_root}/mbpp_assertion/baseline/audit_records.jsonl" \
  "${queue_root}/mbpp_assertion/candidate/audit_records.jsonl" \
  "${queue_root}/mbpp_assertion/fast/audit_records.jsonl"

audit_inputs() {
  local task=$1 arm=$2 count=$3 root=$4
  run_stage "runtime_input_${task}_${arm}" /root/miniconda3/bin/python "${runtime_auditor}" audit \
    "${root}/runtime_inputs/rank_0.jsonl" --expected-records "${count}" \
    --output "${queue_root}/runtime_inputs/${task}_${arm}_audit.json"
}
compare_inputs() {
  local task=$1 arm=$2 baseline_root=$3 candidate_root=$4
  run_stage "runtime_input_compare_${task}_${arm}" /root/miniconda3/bin/python "${runtime_auditor}" compare \
    "${baseline_root}/runtime_inputs/rank_0.jsonl" "${candidate_root}/runtime_inputs/rank_0.jsonl" \
    --output "${queue_root}/runtime_inputs/${task}_${arm}_vs_baseline.json"
}
audit_config() {
  local task=$1 arm=$2 baseline_root=$3 candidate_root=$4
  run_stage "config_${task}_${arm}" /root/miniconda3/bin/python "${config_auditor}" \
    "${baseline_root}/run_config.txt" "${candidate_root}/run_config.txt" \
    --output "${queue_root}/config_fairness/${task}_${arm}.json"
}
for item in \
  "humaneval:164:${he_base}:${he_candidate}:${he_fast}" \
  "math500:500:${math_base}:${math_candidate}:${math_fast}" \
  "gsm8k:1319:${gsm_base}:${gsm_candidate}:${gsm_fast}" \
  "mbpp:500:${mbpp_base}:${mbpp_candidate}:${mbpp_fast}"; do
  IFS=: read -r task count baseline_root candidate_root fast_root <<<"${item}"
  audit_inputs "${task}" baseline "${count}" "${baseline_root}"
  audit_inputs "${task}" candidate "${count}" "${candidate_root}"
  audit_inputs "${task}" fast "${count}" "${fast_root}"
  compare_inputs "${task}" candidate "${baseline_root}" "${candidate_root}"
  compare_inputs "${task}" fast "${baseline_root}" "${fast_root}"
  audit_config "${task}" candidate "${baseline_root}" "${candidate_root}"
  audit_config "${task}" fast "${baseline_root}" "${fast_root}"
  run_stage "leakage_${task}_candidate" /root/miniconda3/bin/python "${leakage_auditor}" \
    --source-root "${strict_root}" --run-root "${candidate_root}" \
    --expected-profile "${candidate_profile}" --output "${queue_root}/leakage_v2/${task}_candidate.json"
  run_stage "leakage_${task}_fast" /root/miniconda3/bin/python "${leakage_auditor}" \
    --source-root "${strict_root}" --run-root "${fast_root}" \
    --expected-profile symmetric_fast --output "${queue_root}/leakage_v2/${task}_fast.json"
done

run_stage offline_artifact_reverify /root/miniconda3/bin/python "${offline_freezer}" verify \
  "${offline_manifest}" --output "${queue_root}/protocol/offline_artifact_verification.json"
run_stage model_weight_reverify sha256sum -c "${weight_manifest}"

/root/miniconda3/bin/python - "${queue_root}" "${candidate_profile}" "${candidate_family}" "${selection_reason}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
tasks={}
for task in ("humaneval","math500","gsm8k","mbpp"):
    tasks[task]={
        "candidate_pair":str(root/"paired"/(task+"_candidate")/"paired_summary.json"),
        "fast_pair":str(root/"paired"/(task+"_fast")/"paired_summary.json"),
        "candidate_input_compare":str(root/"runtime_inputs"/(task+"_candidate_vs_baseline.json")),
        "fast_input_compare":str(root/"runtime_inputs"/(task+"_fast_vs_baseline.json")),
        "candidate_config_compare":str(root/"config_fairness"/(task+"_candidate.json")),
        "fast_config_compare":str(root/"config_fairness"/(task+"_fast.json")),
        "candidate_leakage":str(root/"leakage_v2"/(task+"_candidate.json")),
    }
spec={
    "schema":"strict_unified_offline_finalizer_spec_v1",
    "candidate_profile":sys.argv[2],
    "candidate_family":sys.argv[3],
    "selection_reason":sys.argv[4],
    "offline_manifest_verification":str(root/"protocol"/"offline_artifact_verification.json"),
    "tasks":tasks,
}
(root/"finalizer_spec.json").write_text(json.dumps(spec,indent=2)+"\n")
PY
run_stage finalize /root/miniconda3/bin/python "${finalizer}" \
  "${queue_root}/finalizer_spec.json" --output "${queue_root}/strict_unified_summary.json"

verify_sources
touch "${queue_root}/ACCEPTED" "${queue_root}/DONE"
echo "strict_unified_offline_three_arm_complete profile=${candidate_profile}"
