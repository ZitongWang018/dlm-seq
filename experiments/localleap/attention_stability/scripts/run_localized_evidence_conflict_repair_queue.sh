#!/usr/bin/env bash
set -Eeuo pipefail

source_root=${SOURCE_ROOT:-/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability}
llada_root=${LLADA_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_early_localized_conflict_repair_v3}
parent_root=${PARENT_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_admissible_lazy_guard}
parent_queue_id=${PARENT_QUEUE_ID:-best_framework_full4_20260719_v1}
queue_id=${ATTENTION_QUEUE_ID:-early_localized_evidence_conflict_repair_20260720_v3}
profile=${PROFILE:-trajectory_early_localized_evidence_conflict_repair}
run_prefix=${RUN_PREFIX:-v18}
preregistration=${PREREGISTRATION:-early_localized_evidence_conflict_repair_preregistration_20260719_v2.json}
leakage_auditor=${LEAKAGE_AUDITOR:-audit_generation_leakage_v2.py}
queue_root=${llada_root}/results/experiment_queues/${queue_id}
run_base=${llada_root}/results/best_symmetric_benchmarks/${queue_id}
parent_queue=${parent_root}/results/experiment_queues/${parent_queue_id}
parent_runs=${parent_root}/results/best_symmetric_benchmarks/${parent_queue_id}
runner=${llada_root}/run_best_symmetric_benchmark.sh
controller=${llada_root}/scripts/run_localized_evidence_conflict_repair_queue.sh
pairer=${llada_root}/compare_paired_task_runs.py
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
  timeout --kill-after=5m 48h "$@"
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
  run_stage "${label}" env CUDA_VISIBLE_DEVICES="${gpu}" \
    LLADA_ROOT="${llada_root}" ATTENTION_QUEUE_ID="${queue_id}" "$@"
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
wait_parent() {
  local deadline=$(( $(date +%s) + 72 * 60 * 60 ))
  while [[ ! -e "${parent_queue}/DONE" && ! -e "${parent_queue}/FAILED" ]]; do
    if (( $(date +%s) >= deadline )); then
      touch "${queue_root}/FAILED_PARENT_TIMEOUT" "${queue_root}/FAILED"
      exit 21
    fi
    sleep 20
  done
  [[ -e "${parent_queue}/DONE" ]] || {
    touch "${queue_root}/BLOCKED_PARENT_FAILED" "${queue_root}/FAILED"; exit 22; }
}
sample_file() {
  local run_root=$1 matches=()
  mapfile -t matches < <(find "${run_root}/lm_eval" -type f -name 'samples_mbpp_*.jsonl' | sort)
  [[ ${#matches[@]} -eq 1 ]] || return 23
  printf '%s\n' "${matches[0]}"
}
compare_pair() {
  local label=$1 parent_records=$2 candidate_records=$3 parent_config=$4 candidate_config=$5
  run_stage "compare_${label}" /root/miniconda3/bin/python "${pairer}" \
    "${parent_records}" "${candidate_records}" \
    --baseline-config "${parent_config}" --method-config "${candidate_config}" \
    --allow-source-drift --output-dir "${queue_root}/paired/${label}"
}
reject() {
  local reason=$1
  printf '%s\n' "${reason}" >"${queue_root}/REJECTED"
  append_manifest "$(printf 'promotion\tREJECTED\t%s\t%s\t%s' \
    "$(date --iso-8601=seconds)" "$(date --iso-8601=seconds)" "${reason}")"
  touch "${queue_root}/DONE"
  exit 0
}

export PATH=/root/miniconda3/bin:${PATH}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1 HF_ALLOW_CODE_EVAL=1
unset HF_ENDPOINT TRANSFORMERS_CACHE || true
cd "${llada_root}"
[[ "${run_base}" == */"${queue_id}" ]] || exit 20
[[ -s "${manifest}" ]] || printf 'stage\tstatus\tstart\tfinish\texit_code_or_note\n' >"${manifest}"

if [[ ! -s "${frozen}" ]]; then
  sha256sum generate.py eval_llada.py differential_selector.py \
    run_best_symmetric_benchmark.sh compare_paired_task_runs.py \
    "${leakage_auditor}" audit_mbpp_assertions.py slice_audit_by_index.py \
    postprocess_code.py humaneval_execution.py sanitize.py \
    "${preregistration}" \
    tests/test_attention_stability.py tests/test_differential_selector.py \
    test_localized_repair_queue_contract.py test_slice_audit_by_index.py \
    "${controller}" >"${frozen}"
fi
verify
/root/miniconda3/bin/python -m py_compile generate.py eval_llada.py \
  differential_selector.py compare_paired_task_runs.py audit_mbpp_assertions.py
bash -n "${runner}" "${controller}"
run_stage queue_contract /root/miniconda3/bin/python \
  test_localized_repair_queue_contract.py
run_stage slice_contract /root/miniconda3/bin/python \
  test_slice_audit_by_index.py
run_stage leakage_static /root/miniconda3/bin/python "${leakage_auditor}" \
  --source-root "${llada_root}" --expected-profile "${profile}" \
  --output "${queue_root}/leakage/static.json"

echo "waiting_for_parent_queue=${parent_queue}"
wait_parent

parent_he=${parent_runs}/humaneval/he_method_full164
parent_math=${parent_runs}/localleap_math500/math_method_full500
parent_gsm=${parent_runs}/gsm8k/gsm_method_full1319
parent_mbpp=${parent_runs}/mbpp/mbpp_method_full500
for root in "${parent_he}" "${parent_math}" "${parent_gsm}" "${parent_mbpp}"; do
  [[ -e "${root}/DONE" ]] || { touch "${queue_root}/FAILED_PARENT_INCOMPLETE"; exit 24; }
done

mkdir -p "${queue_root}/parents"
/root/miniconda3/bin/python slice_audit_by_index.py \
  "$(records "${parent_math}")" "${queue_root}/parents/math_dev50.jsonl" --start 0 --end 50
/root/miniconda3/bin/python slice_audit_by_index.py \
  "$(records "${parent_gsm}")" "${queue_root}/parents/gsm_dev64.jsonl" --start 0 --end 64

( run_gpu "he_${run_prefix}_full164" 0 "${runner}" humaneval 0 128 \
    "${profile}" 0.004 trace "he_${run_prefix}_full164" 164 256 ) & p0=$!
( run_gpu "mbpp_${run_prefix}_dev100" 1 "${runner}" mbpp 0 128 \
    "${profile}" 0.004 trace "mbpp_${run_prefix}_dev100" 100 256 ) & p1=$!
wait "${p0}"; wait "${p1}"
he_full=${run_base}/humaneval/he_${run_prefix}_full164
mbpp_dev=${run_base}/mbpp/mbpp_${run_prefix}_dev100
compare_pair he_full164 "$(records "${parent_he}")" "$(records "${he_full}")" \
  "${parent_he}/run_config.txt" "${he_full}/run_config.txt"
run_stage mbpp_dev_assertions /root/miniconda3/bin/python audit_mbpp_assertions.py \
  --samples "$(sample_file "${mbpp_dev}")" --task-records "$(records "${mbpp_dev}")" \
  --output-dir "${queue_root}/mbpp_assertion/dev100"
/root/miniconda3/bin/python slice_audit_by_index.py \
  "${parent_queue}/mbpp_assertion/method/audit_records.jsonl" \
  "${queue_root}/parents/mbpp_dev100.jsonl" --start 0 --end 100
compare_pair mbpp_dev100 "${queue_root}/parents/mbpp_dev100.jsonl" \
  "${queue_root}/mbpp_assertion/dev100/audit_records.jsonl" \
  "${parent_mbpp}/run_config.txt" "${mbpp_dev}/run_config.txt"
if ! /root/miniconda3/bin/python - \
  "${queue_root}/paired/he_full164/paired_summary.json" \
  "${queue_root}/paired/mbpp_dev100/paired_summary.json" <<'PY'
import json,sys
he,mbpp=[json.load(open(p)) for p in sys.argv[1:]]
assert he["method_correct"] > he["baseline_correct"], he
assert he["method_only"] > he["baseline_only"], he
assert mbpp["method_correct"] >= mbpp["baseline_correct"], mbpp
for x in (he,mbpp):
    assert x["prompt_hash_mismatches"] == 0 and x["target_hash_mismatches"] == 0, x
    assert x["duplicate_or_missing_ids"] == 0, x
    assert x["method_total_nfe"] <= 1.05 * x["baseline_total_nfe"], x
PY
then reject "code_domain_gate"; fi
touch "${queue_root}/CODE_DOMAIN_GATE_PASS"

( run_gpu "math_${run_prefix}_dev50" 0 "${runner}" localleap_math500 0 128 \
    "${profile}" 0.004 trace "math_${run_prefix}_dev50" 50 256 ) & p0=$!
( run_gpu "gsm_${run_prefix}_dev64" 1 "${runner}" gsm8k 0 128 \
    "${profile}" 0.004 trace "gsm_${run_prefix}_dev64" 64 256 ) & p1=$!
wait "${p0}"; wait "${p1}"
math_dev=${run_base}/localleap_math500/math_${run_prefix}_dev50
gsm_dev=${run_base}/gsm8k/gsm_${run_prefix}_dev64
compare_pair math_dev50 "${queue_root}/parents/math_dev50.jsonl" "$(records "${math_dev}")" \
  "${parent_math}/run_config.txt" "${math_dev}/run_config.txt"
compare_pair gsm_dev64 "${queue_root}/parents/gsm_dev64.jsonl" "$(records "${gsm_dev}")" \
  "${parent_gsm}/run_config.txt" "${gsm_dev}/run_config.txt"
if ! /root/miniconda3/bin/python - \
  "${queue_root}/paired/math_dev50/paired_summary.json" \
  "${queue_root}/paired/gsm_dev64/paired_summary.json" <<'PY'
import json,sys
rows=[json.load(open(p)) for p in sys.argv[1:]]
assert all(x["method_correct"] >= x["baseline_correct"] for x in rows), rows
assert sum(x["method_correct"] for x in rows) > sum(x["baseline_correct"] for x in rows), rows
for x in rows:
    assert x["prompt_hash_mismatches"] == 0 and x["target_hash_mismatches"] == 0, x
    assert x["duplicate_or_missing_ids"] == 0, x
    assert x["method_total_nfe"] <= 1.05 * x["baseline_total_nfe"], x
PY
then reject "math_gsm_development_gate"; fi
touch "${queue_root}/CROSS_TASK_GATE_PASS" "${queue_root}/FULL_PROMOTION_PASS"

( run_gpu "math_${run_prefix}_full500" 0 "${runner}" localleap_math500 0 128 \
    "${profile}" 0.004 trace "math_${run_prefix}_full500"
  run_gpu "mbpp_${run_prefix}_full500" 0 "${runner}" mbpp 0 128 \
    "${profile}" 0.004 trace "mbpp_${run_prefix}_full500" ) & p0=$!
( run_gpu "gsm_${run_prefix}_full1319" 1 "${runner}" gsm8k 0 128 \
    "${profile}" 0.004 trace "gsm_${run_prefix}_full1319" ) & p1=$!
wait "${p0}"; wait "${p1}"
math_full=${run_base}/localleap_math500/math_${run_prefix}_full500
gsm_full=${run_base}/gsm8k/gsm_${run_prefix}_full1319
mbpp_full=${run_base}/mbpp/mbpp_${run_prefix}_full500
compare_pair math_full500 "$(records "${parent_math}")" "$(records "${math_full}")" \
  "${parent_math}/run_config.txt" "${math_full}/run_config.txt"
compare_pair gsm_full1319 "$(records "${parent_gsm}")" "$(records "${gsm_full}")" \
  "${parent_gsm}/run_config.txt" "${gsm_full}/run_config.txt"
run_stage mbpp_full_assertions /root/miniconda3/bin/python audit_mbpp_assertions.py \
  --samples "$(sample_file "${mbpp_full}")" --task-records "$(records "${mbpp_full}")" \
  --output-dir "${queue_root}/mbpp_assertion/full500"
compare_pair mbpp_full500 "${parent_queue}/mbpp_assertion/method/audit_records.jsonl" \
  "${queue_root}/mbpp_assertion/full500/audit_records.jsonl" \
  "${parent_mbpp}/run_config.txt" "${mbpp_full}/run_config.txt"

if ! /root/miniconda3/bin/python - "${queue_root}" "${profile}" "${run_prefix}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
profile=sys.argv[2]
run_prefix=sys.argv[3]
names=("he_full164","math_full500","gsm_full1319","mbpp_full500")
rows={name:json.load(open(root/"paired"/name/"paired_summary.json")) for name in names}
for name,x in rows.items():
    assert x["method_correct"] >= x["baseline_correct"], (name,x)
    assert x["prompt_hash_mismatches"] == 0 and x["target_hash_mismatches"] == 0, (name,x)
    assert x["duplicate_or_missing_ids"] == 0, (name,x)
    assert x["method_total_nfe"] <= 1.05 * x["baseline_total_nfe"], (name,x)
assert sum(x["method_correct"] for x in rows.values()) > sum(x["baseline_correct"] for x in rows.values()), rows
summary={
    "schema":f"{run_prefix}_localized_repair_full4_v1",
    "single_algorithm":True,
    "profile":profile,
    "accepted":True,
    "tasks":rows,
}
(root/"full4_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
PY
then reject "full4_nonregression_gate"; fi

for item in "humaneval:${he_full}" "math500:${math_full}" "gsm8k:${gsm_full}" "mbpp:${mbpp_full}"; do
  name=${item%%:*}; run=${item#*:}
  run_stage "leakage_${name}" /root/miniconda3/bin/python "${leakage_auditor}" \
    --source-root "${llada_root}" --run-root "${run}" \
    --expected-profile "${profile}" --output "${queue_root}/leakage/${name}.json"
done

touch "${queue_root}/ACCEPTED" "${queue_root}/DONE"
echo "localized_evidence_conflict_repair_queue_complete profile=${profile}"
