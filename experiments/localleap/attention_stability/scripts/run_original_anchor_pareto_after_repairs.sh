#!/usr/bin/env bash
set -Eeuo pipefail

llada_root=${LLADA_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_original_anchor_pareto_v20}
parent_root=${PARENT_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_admissible_lazy_guard}
v18_root=${V18_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_early_localized_conflict_repair_v2}
v19_root=${V19_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_sparse_context_repair_v19_direct_v3}
parent_id=${PARENT_ID:-best_framework_full4_20260719_v1}
v18_id=${V18_ID:-early_localized_evidence_conflict_repair_20260719_v2}
v19_id=${V19_ID:-sparse_context_repair_direct_20260720_v3}
queue_id=${ATTENTION_QUEUE_ID:-original_anchor_pareto_rapid_20260720_v1}
queue_root=${llada_root}/results/experiment_queues/${queue_id}
run_base=${llada_root}/results/best_symmetric_benchmarks/${queue_id}
parent_queue=${parent_root}/results/experiment_queues/${parent_id}
parent_runs=${parent_root}/results/best_symmetric_benchmarks/${parent_id}
v18_queue=${v18_root}/results/experiment_queues/${v18_id}
v19_queue=${v19_root}/results/experiment_queues/${v19_id}

controller=${llada_root}/scripts/run_original_anchor_pareto_after_repairs.sh
runner=${llada_root}/scripts/run_best_symmetric_benchmark.sh
pairer=${llada_root}/compare_paired_task_runs.py
leakage=${llada_root}/audit_generation_leakage_v2.py
mbpp_auditor=${llada_root}/audit_mbpp_assertions.py
slicer=${llada_root}/slice_audit_records.py
preregistration=${llada_root}/original_anchor_pareto_preregistration_20260720_v1.json
manifest=${queue_root}/formal_manifest.tsv
frozen=${queue_root}/frozen_sources.sha256

mkdir -p "${queue_root}"
printf '%s\n' "$$" >"${queue_root}/controller.pid"
exec > >(tee -a "${queue_root}/formal_controller.log") 2>&1
trap 'rc=$?; echo "anchor_pareto_error rc=${rc} line=${LINENO}"; touch "${queue_root}/FAILED"; exit "${rc}"' ERR

export PATH=/root/miniconda3/bin:${PATH}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true
unset HF_ENDPOINT TRANSFORMERS_CACHE || true

append_manifest() { printf '%s\n' "$1" >>"${manifest}"; }
verify() { ( cd "${llada_root}" && sha256sum -c "${frozen}" >/dev/null ); }
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
sample_file() {
  local root=$1 matches=()
  mapfile -t matches < <(find "${root}/lm_eval" -type f -name 'samples_mbpp_*.jsonl' | sort)
  [[ ${#matches[@]} -eq 1 ]] || return 24
  printf '%s\n' "${matches[0]}"
}
wait_terminal() {
  local root=$1 deadline=$(( $(date +%s) + 14 * 24 * 60 * 60 ))
  while [[ ! -e "${root}/DONE" && ! -e "${root}/FAILED" ]]; do
    verify
    (( $(date +%s) < deadline )) || return 21
    sleep 20
  done
}
reject() {
  printf '%s\n' "$1" >"${queue_root}/REJECTED"
  touch "${queue_root}/DONE"
  echo "original_anchor_pareto_rejected reason=$1"
  exit 0
}
compare_pair() {
  local label=$1 baseline_records=$2 candidate_records=$3 baseline_config=$4 candidate_config=$5
  run_stage "pair_${label}" /root/miniconda3/bin/python "${pairer}" \
    "${baseline_records}" "${candidate_records}" \
    --baseline-config "${baseline_config}" --method-config "${candidate_config}" \
    --allow-source-drift --output-dir "${queue_root}/paired/${label}"
}

[[ -s "${manifest}" ]] || printf 'stage\tstatus\tstart\tfinish\texit_code_or_note\n' >"${manifest}"
if [[ ! -s "${frozen}" ]]; then
  ( cd "${llada_root}" && sha256sum \
      generate.py eval_llada.py differential_selector.py compare_paired_task_runs.py \
      audit_generation_leakage_v2.py audit_mbpp_assertions.py slice_audit_records.py \
      scripts/run_best_symmetric_benchmark.sh \
      scripts/run_original_anchor_pareto_after_repairs.sh \
      tests/test_attention_stability.py test_original_anchor_pareto_queue_contract.py \
      original_anchor_pareto_preregistration_20260720_v1.json ) >"${frozen}"
fi
verify
run_stage source_compile /root/miniconda3/bin/python -m py_compile \
  "${llada_root}/generate.py" "${llada_root}/eval_llada.py"
run_stage selector_tests env PYTHONPATH="${llada_root}" /root/miniconda3/bin/python \
  "${llada_root}/tests/test_attention_stability.py"
run_stage queue_contract env PYTHONPATH="${llada_root}" /root/miniconda3/bin/python \
  "${llada_root}/test_original_anchor_pareto_queue_contract.py"
run_stage shell_contract bash -n "${controller}" "${runner}"
run_stage preregistration /root/miniconda3/bin/python -m json.tool "${preregistration}"
run_stage leakage_static /root/miniconda3/bin/python "${leakage}" \
  --source-root "${llada_root}" --output "${queue_root}/leakage/static.json"

echo "waiting_for_direct_v19_terminal=${v19_queue}"
wait_terminal "${v19_queue}"
[[ -e "${v19_queue}/DONE" && ! -e "${v19_queue}/FAILED" ]] || {
  touch "${queue_root}/BLOCKED_REPAIR_PIPELINE_FAILURE"; exit 22; }
if [[ -e "${v19_queue}/ACCEPTED" || -e "${v18_queue}/ACCEPTED" ]]; then
  printf 'reason=an_existing_preregistered_unified_candidate_passed\nfinished=%s\n' \
    "$(date --iso-8601=seconds)" >"${queue_root}/SKIPPED_EXISTING_CANDIDATE"
  touch "${queue_root}/DONE"
  exit 0
fi
[[ -e "${v19_queue}/REJECTED" && -e "${v18_queue}/REJECTED" ]] || {
  touch "${queue_root}/BLOCKED_REPAIR_PIPELINE_NO_DECISION"; exit 23; }

mkdir -p "${queue_root}/parents" "${queue_root}/paired" "${queue_root}/leakage"
math_parent=${parent_runs}/localleap_math500/math_baseline_full500
gsm_parent=${parent_runs}/gsm8k/gsm_baseline_full1319
mbpp_parent=${parent_runs}/mbpp/mbpp_baseline_full500
for root in "${math_parent}" "${gsm_parent}" "${mbpp_parent}"; do
  [[ -e "${root}/DONE" ]] || { touch "${queue_root}/BLOCKED_PARENT_BASELINE_MISSING"; exit 25; }
done
run_stage slice_math_parent /root/miniconda3/bin/python "${slicer}" \
  "$(records "${math_parent}")" "${queue_root}/parents/math_dev50.jsonl" --start 0 --end 50
run_stage slice_gsm_parent /root/miniconda3/bin/python "${slicer}" \
  "$(records "${gsm_parent}")" "${queue_root}/parents/gsm_dev64.jsonl" --start 0 --end 64
run_stage slice_mbpp_parent /root/miniconda3/bin/python "${slicer}" \
  "${parent_queue}/mbpp_assertion/baseline/audit_records.jsonl" \
  "${queue_root}/parents/mbpp_dev100.jsonl" --start 0 --end 100

# A fresh HumanEval anchor is generated first because historical HE baselines
# came from a different host. The remaining rapid baselines are fresh full4
# current-host runs with identical prompt/evaluator settings.
run_gpu he_anchor_baseline_dev32 0 "${runner}" humaneval 0 128 \
  baseline 0 trace he_anchor_baseline_dev32 32 256
( run_gpu he_anchor_pareto_dev32 0 "${runner}" humaneval 0 128 \
    trajectory_original_anchor_pareto 0.004 trace he_anchor_pareto_dev32 32 256 ) & p0=$!
( run_gpu gsm_anchor_pareto_dev64 1 "${runner}" gsm8k 0 128 \
    trajectory_original_anchor_pareto 0.004 trace gsm_anchor_pareto_dev64 64 256 ) & p1=$!
wait "${p0}"; wait "${p1}"

he_base=${run_base}/humaneval/he_anchor_baseline_dev32
he_candidate=${run_base}/humaneval/he_anchor_pareto_dev32
gsm_candidate=${run_base}/gsm8k/gsm_anchor_pareto_dev64
compare_pair he_dev32 "$(records "${he_base}")" "$(records "${he_candidate}")" \
  "${he_base}/run_config.txt" "${he_candidate}/run_config.txt"
compare_pair gsm_dev64 "${queue_root}/parents/gsm_dev64.jsonl" "$(records "${gsm_candidate}")" \
  "${gsm_parent}/run_config.txt" "${gsm_candidate}/run_config.txt"
if ! /root/miniconda3/bin/python - \
  "${queue_root}/paired/he_dev32/paired_summary.json" \
  "${queue_root}/paired/gsm_dev64/paired_summary.json" <<'PY'
import json,sys
he,gsm=[json.load(open(p)) for p in sys.argv[1:]]
assert he["method_correct"] > he["baseline_correct"] and he["method_only"] > he["baseline_only"], he
assert gsm["method_correct"] >= gsm["baseline_correct"], gsm
for row in (he,gsm):
    assert row["prompt_hash_mismatches"] == row["target_hash_mismatches"] == 0, row
    assert row["duplicate_or_missing_ids"] == 0, row
    assert row["method_total_nfe"] <= 2.50 * row["baseline_total_nfe"], row
PY
then reject stage_one_he_gsm_gate; fi
touch "${queue_root}/STAGE_ONE_PASS"

( run_gpu math_anchor_pareto_dev50 0 "${runner}" localleap_math500 0 128 \
    trajectory_original_anchor_pareto 0.004 trace math_anchor_pareto_dev50 50 256 ) & p0=$!
( run_gpu mbpp_anchor_pareto_dev100 1 "${runner}" mbpp 0 128 \
    trajectory_original_anchor_pareto 0.004 trace mbpp_anchor_pareto_dev100 100 256 ) & p1=$!
wait "${p0}"; wait "${p1}"
math_candidate=${run_base}/localleap_math500/math_anchor_pareto_dev50
mbpp_candidate=${run_base}/mbpp/mbpp_anchor_pareto_dev100
compare_pair math_dev50 "${queue_root}/parents/math_dev50.jsonl" "$(records "${math_candidate}")" \
  "${math_parent}/run_config.txt" "${math_candidate}/run_config.txt"
run_stage mbpp_candidate_assertions /root/miniconda3/bin/python "${mbpp_auditor}" \
  --samples "$(sample_file "${mbpp_candidate}")" --task-records "$(records "${mbpp_candidate}")" \
  --output-dir "${queue_root}/mbpp_assertion/candidate_dev100"
compare_pair mbpp_dev100 "${queue_root}/parents/mbpp_dev100.jsonl" \
  "${queue_root}/mbpp_assertion/candidate_dev100/audit_records.jsonl" \
  "${mbpp_parent}/run_config.txt" "${mbpp_candidate}/run_config.txt"

if ! /root/miniconda3/bin/python - "${queue_root}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
names=("he_dev32","gsm_dev64","math_dev50","mbpp_dev100")
rows={name:json.load(open(root/"paired"/name/"paired_summary.json")) for name in names}
for name,row in rows.items():
    assert row["method_correct"] >= row["baseline_correct"], (name,row)
    assert row["prompt_hash_mismatches"] == row["target_hash_mismatches"] == 0, (name,row)
    assert row["duplicate_or_missing_ids"] == 0, (name,row)
    assert row["method_total_nfe"] <= 2.50 * row["baseline_total_nfe"], (name,row)
assert sum(r["method_correct"] for r in rows.values()) > sum(r["baseline_correct"] for r in rows.values()), rows
(root/"rapid_summary.json").write_text(json.dumps({
    "schema":"original_anchor_pareto_rapid_gate_v1",
    "single_algorithm":True,
    "task_specific_routing":False,
    "accepted":True,
    "tasks":rows,
},indent=2)+"\n")
PY
then reject stage_two_cross_task_gate; fi

for item in "humaneval:${he_candidate}" "gsm8k:${gsm_candidate}" \
  "math500:${math_candidate}" "mbpp:${mbpp_candidate}"; do
  name=${item%%:*}; root=${item#*:}
  run_stage "leakage_${name}" /root/miniconda3/bin/python "${leakage}" \
    --source-root "${llada_root}" --run-root "${root}" \
    --expected-profile trajectory_original_anchor_pareto \
    --output "${queue_root}/leakage/${name}.json"
done
touch "${queue_root}/ACCEPTED" "${queue_root}/DONE"
echo original_anchor_pareto_rapid_gate_accepted
