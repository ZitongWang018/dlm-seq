#!/usr/bin/env bash
set -Eeuo pipefail

canonical_root=${CANONICAL_ROOT:-/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability}
llada_root=${LLADA_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_admissible_lazy_guard}
full4_id=${FULL4_ID:-best_framework_full4_20260719_v1}
full4_queue=${llada_root}/results/experiment_queues/${full4_id}
run_base=${llada_root}/results/best_symmetric_benchmarks/${full4_id}
original_he=${ORIGINAL_HE:-/root/autodl-tmp/LocalLeap/llada/results/best_symmetric_benchmarks/confirmed_bidirectional_rapid_20260719_v1/humaneval/he_base_full164}
v18_root=${V18_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_early_localized_conflict_repair_v2}
v18_id=${V18_ID:-early_localized_evidence_conflict_repair_20260719_v2}
v18_queue=${v18_root}/results/experiment_queues/${v18_id}
recovery_id=${RECOVERY_ID:-full4_leakage_recovery_20260720_v3}
queue_root=${llada_root}/results/experiment_queues/${recovery_id}
controller=${canonical_root}/scripts/run_full4_leakage_recovery_and_resume_v3.sh
leakage_auditor=${canonical_root}/audit_generation_leakage_v2.py
mbpp_auditor=/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability_dev_admissible_lazy_guard/audit_mbpp_assertions.py
pairer=${llada_root}/compare_paired_task_runs.py
manifest=${queue_root}/recovery_manifest.tsv
frozen=${queue_root}/frozen_sources.sha256

mkdir -p "${queue_root}"
printf '%s\n' "$$" >"${queue_root}/controller.pid"
exec > >(tee -a "${queue_root}/formal_controller.log") 2>&1
trap 'rc=$?; echo "recovery_error rc=${rc} line=${LINENO}"; touch "${queue_root}/FAILED"; exit "${rc}"' ERR

export PATH=/root/miniconda3/bin:${PATH}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true

if [[ ! -s "${frozen}" ]]; then
  ( cd "${llada_root}" && sha256sum generate.py eval_llada.py differential_selector.py \
      compare_paired_task_runs.py postprocess_code.py humaneval_execution.py sanitize.py ) >"${frozen}"
  sha256sum "${controller}" "${leakage_auditor}" "${mbpp_auditor}" \
    "${canonical_root}/test_audit_generation_leakage_v2.py" >>"${frozen}"
fi
verify() { ( cd "${llada_root}" && sha256sum -c "${frozen}" >/dev/null ); }
append_manifest() { printf '%s\n' "$1" >>"${manifest}"; }
run_stage() {
  local label=$1 start finish rc
  shift
  verify
  start=$(date --iso-8601=seconds)
  append_manifest "$(printf '%s\tSTARTED\t%s\t\t' "${label}" "${start}")"
  set +e
  timeout --kill-after=2m 8h "$@"
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
wait_parent_terminal() {
  local deadline=$(( $(date +%s) + 7 * 24 * 60 * 60 )) pid
  while [[ ! -e "${full4_queue}/DONE" && ! -e "${full4_queue}/FAILED" ]]; do
    verify
    (( $(date +%s) < deadline )) || return 21
    sleep 20
  done
  pid=$(cat "${full4_queue}/controller.pid")
  while kill -0 "${pid}" 2>/dev/null; do sleep 2; done
}
validate_recoverable_failure() {
  /root/miniconda3/bin/python - "${full4_queue}/formal_manifest.tsv" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter="\t"))
required={
 "he_method_full164","gsm_method_full1319","math_baseline_full500",
 "mbpp_baseline_full500","math_method_full500","gsm_baseline_full1319",
 "mbpp_method_full500",
}
done={row["stage"] for row in rows if row["status"]=="DONE"}
missing=sorted(required-done)
assert not missing, ("generation_incomplete",missing)
failed=[row["stage"] for row in rows if row["status"]=="FAILED"]
assert failed and all(stage.startswith("leakage_") for stage in failed), failed
PY
}
compare_pair() {
  local label=$1 baseline_records=$2 method_records=$3 baseline_config=$4 method_config=$5 allow=${6:-false}
  local args=()
  [[ "${allow}" == true ]] && args+=(--allow-source-drift)
  run_stage "pair_${label}" /root/miniconda3/bin/python "${pairer}" \
    "${baseline_records}" "${method_records}" \
    --baseline-config "${baseline_config}" --method-config "${method_config}" \
    "${args[@]}" --output-dir "${full4_queue}/paired/${label}"
}
resume_v18() {
  if [[ ! -e "${v18_queue}/DONE" && ! -e "${v18_queue}/FAILED" ]]; then
    nohup env SOURCE_ROOT="${canonical_root}" LLADA_ROOT="${v18_root}" \
      PARENT_ROOT="${llada_root}" PARENT_QUEUE_ID="${full4_id}" ATTENTION_QUEUE_ID="${v18_id}" \
      bash "${v18_root}/scripts/run_localized_evidence_conflict_repair_queue.sh" \
      >"${v18_queue}/launcher_recovery_v2.log" 2>&1 &
    printf '%s\n' "$!" >"${v18_queue}/launcher_recovery_v2.pid"
  fi
  printf 'reason=strict_v5_replaces_provisional_three_arm\nrecorded=%s\n' \
    "$(date --iso-8601=seconds)" >"${queue_root}/SKIPPED_REDUNDANT_FAIR"
  touch "${queue_root}/V18_RESUMED"
}

[[ -s "${manifest}" ]] || printf 'stage\tstatus\tstart\tfinish\texit_code_or_note\n' >"${manifest}"
verify
/root/miniconda3/bin/python -m py_compile "${leakage_auditor}"
PYTHONPATH="${canonical_root}" /root/miniconda3/bin/python "${canonical_root}/test_audit_generation_leakage_v2.py"
bash -n "${controller}"
echo "waiting_for_full4_terminal=${full4_queue}"
wait_parent_terminal

if [[ -e "${full4_queue}/DONE" && ! -e "${full4_queue}/FAILED" ]]; then
  touch "${queue_root}/NO_RECOVERY_NEEDED" "${queue_root}/DONE"
  resume_v18
  exit 0
fi
validate_recoverable_failure

he_method=${run_base}/humaneval/he_method_full164
math_method=${run_base}/localleap_math500/math_method_full500
gsm_method=${run_base}/gsm8k/gsm_method_full1319
mbpp_method=${run_base}/mbpp/mbpp_method_full500
math_base=${run_base}/localleap_math500/math_baseline_full500
gsm_base=${run_base}/gsm8k/gsm_baseline_full1319
mbpp_base=${run_base}/mbpp/mbpp_baseline_full500
for root in "${he_method}" "${math_method}" "${gsm_method}" "${mbpp_method}" \
  "${math_base}" "${gsm_base}" "${mbpp_base}"; do [[ -e "${root}/DONE" ]]; done

run_stage leakage_static_v2 /root/miniconda3/bin/python "${leakage_auditor}" \
  --source-root "${llada_root}" --output "${full4_queue}/leakage_v2/static.json"
for item in "humaneval:${he_method}" "math500:${math_method}" "gsm8k:${gsm_method}" "mbpp:${mbpp_method}"; do
  name=${item%%:*}; root=${item#*:}
  run_stage "leakage_${name}_v2" /root/miniconda3/bin/python "${leakage_auditor}" \
    --source-root "${llada_root}" --run-root "${root}" \
    --expected-profile trajectory_early_lazy_confirmed_public_guard \
    --output "${full4_queue}/leakage_v2/${name}.json"
done

mkdir -p "${full4_queue}/mbpp_assertion"
run_stage mbpp_assertion_method_v2 /root/miniconda3/bin/python "${mbpp_auditor}" \
  --samples "$(sample_file "${mbpp_method}")" --task-records "$(records "${mbpp_method}")" \
  --output-dir "${full4_queue}/mbpp_assertion/method"
run_stage mbpp_assertion_baseline_v2 /root/miniconda3/bin/python "${mbpp_auditor}" \
  --samples "$(sample_file "${mbpp_base}")" --task-records "$(records "${mbpp_base}")" \
  --output-dir "${full4_queue}/mbpp_assertion/baseline"

mkdir -p "${full4_queue}/paired"
compare_pair humaneval "$(records "${original_he}")" "$(records "${he_method}")" \
  "${original_he}/run_config.txt" "${he_method}/run_config.txt" true
compare_pair math500 "$(records "${math_base}")" "$(records "${math_method}")" \
  "${math_base}/run_config.txt" "${math_method}/run_config.txt"
compare_pair gsm8k "$(records "${gsm_base}")" "$(records "${gsm_method}")" \
  "${gsm_base}/run_config.txt" "${gsm_method}/run_config.txt"
compare_pair mbpp "${full4_queue}/mbpp_assertion/baseline/audit_records.jsonl" \
  "${full4_queue}/mbpp_assertion/method/audit_records.jsonl" \
  "${mbpp_base}/run_config.txt" "${mbpp_method}/run_config.txt"

/root/miniconda3/bin/python - "${full4_queue}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
summary={
 "schema":"best_framework_full4_summary_recovered_leakage_v2",
 "selected_family":"v15_admissible_early_abort",
 "selected_profile":"trajectory_early_lazy_confirmed_public_guard",
 "single_algorithm":True,
 "leakage_audit_version":"generation_information_leakage_audit_v2",
 "leakage_audit_pass":True,
 "original_failed_marker_preserved":True,
 "tasks":{},
}
for task in ("humaneval","math500","gsm8k","mbpp"):
 row=json.load(open(root/"paired"/task/"paired_summary.json"))
 summary["tasks"][task]=row
(root/"full4_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
PY
printf 'recovered_at=%s\nevaluator=generation_information_leakage_audit_v2\noriginal_failed_preserved=true\n' \
  "$(date --iso-8601=seconds)" >"${full4_queue}/RECOVERED_BY_LEAKAGE_V2"
touch "${full4_queue}/DONE" "${queue_root}/DONE"
resume_v18
echo "full4_leakage_recovery_complete"
