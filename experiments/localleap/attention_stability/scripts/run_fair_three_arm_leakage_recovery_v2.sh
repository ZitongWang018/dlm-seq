#!/usr/bin/env bash
set -Eeuo pipefail

canonical_root=${CANONICAL_ROOT:-/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability}
v15_root=${V15_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_admissible_lazy_guard}
v18_root=${V18_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_early_localized_conflict_repair_v2}
full4_id=${FULL4_ID:-best_framework_full4_20260719_v1}
v18_id=${V18_ID:-early_localized_evidence_conflict_repair_20260719_v2}
fair_id=${FAIR_ID:-fair_three_arm_reproduction_20260719_v1}
recovery_id=${RECOVERY_ID:-fair_three_arm_leakage_recovery_20260720_v2}
control_root=${v15_root}/results/experiment_queues/${fair_id}
queue_root=${v15_root}/results/experiment_queues/${recovery_id}
controller=${canonical_root}/scripts/run_fair_three_arm_leakage_recovery_v2.sh
auditor=${canonical_root}/audit_generation_leakage_v2.py
frozen=${queue_root}/frozen_sources.sha256
manifest=${queue_root}/recovery_manifest.tsv

mkdir -p "${queue_root}"
printf '%s\n' "$$" >"${queue_root}/controller.pid"
exec > >(tee -a "${queue_root}/formal_controller.log") 2>&1
trap 'rc=$?; echo "recovery_error rc=${rc} line=${LINENO}"; touch "${queue_root}/FAILED"; exit "${rc}"' ERR

export PATH=/root/miniconda3/bin:${PATH}
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

if [[ ! -s "${frozen}" ]]; then
  sha256sum "${controller}" "${auditor}" \
    "${canonical_root}/test_audit_generation_leakage_v2.py" >"${frozen}"
fi
verify() { sha256sum -c "${frozen}" >/dev/null; }
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
wait_terminal() {
  local root=$1 deadline=$(( $(date +%s) + 7 * 24 * 60 * 60 ))
  while [[ ! -e "${root}/DONE" && ! -e "${root}/FAILED" ]]; do
    verify
    (( $(date +%s) < deadline )) || return 21
    sleep 20
  done
}
read_tsv_value() {
  local key=$1 file=$2 row_key row_value
  while IFS=$'\t' read -r row_key row_value; do
    [[ "${row_key}" == "${key}" ]] && { printf '%s\n' "${row_value}"; return 0; }
  done <"${file}"
  return 1
}
audit_run() {
  local label=$1 source_root=$2 run_root=$3 profile=$4
  run_stage "leakage_${label}_v2" /root/miniconda3/bin/python "${auditor}" \
    --source-root "${source_root}" --run-root "${run_root}" \
    --expected-profile "${profile}" --output "${queue_root}/leakage_v2/${label}.json"
}

[[ -s "${manifest}" ]] || printf 'stage\tstatus\tstart\tfinish\texit_code_or_note\n' >"${manifest}"
verify
/root/miniconda3/bin/python -m py_compile "${auditor}"
PYTHONPATH="${canonical_root}" /root/miniconda3/bin/python \
  "${canonical_root}/test_audit_generation_leakage_v2.py"
bash -n "${controller}"

echo "waiting_for_fair_terminal=${control_root}"
wait_terminal "${control_root}"
[[ -e "${control_root}/DONE" && ! -e "${control_root}/FAILED" ]] || {
  touch "${queue_root}/BLOCKED_NONLEAKAGE_FAIR_FAILURE"; exit 22; }

actual_root=${control_root}
if [[ -s "${control_root}/REDIRECTED_QUEUE_ROOT" ]]; then
  actual_root=$(<"${control_root}/REDIRECTED_QUEUE_ROOT")
fi
selection=${actual_root}/accuracy_selection.tsv
[[ -s "${selection}" && -s "${actual_root}/fair_three_arm_summary.json" ]]
eval_root=$(read_tsv_value eval_root "${selection}")
accuracy_profile=$(read_tsv_value accuracy_profile "${selection}")
run_base=${eval_root}/results/best_symmetric_benchmarks/${fair_id}

if [[ "${accuracy_profile}" == "trajectory_early_localized_evidence_conflict_repair" ]]; then
  accuracy_base=${v18_root}/results/best_symmetric_benchmarks/${v18_id}
  accuracy_he=${accuracy_base}/humaneval/he_v18_full164
  accuracy_math=${accuracy_base}/localleap_math500/math_v18_full500
  accuracy_gsm=${accuracy_base}/gsm8k/gsm_v18_full1319
  accuracy_mbpp=${accuracy_base}/mbpp/mbpp_v18_full500
elif [[ "${accuracy_profile}" == "trajectory_early_lazy_confirmed_public_guard" ]]; then
  accuracy_base=${v15_root}/results/best_symmetric_benchmarks/${full4_id}
  accuracy_he=${accuracy_base}/humaneval/he_method_full164
  accuracy_math=${accuracy_base}/localleap_math500/math_method_full500
  accuracy_gsm=${accuracy_base}/gsm8k/gsm_method_full1319
  accuracy_mbpp=${accuracy_base}/mbpp/mbpp_method_full500
else
  printf '%s\n' "${accuracy_profile}" >"${queue_root}/BLOCKED_UNKNOWN_ACCURACY_PROFILE"
  exit 23
fi

mkdir -p "${queue_root}/leakage_v2"
run_stage leakage_static_v2 /root/miniconda3/bin/python "${auditor}" \
  --source-root "${eval_root}" --output "${queue_root}/leakage_v2/static.json"

audit_run he_accuracy "${eval_root}" "${accuracy_he}" "${accuracy_profile}"
audit_run math_accuracy "${eval_root}" "${accuracy_math}" "${accuracy_profile}"
audit_run gsm_accuracy "${eval_root}" "${accuracy_gsm}" "${accuracy_profile}"
audit_run mbpp_accuracy "${eval_root}" "${accuracy_mbpp}" "${accuracy_profile}"
audit_run he_fast "${eval_root}" "${run_base}/humaneval/he_fast_full164" symmetric_fast
audit_run math_fast "${eval_root}" "${run_base}/localleap_math500/math_fast_full500" symmetric_fast
audit_run gsm_fast "${eval_root}" "${run_base}/gsm8k/gsm_fast_full1319" symmetric_fast
audit_run mbpp_fast "${eval_root}" "${run_base}/mbpp/mbpp_fast_full500" symmetric_fast

run_stage finalize_recovery /root/miniconda3/bin/python - \
  "${actual_root}" "${queue_root}" "${accuracy_profile}" <<'PY'
import hashlib,json,sys
from pathlib import Path
old=Path(sys.argv[1]); recovery=Path(sys.argv[2]); profile=sys.argv[3]
leakage={}
for path in sorted((recovery/'leakage_v2').glob('*.json')):
    row=json.load(open(path))
    assert row['pass'], (path,row)
    leakage[path.stem]=str(path)
for task in ('humaneval','math500','gsm8k','mbpp'):
    for arm in ('accuracy','fast'):
        comp=json.load(open(old/'model_inputs'/f'{task}_{arm}_vs_baseline.json'))
        assert comp['all_equal'], (task,arm,comp)
        paired=json.load(open(old/'paired'/f'{task}_{arm}'/'paired_summary.json'))
        for key in ('prompt_hash_mismatches','target_hash_mismatches','duplicate_or_missing_ids','source_hash_mismatches'):
            assert paired[key] == 0, (task,arm,key,paired[key])
original=old/'fair_three_arm_summary.json'
summary=json.load(open(original))
assert summary['single_algorithm'] is True
assert summary['task_specific_routing'] is False
summary.update({
    'schema':'localleap_fair_three_arm_finalizer_recovered_leakage_v2',
    'accuracy_profile_reaudited':profile,
    'leakage_audit_version':'generation_information_leakage_audit_v2',
    'leakage_audit_pass':True,
    'baseline_generation_basis':'exact model-input equality plus static source audit; baseline has no selector trace',
    'original_summary_preserved':True,
    'original_summary_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
    'leakage_reports':leakage,
})
(recovery/'fair_three_arm_summary_recovered_v2.json').write_text(json.dumps(summary,indent=2)+'\n')
PY

printf 'recovered_at=%s\nevaluator=generation_information_leakage_audit_v2\noriginal_summary_preserved=true\n' \
  "$(date --iso-8601=seconds)" >"${control_root}/RECOVERED_BY_LEAKAGE_V2"
touch "${queue_root}/DONE"
echo "fair_three_arm_leakage_recovery_complete actual_root=${actual_root}"
