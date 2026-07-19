#!/usr/bin/env bash
set -Eeuo pipefail

source_root=${SOURCE_ROOT:-/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability_dev_admissible_lazy_guard}
llada_root=${LLADA_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_admissible_lazy_guard}
v15_queue=${V15_QUEUE:-${llada_root}/results/experiment_queues/admissible_lazy_guard_recovery_v2_20260719}
v11_queue=${V11_QUEUE:-/root/autodl-tmp/LocalLeap/llada_slot_public_guard/results/experiment_queues/public_example_guard_20260719_v1}
v9_runs=${V9_RUNS:-/root/autodl-tmp/LocalLeap/llada_slot_confirmed_block/results/best_symmetric_benchmarks/confirmed_bidirectional_rapid_20260719_v1}
original_he=${ORIGINAL_HE:-/root/autodl-tmp/LocalLeap/llada/results/best_symmetric_benchmarks/confirmed_bidirectional_rapid_20260719_v1/humaneval/he_base_full164}
queue_id=${ATTENTION_QUEUE_ID:-best_framework_full4_20260719_v1}
queue_root=${llada_root}/results/experiment_queues/${queue_id}
run_base=${llada_root}/results/best_symmetric_benchmarks/${queue_id}
runner=${llada_root}/run_best_symmetric_benchmark.sh
controller=${source_root}/scripts/run_best_framework_full4_queue.sh
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
wait_terminal() {
  local root=$1 deadline=$(( $(date +%s) + 48 * 60 * 60 ))
  while [[ ! -e "${root}/DONE" && ! -e "${root}/FAILED" ]]; do
    if (( $(date +%s) >= deadline )); then
      touch "${queue_root}/FAILED_PARENT_TIMEOUT" "${queue_root}/FAILED"
      exit 21
    fi
    sleep 20
  done
}
wait_for_file() {
  local path=$1 deadline=$(( $(date +%s) + 48 * 60 * 60 ))
  while [[ ! -e "${path}" ]]; do
    if (( $(date +%s) >= deadline )); then return 21; fi
    sleep 20
  done
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

echo "waiting_for_v15_terminal=${v15_queue}"
wait_terminal "${v15_queue}"
wait_for_file "${v11_queue}/DONE"
wait_for_file "${original_he}/DONE"

if [[ ! -s "${frozen}" ]]; then
  sha256sum generate.py eval_llada.py run_best_symmetric_benchmark.sh \
    compare_paired_task_runs.py "${source_root}/audit_mbpp_assertions.py" \
    "${source_root}/audit_generation_leakage.py" \
    "${source_root}/compare_exact_output_runs.py" "${controller}" \
    "${source_root}/test_audit_mbpp_assertions.py" \
    "${source_root}/test_compare_exact_output_runs.py" \
    "${source_root}/test_full4_queue_contract.py" >"${frozen}"
fi
verify
/root/miniconda3/bin/python -m py_compile generate.py eval_llada.py \
  compare_paired_task_runs.py "${source_root}/audit_mbpp_assertions.py" \
  "${source_root}/compare_exact_output_runs.py"
PYTHONPATH="${source_root}" /root/miniconda3/bin/python "${source_root}/test_audit_mbpp_assertions.py"
PYTHONPATH="${source_root}" /root/miniconda3/bin/python "${source_root}/test_compare_exact_output_runs.py"
/root/miniconda3/bin/python "${source_root}/test_full4_queue_contract.py"
bash -n "${runner}" "${controller}"
run_stage leakage_static /root/miniconda3/bin/python \
  "${source_root}/audit_generation_leakage.py" --source-root "${llada_root}" \
  --output "${queue_root}/leakage/static.json"

selected_family=v11_confirmed_public_guard
selected_profile=trajectory_confirmed_public_guard
if [[ -e "${v15_queue}/DONE" ]] && /root/miniconda3/bin/python - \
  "${v15_queue}/exact/he_full164/exact_summary.json" \
  "${v15_queue}/exact/math_n50/exact_summary.json" \
  "${v15_queue}/exact/mbpp_n100/exact_summary.json" \
  "${v15_queue}/exact/gsm_n64/exact_summary.json" <<'PY'
import json,sys
for path in sys.argv[1:]:
    row=json.load(open(path))
    assert row["all_invariants_pass"], (path,row)
    assert row["nfe_reduction"] > 0, (path,row)
PY
then
  selected_family=v15_admissible_early_abort
  selected_profile=trajectory_early_lazy_confirmed_public_guard
fi
printf 'selected_family\t%s\nselected_profile\t%s\nselected_at\t%s\n' \
  "${selected_family}" "${selected_profile}" "$(date --iso-8601=seconds)" \
  >"${queue_root}/selection.tsv"
echo "selected_family=${selected_family} selected_profile=${selected_profile}"

printf 'task\tprofile\tlimit\n' >"${queue_root}/selected_profiles.tsv"
printf 'humaneval\t%s\t164\nlocalleap_math500\t%s\t500\ngsm8k\t%s\t1319\nmbpp\t%s\t500\n' \
  "${selected_profile}" "${selected_profile}" \
  "${selected_profile}" "${selected_profile}" \
  >>"${queue_root}/selected_profiles.tsv"

# GPU0: short code result first, then the long GSM method, followed by two
# cheaper original-LLaDA baselines. GPU1 handles MATH, its baseline, GSM
# baseline, then full MBPP. This balances the measured per-sample runtimes.
(
  run_gpu he_method_full164 0 "${runner}" humaneval 0 128 \
    "${selected_profile}" 0.004 trace he_method_full164
  run_gpu gsm_method_full1319 0 "${runner}" gsm8k 0 128 \
    "${selected_profile}" 0.004 trace gsm_method_full1319
  run_gpu math_baseline_full500 0 "${runner}" localleap_math500 0 128 \
    baseline 0 trace math_baseline_full500
  run_gpu mbpp_baseline_full500 0 "${runner}" mbpp 0 128 \
    baseline 0 trace mbpp_baseline_full500
) & worker0=$!
(
  run_gpu math_method_full500 1 "${runner}" localleap_math500 0 128 \
    "${selected_profile}" 0.004 trace math_method_full500
  run_gpu gsm_baseline_full1319 1 "${runner}" gsm8k 0 128 \
    baseline 0 trace gsm_baseline_full1319
  run_gpu mbpp_method_full500 1 "${runner}" mbpp 0 128 \
    "${selected_profile}" 0.004 trace mbpp_method_full500
) & worker1=$!
wait "${worker0}"
wait "${worker1}"

he_method=${run_base}/humaneval/he_method_full164
math_method=${run_base}/localleap_math500/math_method_full500
gsm_method=${run_base}/gsm8k/gsm_method_full1319
mbpp_method=${run_base}/mbpp/mbpp_method_full500
math_base=${run_base}/localleap_math500/math_baseline_full500
gsm_base=${run_base}/gsm8k/gsm_baseline_full1319
mbpp_base=${run_base}/mbpp/mbpp_baseline_full500

for item in \
  "humaneval:${he_method}" \
  "math500:${math_method}" \
  "gsm8k:${gsm_method}" \
  "mbpp:${mbpp_method}"
do
  task_name=${item%%:*}
  task_root=${item#*:}
  run_stage "leakage_${task_name}" /root/miniconda3/bin/python \
    "${source_root}/audit_generation_leakage.py" --source-root "${llada_root}" \
    --run-root "${task_root}" --expected-profile "${selected_profile}" \
    --output "${queue_root}/leakage/${task_name}.json"
done

mkdir -p "${queue_root}/mbpp_assertion/method" "${queue_root}/mbpp_assertion/baseline"
run_stage mbpp_assertion_method /root/miniconda3/bin/python \
  "${source_root}/audit_mbpp_assertions.py" \
  --samples "$(sample_file "${mbpp_method}")" \
  --task-records "$(records "${mbpp_method}")" \
  --output-dir "${queue_root}/mbpp_assertion/method"
run_stage mbpp_assertion_baseline /root/miniconda3/bin/python \
  "${source_root}/audit_mbpp_assertions.py" \
  --samples "$(sample_file "${mbpp_base}")" \
  --task-records "$(records "${mbpp_base}")" \
  --output-dir "${queue_root}/mbpp_assertion/baseline"

mkdir -p "${queue_root}/paired"
run_stage compare_he /root/miniconda3/bin/python compare_paired_task_runs.py \
  "$(records "${original_he}")" "$(records "${he_method}")" \
  --baseline-config "${original_he}/run_config.txt" \
  --method-config "${he_method}/run_config.txt" --allow-source-drift \
  --output-dir "${queue_root}/paired/humaneval"
run_stage compare_math /root/miniconda3/bin/python compare_paired_task_runs.py \
  "$(records "${math_base}")" "$(records "${math_method}")" \
  --baseline-config "${math_base}/run_config.txt" \
  --method-config "${math_method}/run_config.txt" --allow-source-drift \
  --output-dir "${queue_root}/paired/math500"
run_stage compare_gsm /root/miniconda3/bin/python compare_paired_task_runs.py \
  "$(records "${gsm_base}")" "$(records "${gsm_method}")" \
  --baseline-config "${gsm_base}/run_config.txt" \
  --method-config "${gsm_method}/run_config.txt" --allow-source-drift \
  --output-dir "${queue_root}/paired/gsm8k"
run_stage compare_mbpp /root/miniconda3/bin/python compare_paired_task_runs.py \
  "${queue_root}/mbpp_assertion/baseline/audit_records.jsonl" \
  "${queue_root}/mbpp_assertion/method/audit_records.jsonl" \
  --baseline-config "${mbpp_base}/run_config.txt" \
  --method-config "${mbpp_method}/run_config.txt" --allow-source-drift \
  --output-dir "${queue_root}/paired/mbpp"

/root/miniconda3/bin/python - "${queue_root}" "${selected_family}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
tasks={
    "humaneval":"humaneval",
    "math500":"math500",
    "gsm8k":"gsm8k",
    "mbpp":"mbpp",
}
summary={"schema":"best_framework_full4_summary_v1","selected_family":sys.argv[2],"single_algorithm":True,"leakage_audit_pass":True,"tasks":{}}
for name,directory in tasks.items():
    data=json.load(open(root/"paired"/directory/"paired_summary.json"))
    summary["tasks"][name]={
        key:data[key] for key in (
            "total","baseline_correct","method_correct","baseline_accuracy",
            "method_accuracy","baseline_only","method_only","baseline_total_nfe",
            "method_total_nfe","nfe_ratio_method_over_baseline",
            "wall_speedup_baseline_over_method","prompt_hash_mismatches",
            "target_hash_mismatches","duplicate_or_missing_ids"
        )
    }
(root/"full4_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,sort_keys=True))
PY

touch "${queue_root}/DONE"
echo "best_framework_full4_queue_complete selected_family=${selected_family}"
