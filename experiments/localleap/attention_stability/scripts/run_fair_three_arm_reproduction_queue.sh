#!/usr/bin/env bash
set -Eeuo pipefail

canonical_root=${CANONICAL_ROOT:-/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability}
v15_root=${V15_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_admissible_lazy_guard}
v18_root=${V18_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_early_localized_conflict_repair_v2}
full4_id=${FULL4_ID:-best_framework_full4_20260719_v1}
v18_id=${V18_ID:-early_localized_evidence_conflict_repair_20260719_v2}
queue_id=${ATTENTION_QUEUE_ID:-fair_three_arm_reproduction_20260719_v1}
model_path=${MODEL_PATH:-/root/autodl-tmp/model/LLaDA/instruct}
full4_queue=${v15_root}/results/experiment_queues/${full4_id}
v18_queue=${v18_root}/results/experiment_queues/${v18_id}
controller=${canonical_root}/scripts/run_fair_three_arm_reproduction_queue.sh
input_auditor=${canonical_root}/audit_model_input_hashes.py
finalizer=${canonical_root}/finalize_fair_three_arm.py
preregistration=${canonical_root}/fair_three_arm_preregistration_20260719_v1.json

# The queue root is deliberately outside either candidate result tree until the
# accepted accuracy runtime is known. The launcher pre-creates this directory.
control_root=${CONTROL_ROOT:-${v15_root}/results/experiment_queues/${queue_id}}
mkdir -p "${control_root}"
printf '%s\n' "$$" >"${control_root}/controller.pid"
exec > >(tee -a "${control_root}/formal_controller.log") 2>&1
trap 'rc=$?; echo "controller_error rc=${rc} line=${LINENO}"; touch "${control_root}/FAILED"; exit "${rc}"' ERR

controller_frozen=${control_root}/controller.sha256
if [[ ! -s "${controller_frozen}" ]]; then
  sha256sum "${controller}" "${input_auditor}" "${finalizer}" \
    "${preregistration}" >"${controller_frozen}"
fi
verify_controller() { sha256sum -c "${controller_frozen}" >/dev/null; }

wait_terminal() {
  local root=$1 deadline=$(( $(date +%s) + 7 * 24 * 60 * 60 ))
  while [[ ! -e "${root}/DONE" && ! -e "${root}/FAILED" ]]; do
    verify_controller
    if (( $(date +%s) >= deadline )); then
      printf '%s\n' "${root}" >"${control_root}/FAILED_PARENT_TIMEOUT"
      return 21
    fi
    sleep 20
  done
}

export PATH=/root/miniconda3/bin:${PATH}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
unset HF_ENDPOINT || true
unset TRANSFORMERS_CACHE || true

echo "waiting_for_full4=${full4_queue}"
wait_terminal "${full4_queue}"
[[ -e "${full4_queue}/DONE" ]] || {
  touch "${control_root}/BLOCKED_FULL4_FAILED"; exit 22; }
echo "waiting_for_v18=${v18_queue}"
wait_terminal "${v18_queue}"

if [[ -e "${v18_queue}/ACCEPTED" && -e "${v18_queue}/DONE" ]]; then
  eval_root=${v18_root}
  accuracy_profile=trajectory_early_localized_evidence_conflict_repair
  accuracy_family=v18_early_localized_evidence_conflict_repair
  accuracy_queue=${v18_queue}
  accuracy_base=${v18_root}/results/best_symmetric_benchmarks/${v18_id}
  accuracy_he=${accuracy_base}/humaneval/he_v18_full164
  accuracy_math=${accuracy_base}/localleap_math500/math_v18_full500
  accuracy_gsm=${accuracy_base}/gsm8k/gsm_v18_full1319
  accuracy_mbpp=${accuracy_base}/mbpp/mbpp_v18_full500
  accuracy_mbpp_records=${v18_queue}/mbpp_assertion/full500/audit_records.jsonl
  leakage_auditor=${v18_root}/audit_generation_leakage.py
  mbpp_auditor=${v18_root}/audit_mbpp_assertions.py
  selection_reason=v18_preregistered_full4_accepted
else
  eval_root=${v15_root}
  accuracy_profile=trajectory_early_lazy_confirmed_public_guard
  accuracy_family=v15_exact_speed_v11_descendant
  accuracy_queue=${full4_queue}
  accuracy_base=${v15_root}/results/best_symmetric_benchmarks/${full4_id}
  accuracy_he=${accuracy_base}/humaneval/he_method_full164
  accuracy_math=${accuracy_base}/localleap_math500/math_method_full500
  accuracy_gsm=${accuracy_base}/gsm8k/gsm_method_full1319
  accuracy_mbpp=${accuracy_base}/mbpp/mbpp_method_full500
  accuracy_mbpp_records=${full4_queue}/mbpp_assertion/method/audit_records.jsonl
  leakage_auditor=/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability_dev_admissible_lazy_guard/audit_generation_leakage.py
  mbpp_auditor=/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability_dev_admissible_lazy_guard/audit_mbpp_assertions.py
  selection_reason=v18_not_accepted_fallback_to_v15_v11_family
fi

runner=${eval_root}/run_best_symmetric_benchmark.sh
pairer=${eval_root}/compare_paired_task_runs.py
queue_root=${eval_root}/results/experiment_queues/${queue_id}
run_base=${eval_root}/results/best_symmetric_benchmarks/${queue_id}
manifest=${queue_root}/formal_manifest.tsv
frozen=${queue_root}/frozen_sources.sha256

if [[ "${control_root}" != "${queue_root}" ]]; then
  mkdir -p "${queue_root}"
  cp "${control_root}/controller.pid" "${queue_root}/controller.pid"
  printf '%s\n' "${queue_root}" >"${control_root}/REDIRECTED_QUEUE_ROOT"
fi
[[ "${run_base}" == */"${queue_id}" ]] || exit 20
[[ -s "${manifest}" ]] || printf 'stage\tstatus\tstart\tfinish\texit_code_or_note\n' >"${manifest}"
printf 'accuracy_family\t%s\naccuracy_profile\t%s\nselection_reason\t%s\neval_root\t%s\nselected_at\t%s\n' \
  "${accuracy_family}" "${accuracy_profile}" "${selection_reason}" "${eval_root}" \
  "$(date --iso-8601=seconds)" >"${queue_root}/accuracy_selection.tsv"

if [[ ! -s "${frozen}" ]]; then
  ( cd "${eval_root}" && sha256sum generate.py eval_llada.py differential_selector.py \
      run_best_symmetric_benchmark.sh compare_paired_task_runs.py \
      postprocess_code.py humaneval_execution.py sanitize.py ) \
    >"${frozen}"
  sha256sum "${controller}" "${input_auditor}" "${finalizer}" \
    "${preregistration}" "${canonical_root}/test_audit_model_input_hashes.py" \
    "${canonical_root}/test_finalize_fair_three_arm.py" \
    "${canonical_root}/test_fair_three_arm_queue_contract.py" \
    "${mbpp_auditor}" "${leakage_auditor}" >>"${frozen}"
fi
verify() {
  verify_controller
  ( cd "${eval_root}" && sha256sum -c "${frozen}" >/dev/null )
}
append_manifest() { printf '%s\n' "$1" >>"${manifest}"; }
run_stage() {
  local label=$1 start finish rc
  shift
  verify
  start=$(date --iso-8601=seconds)
  append_manifest "$(printf '%s\tSTARTED\t%s\t\t' "${label}" "${start}")"
  set +e
  timeout --kill-after=5m 72h "$@"
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
  run_stage "${label}" env CUDA_VISIBLE_DEVICES="${gpu}" LLADA_ROOT="${eval_root}" \
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
  local task=$1 run_root=$2 matches=()
  mapfile -t matches < <(find "${run_root}/lm_eval" -type f -name "samples_${task}_*.jsonl" | sort)
  [[ ${#matches[@]} -eq 1 ]] || {
    echo "expected one sample file task=${task} root=${run_root}, got ${#matches[@]}" >&2
    return 24
  }
  printf '%s\n' "${matches[0]}"
}
run_task_pair() {
  local task=$1 shots=$2 baseline_tag=$3 fast_tag=$4 p0 p1 rc0 rc1
  ( run_gpu "${baseline_tag}" 0 "${runner}" "${task}" "${shots}" 128 \
      baseline 0 trace "${baseline_tag}" full 256 ) & p0=$!
  ( run_gpu "${fast_tag}" 1 "${runner}" "${task}" "${shots}" 128 \
      symmetric_fast 0.004 trace "${fast_tag}" full 256 ) & p1=$!
  set +e
  wait "${p0}"; rc0=$?
  wait "${p1}"; rc1=$?
  set -e
  [[ ${rc0} -eq 0 && ${rc1} -eq 0 ]]
}

verify
/root/miniconda3/bin/python -m py_compile "${input_auditor}" "${finalizer}"
PYTHONPATH="${canonical_root}" /root/miniconda3/bin/python "${canonical_root}/test_audit_model_input_hashes.py"
PYTHONPATH="${canonical_root}" /root/miniconda3/bin/python "${canonical_root}/test_finalize_fair_three_arm.py"
/root/miniconda3/bin/python "${canonical_root}/test_fair_three_arm_queue_contract.py"
bash -n "${runner}" "${controller}"

mkdir -p "${queue_root}/protocol" "${queue_root}/leakage"
run_stage leakage_static /root/miniconda3/bin/python "${leakage_auditor}" \
  --source-root "${eval_root}" --output "${queue_root}/leakage/static.json"
run_stage model_weight_manifest bash -c \
  'sha256sum "$1"/*.safetensors | sort -k2 >"$2"' _ \
  "${model_path}" "${queue_root}/protocol/model_weights.sha256"
{
  echo "captured=$(date --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "canonical_commit=$(git -C /root/autodl-tmp/dlm-seq-flow rev-parse HEAD)"
  echo "eval_root=${eval_root}"
  echo "model_path=${model_path}"
  /root/miniconda3/bin/python - <<'PY'
import importlib.metadata as m
for name in ("torch","transformers","lm_eval","accelerate","datasets"):
    try: print(f"{name}={m.version(name)}")
    except m.PackageNotFoundError: print(f"{name}=MISSING")
PY
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
} >"${queue_root}/protocol/environment.txt"

# Current-server re-evaluation only. No old-server generation is a formal arm.
run_task_pair humaneval 0 he_baseline_full164 he_fast_full164
run_task_pair localleap_math500 0 math_baseline_full500 math_fast_full500
run_task_pair gsm8k 0 gsm_baseline_full1319 gsm_fast_full1319
run_task_pair mbpp 0 mbpp_baseline_full500 mbpp_fast_full500

he_base=${run_base}/humaneval/he_baseline_full164
he_fast=${run_base}/humaneval/he_fast_full164
math_base=${run_base}/localleap_math500/math_baseline_full500
math_fast=${run_base}/localleap_math500/math_fast_full500
gsm_base=${run_base}/gsm8k/gsm_baseline_full1319
gsm_fast=${run_base}/gsm8k/gsm_fast_full1319
mbpp_base=${run_base}/mbpp/mbpp_baseline_full500
mbpp_fast=${run_base}/mbpp/mbpp_fast_full500

mkdir -p "${queue_root}/mbpp_assertion"
run_stage mbpp_assertion_baseline /root/miniconda3/bin/python "${mbpp_auditor}" \
  --samples "$(sample_file mbpp "${mbpp_base}")" --task-records "$(records "${mbpp_base}")" \
  --output-dir "${queue_root}/mbpp_assertion/baseline"
run_stage mbpp_assertion_fast /root/miniconda3/bin/python "${mbpp_auditor}" \
  --samples "$(sample_file mbpp "${mbpp_fast}")" --task-records "$(records "${mbpp_fast}")" \
  --output-dir "${queue_root}/mbpp_assertion/fast"

for item in \
  "humaneval:${accuracy_he}:${he_fast}" \
  "math500:${accuracy_math}:${math_fast}" \
  "gsm8k:${accuracy_gsm}:${gsm_fast}" \
  "mbpp:${accuracy_mbpp}:${mbpp_fast}"
do
  name=${item%%:*}; remainder=${item#*:}; accuracy_run=${remainder%%:*}; fast_run=${remainder#*:}
  run_stage "leakage_${name}_accuracy" /root/miniconda3/bin/python "${leakage_auditor}" \
    --source-root "${eval_root}" --run-root "${accuracy_run}" \
    --expected-profile "${accuracy_profile}" --output "${queue_root}/leakage/${name}_accuracy.json"
  run_stage "leakage_${name}_fast" /root/miniconda3/bin/python "${leakage_auditor}" \
    --source-root "${eval_root}" --run-root "${fast_run}" \
    --expected-profile symmetric_fast --output "${queue_root}/leakage/${name}_fast.json"
done

mkdir -p "${queue_root}/paired"
compare_task() {
  local task=$1 baseline_root=$2 accuracy_root=$3 fast_root=$4 baseline_records=$5 accuracy_records=$6 fast_records=$7
  run_stage "pair_${task}_accuracy" /root/miniconda3/bin/python "${pairer}" \
    "${baseline_records}" "${accuracy_records}" \
    --baseline-config "${baseline_root}/run_config.txt" \
    --method-config "${accuracy_root}/run_config.txt" \
    --output-dir "${queue_root}/paired/${task}_accuracy"
  run_stage "pair_${task}_fast" /root/miniconda3/bin/python "${pairer}" \
    "${baseline_records}" "${fast_records}" \
    --baseline-config "${baseline_root}/run_config.txt" \
    --method-config "${fast_root}/run_config.txt" \
    --output-dir "${queue_root}/paired/${task}_fast"
}
compare_task humaneval "${he_base}" "${accuracy_he}" "${he_fast}" \
  "$(records "${he_base}")" "$(records "${accuracy_he}")" "$(records "${he_fast}")"
compare_task math500 "${math_base}" "${accuracy_math}" "${math_fast}" \
  "$(records "${math_base}")" "$(records "${accuracy_math}")" "$(records "${math_fast}")"
compare_task gsm8k "${gsm_base}" "${accuracy_gsm}" "${gsm_fast}" \
  "$(records "${gsm_base}")" "$(records "${accuracy_gsm}")" "$(records "${gsm_fast}")"
compare_task mbpp "${mbpp_base}" "${accuracy_mbpp}" "${mbpp_fast}" \
  "${queue_root}/mbpp_assertion/baseline/audit_records.jsonl" "${accuracy_mbpp_records}" \
  "${queue_root}/mbpp_assertion/fast/audit_records.jsonl"

audit_inputs() {
  local label=$1 task=$2 count=$3 run_root=$4
  run_stage "input_${label}" /root/miniconda3/bin/python "${input_auditor}" audit \
    --samples "$(sample_file "${task}" "${run_root}")" --model-path "${model_path}" \
    --model-weight-manifest "${queue_root}/protocol/model_weights.sha256" \
    --expected-records "${count}" --output-dir "${queue_root}/model_inputs/${label}"
}
compare_inputs() {
  local task=$1 arm=$2
  run_stage "input_compare_${task}_${arm}" /root/miniconda3/bin/python "${input_auditor}" compare \
    --reference "${queue_root}/model_inputs/${task}_baseline/model_input_records.jsonl" \
    --candidate "${queue_root}/model_inputs/${task}_${arm}/model_input_records.jsonl" \
    --output "${queue_root}/model_inputs/${task}_${arm}_vs_baseline.json"
}
audit_inputs humaneval_baseline humaneval 164 "${he_base}"
audit_inputs humaneval_accuracy humaneval 164 "${accuracy_he}"
audit_inputs humaneval_fast humaneval 164 "${he_fast}"
audit_inputs math500_baseline localleap_math500 500 "${math_base}"
audit_inputs math500_accuracy localleap_math500 500 "${accuracy_math}"
audit_inputs math500_fast localleap_math500 500 "${math_fast}"
audit_inputs gsm8k_baseline gsm8k 1319 "${gsm_base}"
audit_inputs gsm8k_accuracy gsm8k 1319 "${accuracy_gsm}"
audit_inputs gsm8k_fast gsm8k 1319 "${gsm_fast}"
audit_inputs mbpp_baseline mbpp 500 "${mbpp_base}"
audit_inputs mbpp_accuracy mbpp 500 "${accuracy_mbpp}"
audit_inputs mbpp_fast mbpp 500 "${mbpp_fast}"
for task in humaneval math500 gsm8k mbpp; do
  compare_inputs "${task}" accuracy
  compare_inputs "${task}" fast
done

/root/miniconda3/bin/python - "${queue_root}" "${accuracy_profile}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
tasks={}
for task in ("humaneval","math500","gsm8k","mbpp"):
    tasks[task]={
        "accuracy":str(root/"paired"/(task+"_accuracy")/"paired_summary.json"),
        "fast":str(root/"paired"/(task+"_fast")/"paired_summary.json"),
        "accuracy_input_compare":str(root/"model_inputs"/(task+"_accuracy_vs_baseline.json")),
        "fast_input_compare":str(root/"model_inputs"/(task+"_fast_vs_baseline.json")),
    }
spec={
    "schema":"fair_three_arm_finalizer_spec_v1",
    "accuracy_profile":sys.argv[2],
    "fast_profile":"symmetric_fast",
    "fallback_profile":"trajectory_confirmed_public_guard_v11_family",
    "tasks":tasks,
}
(root/"finalizer_spec.json").write_text(json.dumps(spec,indent=2)+"\n")
PY
run_stage finalize /root/miniconda3/bin/python "${finalizer}" \
  "${queue_root}/finalizer_spec.json" --output "${queue_root}/fair_three_arm_summary.json"
run_stage model_weight_reverify sha256sum -c "${queue_root}/protocol/model_weights.sha256"

verify
touch "${queue_root}/DONE"
if [[ "${control_root}" != "${queue_root}" ]]; then touch "${control_root}/DONE"; fi
echo "fair_three_arm_reproduction_complete queue_root=${queue_root}"
