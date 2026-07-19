#!/usr/bin/env bash
set -Eeuo pipefail

source_root=${SOURCE_ROOT:-/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability_dev_outcome_arbiter}
llada_root=${LLADA_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_outcome_arbiter}
parent_root=/root/autodl-tmp/LocalLeap/llada/results/best_symmetric_benchmarks/confirmed_bidirectional_rapid_20260719_v1
parent_queue=/root/autodl-tmp/LocalLeap/llada/results/experiment_queues/confirmed_bidirectional_rapid_20260719_v1
wait_queue=/root/autodl-tmp/LocalLeap/llada_slot_lazy_public_guard/results/experiment_queues/lazy_public_guard_exact_20260719_v1
queue_id=${ATTENTION_QUEUE_ID:-outcome_arbiter_prefill_20260719_v1}
queue_root=${llada_root}/results/experiment_queues/${queue_id}
run_base=${llada_root}/results/best_symmetric_benchmarks/${queue_id}
runner=${llada_root}/run_best_symmetric_benchmark.sh
manifest=${queue_root}/manifest.tsv
frozen=${queue_root}/frozen_sources.sha256

mkdir -p "${queue_root}"
printf '%s\n' "$$" >"${queue_root}/controller.pid"
exec > >(tee -a "${queue_root}/controller.log") 2>&1
trap 'rc=$?; echo "controller_error rc=${rc} line=${LINENO}"; touch "${queue_root}/FAILED"; exit "${rc}"' ERR

while [[ ! -e "${wait_queue}/DONE" ]]; do
  [[ ! -e "${wait_queue}/FAILED" ]] || exit 21
  sleep 10
done

export PATH=/root/miniconda3/bin:${PATH}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_ALLOW_CODE_EVAL=1
cd "${llada_root}"
[[ -s "${manifest}" ]] || printf 'stage\tstatus\tstart\tfinish\texit_code\n' >"${manifest}"

if [[ ! -s "${frozen}" ]]; then
  sha256sum generate.py eval_llada.py differential_selector.py outcome_arbiter.py \
    run_best_symmetric_benchmark.sh compare_paired_task_runs.py slice_audit_by_index.py \
    "${source_root}/outcome_arbiter_preregistration_20260719.json" >"${frozen}"
fi
sha256sum -c "${frozen}"
/root/miniconda3/bin/python -m py_compile generate.py eval_llada.py differential_selector.py outcome_arbiter.py
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/test_outcome_arbiter.py"

records() {
  if [[ -s "$1/audit/audit_records.jsonl" ]]; then printf '%s\n' "$1/audit/audit_records.jsonl";
  else printf '%s\n' "$1/audit/task_audit_records.jsonl"; fi
}
run_one() {
  local label=$1 task=$2 run_name=$3 parent_run=$4
  local start finish rc method summary parent_correct method_correct parent_only method_only
  start=$(date --iso-8601=seconds)
  printf '%s\tSTARTED\t%s\t\t\n' "${label}" "${start}" >>"${manifest}"
  set +e
  timeout --kill-after=5m 12h env CUDA_VISIBLE_DEVICES=1 LLADA_ROOT="${llada_root}" \
    ATTENTION_QUEUE_ID="${queue_id}" "${runner}" "${task}" 0 128 \
    trajectory_outcome_arbiter 0.004 trace "${run_name}" 16 256
  rc=$?
  set -e
  finish=$(date --iso-8601=seconds)
  printf '%s\t%s\t%s\t%s\t%s\n' "${label}" "$([[ ${rc} -eq 0 ]] && echo DONE || echo FAILED)" "${start}" "${finish}" "${rc}" >>"${manifest}"
  [[ ${rc} -eq 0 ]]
  if [[ "${task}" == localleap_math500 ]]; then method=${run_base}/localleap_math500/${run_name}; else method=${run_base}/gsm8k/${run_name}; fi
  mkdir -p "${queue_root}/slices" "${queue_root}/paired"
  /root/miniconda3/bin/python slice_audit_by_index.py "$(records "${parent_run}")" \
    "${queue_root}/slices/${label}_parent.jsonl" --start 0 --end 16
  /root/miniconda3/bin/python compare_paired_task_runs.py \
    "${queue_root}/slices/${label}_parent.jsonl" "$(records "${method}")" \
    --baseline-config "${parent_queue}/frozen_sources.sha256" --method-config "${method}/run_config.txt" \
    --method-log "${queue_root}/controller.log" --allow-source-drift \
    --output-dir "${queue_root}/paired/${label}"
  summary=${queue_root}/paired/${label}/paired_summary.json
  read -r parent_correct method_correct parent_only method_only < <(/root/miniconda3/bin/python - "${summary}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
print(x['baseline_correct'],x['method_correct'],x['baseline_only'],x['method_only'])
PY
)
  echo "${label} parent=${parent_correct} method=${method_correct} method_only=${method_only} parent_only=${parent_only}"
  if (( method_correct > parent_correct && method_only > parent_only )); then touch "${queue_root}/DEV_PASS_${label}"; else touch "${queue_root}/DEV_REJECT_${label}"; fi
}

run_one gsm_dev gsm8k gsm_dev_n16 "${parent_root}/gsm8k/gsm_cbv_n64"
run_one math_dev localleap_math500 math_dev_n16 "${parent_root}/localleap_math500/math_cbv_n50"
touch "${queue_root}/DONE"
