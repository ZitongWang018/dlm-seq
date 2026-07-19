#!/usr/bin/env bash
set -Eeuo pipefail

recovery_root=${RECOVERY_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_sparse_context_repair_v19_recovery_v2}
original_root=${ORIGINAL_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_sparse_context_repair_v19}
parent_root=${PARENT_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_admissible_lazy_guard}
original_id=${ORIGINAL_ID:-sparse_context_repair_rapid_20260720_v1}
recovery_id=${ATTENTION_QUEUE_ID:-sparse_context_repair_rapid_20260720_v2}
parent_id=${PARENT_QUEUE_ID:-best_framework_full4_20260719_v1}
original_queue=${original_root}/results/experiment_queues/${original_id}
queue_root=${recovery_root}/results/experiment_queues/${recovery_id}
controller=${recovery_root}/scripts/run_sparse_context_repair_infrastructure_recovery_v2.sh
generic=${recovery_root}/scripts/run_localized_evidence_conflict_repair_queue.sh
preregistration=sparse_context_repair_preregistration_20260720_v1.json

mkdir -p "${queue_root}"
printf '%s\n' "$$" >"${queue_root}/controller.pid"
exec > >(tee -a "${queue_root}/formal_controller.log") 2>&1
trap 'rc=$?; echo "v19_recovery_error rc=${rc} line=${LINENO}"; touch "${queue_root}/FAILED"; exit "${rc}"' ERR

export PATH=/root/miniconda3/bin:${PATH}
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1

frozen=${queue_root}/recovery_sources.sha256
if [[ ! -s "${frozen}" ]]; then
  ( cd "${recovery_root}" && sha256sum \
      scripts/run_sparse_context_repair_infrastructure_recovery_v2.sh \
      scripts/run_localized_evidence_conflict_repair_queue.sh \
      run_best_symmetric_benchmark.sh generate.py eval_llada.py \
      differential_selector.py postprocess_code.py humaneval_execution.py \
      sanitize.py "${preregistration}" ) >"${frozen}"
fi
verify() { ( cd "${recovery_root}" && sha256sum -c "${frozen}" >/dev/null ); }
wait_terminal() {
  local root=$1 deadline=$(( $(date +%s) + 14 * 24 * 60 * 60 ))
  while [[ ! -e "${root}/DONE" && ! -e "${root}/FAILED" ]]; do
    verify
    (( $(date +%s) < deadline )) || return 21
    sleep 20
  done
}

verify
bash -n "${controller}" "${generic}"
/root/miniconda3/bin/python -m json.tool "${recovery_root}/${preregistration}" >/dev/null
echo "waiting_for_original_v19_terminal=${original_queue}"
wait_terminal "${original_queue}"

if [[ -e "${original_queue}/DONE" && ! -e "${original_queue}/FAILED" ]]; then
  if [[ -e "${original_queue}/ACCEPTED" ]]; then touch "${queue_root}/ACCEPTED"; fi
  if [[ -e "${original_queue}/REJECTED" ]]; then touch "${queue_root}/REJECTED"; fi
  touch "${queue_root}/ORIGINAL_V19_COMPLETED" "${queue_root}/DONE"
  exit 0
fi

# Only recover the known source-packaging failure. Any algorithm, evaluator,
# CUDA, OOM or unknown infrastructure failure remains terminal for inspection.
log=${original_queue}/formal_controller.log
[[ -s "${log}" ]]
if ! grep -Eq '(postprocess_code|humaneval_execution|sanitize)\.py: No such file or directory' "${log}"; then
  touch "${queue_root}/BLOCKED_UNKNOWN_ORIGINAL_FAILURE"
  exit 22
fi
original_pid=$(cat "${original_queue}/controller.pid")
if kill -0 "${original_pid}" 2>/dev/null; then
  touch "${queue_root}/BLOCKED_ORIGINAL_CONTROLLER_STILL_ALIVE"
  exit 23
fi
touch "${queue_root}/RECOVERING_KNOWN_SOURCE_PACKAGING_FAILURE"
verify
exec env LLADA_ROOT="${recovery_root}" PARENT_ROOT="${parent_root}" \
  PARENT_QUEUE_ID="${parent_id}" ATTENTION_QUEUE_ID="${recovery_id}" \
  PROFILE=trajectory_early_sparse_context_repair RUN_PREFIX=v19 \
  PREREGISTRATION="${preregistration}" bash "${generic}"
