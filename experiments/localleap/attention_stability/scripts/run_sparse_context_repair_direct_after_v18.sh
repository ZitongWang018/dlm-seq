#!/usr/bin/env bash
set -Eeuo pipefail

llada_root=${LLADA_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_sparse_context_repair_v19_direct_v5}
parent_root=${PARENT_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_admissible_lazy_guard}
v18_root=${V18_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_early_localized_conflict_repair_v4}
parent_queue_id=${PARENT_QUEUE_ID:-best_framework_full4_20260719_v1}
v18_queue_id=${V18_QUEUE_ID:-early_localized_evidence_conflict_repair_20260720_v4}
queue_id=${ATTENTION_QUEUE_ID:-sparse_context_repair_direct_20260720_v5}
queue_root=${llada_root}/results/experiment_queues/${queue_id}
v18_queue=${v18_root}/results/experiment_queues/${v18_queue_id}
bootstrap=${llada_root}/scripts/run_sparse_context_repair_direct_after_v18.sh
generic=${llada_root}/scripts/run_localized_evidence_conflict_repair_queue.sh
preregistration=sparse_context_repair_preregistration_20260720_v2.json

mkdir -p "${queue_root}"
printf '%s\n' "$$" >"${queue_root}/controller.pid"
exec > >(tee -a "${queue_root}/formal_controller.log") 2>&1
trap 'rc=$?; echo "bootstrap_error rc=${rc} line=${LINENO}"; touch "${queue_root}/FAILED"; exit "${rc}"' ERR

export PATH=/root/miniconda3/bin:${PATH}
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1

bootstrap_hashes=${queue_root}/bootstrap_sources.sha256
if [[ ! -s "${bootstrap_hashes}" ]]; then
  sha256sum "${bootstrap}" "${generic}" \
    "${llada_root}/${preregistration}" >"${bootstrap_hashes}"
fi
verify_bootstrap() { sha256sum -c "${bootstrap_hashes}" >/dev/null; }

wait_terminal() {
  local root=$1 deadline=$(( $(date +%s) + 7 * 24 * 60 * 60 ))
  while [[ ! -e "${root}/DONE" && ! -e "${root}/FAILED" ]]; do
    verify_bootstrap
    (( $(date +%s) < deadline )) || return 21
    sleep 20
  done
}

verify_bootstrap
bash -n "${bootstrap}" "${generic}"
/root/miniconda3/bin/python -m json.tool \
  "${llada_root}/${preregistration}" >/dev/null

echo "waiting_for_v18_queue=${v18_queue}"
wait_terminal "${v18_queue}"
[[ -e "${v18_queue}/DONE" && ! -e "${v18_queue}/FAILED" ]] || {
  touch "${queue_root}/BLOCKED_V18_INFRASTRUCTURE_FAILURE" \
    "${queue_root}/FAILED"; exit 22; }

if [[ -e "${v18_queue}/ACCEPTED" ]]; then
  printf 'reason=v18_already_passed_unified_gate\nfinished=%s\n' \
    "$(date --iso-8601=seconds)" >"${queue_root}/SKIPPED_V18_ACCEPTED"
  touch "${queue_root}/DONE"
  exit 0
fi
[[ -e "${v18_queue}/REJECTED" ]] || {
  touch "${queue_root}/BLOCKED_V18_TERMINAL_WITHOUT_DECISION" \
    "${queue_root}/FAILED"; exit 23; }

verify_bootstrap
echo "launching_sparse_context_repair_direct_after_v18_rejection"
exec env LLADA_ROOT="${llada_root}" PARENT_ROOT="${parent_root}" \
  PARENT_QUEUE_ID="${parent_queue_id}" ATTENTION_QUEUE_ID="${queue_id}" \
  PROFILE=trajectory_early_sparse_context_repair RUN_PREFIX=v19 \
  PREREGISTRATION="${preregistration}" bash "${generic}"
