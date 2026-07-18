#!/usr/bin/env bash
set -euo pipefail

queue_id=${ATTENTION_QUEUE_ID:-trajectory_lazy_consensus_speed_20260719_v1}
llada_root=${LLADA_ROOT:-/root/autodl-tmp/LocalLeap/llada_slot_lazy_consensus}
source_root=${ATTENTION_SOURCE_ROOT:-/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability_dev_lazy_consensus}
runner=${source_root}/scripts/run_best_symmetric_benchmark.sh
controller=${source_root}/scripts/run_lazy_consensus_speed_queue.sh
comparator=${source_root}/compare_lazy_consensus.py
queue_root=${llada_root}/results/experiment_queues/${queue_id}
run_root=${llada_root}/results/best_symmetric_benchmarks/${queue_id}/humaneval/he_lazy_consensus_n16
formal_root=/root/autodl-tmp/LocalLeap/llada/results/best_symmetric_benchmarks/trajectory_consensus_formal_20260719_v3/humaneval
formal_queue=/root/autodl-tmp/LocalLeap/llada/results/experiment_queues/trajectory_consensus_formal_20260719_v3
full_root=/root/autodl-tmp/LocalLeap/llada/results/best_symmetric_benchmarks/trajectory_consensus_fallback_20260719_v1/humaneval/he_consensus_n16
frozen=${queue_root}/FROZEN_SOURCE_SHA256
mkdir -p "${queue_root}"
exec > >(tee -a "${queue_root}/controller.log") 2>&1

verify() {
  (cd "${llada_root}" && sha256sum -c "${frozen}" >/dev/null) || {
    touch "${queue_root}/FAILED_SOURCE_DRIFT"
    exit 21
  }
}

export PATH=/root/miniconda3/bin:${PATH}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=/root/autodl-tmp/.cache/huggingface HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_ALLOW_CODE_EVAL=1
cd "${llada_root}"
if [[ ! -s "${frozen}" ]]; then
  sha256sum generate.py eval_llada.py compare_paired_task_runs.py audit_attention_stability.py \
    audit_lm_eval_task.py postprocess_code.py humaneval_execution.py sanitize.py \
    "${runner}" "${controller}" "${comparator}" "${source_root}/tests/test_attention_stability.py" >"${frozen}"
fi
verify
/root/miniconda3/bin/python -m py_compile generate.py eval_llada.py "${comparator}"
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_attention_stability.py"
bash -n "${runner}" "${controller}"

# GPU1 is owned by the formal queue until both same-source comparators finish.
# The formal controller cannot advance to MATH while its slower GPU0 consensus
# job is unfinished, so this window is contention-free.
while [[ ! -e "${formal_root}/he_base_n96/DONE" ]]; do
  formal_pid=$(cat "${formal_queue}/controller.pid" 2>/dev/null || true)
  if [[ -z "${formal_pid}" ]] || ! kill -0 "${formal_pid}" 2>/dev/null; then
    touch "${queue_root}/FAILED_FORMAL_QUEUE_ENDED"
    exit 22
  fi
  sleep 10
done
if [[ -e "${formal_root}/he_consensus_n96/DONE" ]]; then
  touch "${queue_root}/DONE_DEFERRED_NO_SAFE_WINDOW" "${queue_root}/DONE"
  exit 0
fi

CUDA_VISIBLE_DEVICES=1 ATTENTION_QUEUE_ID="${queue_id}" LLADA_ROOT="${llada_root}" \
  "${runner}" humaneval 0 128 trajectory_lazy_consensus_block 0.004 trace he_lazy_consensus_n16 16 256
[[ -e "${run_root}/DONE" ]] || { touch "${queue_root}/FAILED_RUN"; exit 23; }
verify
/root/miniconda3/bin/python "${comparator}" \
  --full-audit "${full_root}/audit/audit_records.jsonl" \
  --lazy-audit "${run_root}/audit/audit_records.jsonl" \
  --full-trace "${full_root}/trace/rank_0.jsonl" \
  --lazy-trace "${run_root}/trace/rank_0.jsonl" \
  --output "${queue_root}/lazy_equivalence_summary.json"
touch "${queue_root}/DONE_EQUIVALENT" "${queue_root}/DONE"
