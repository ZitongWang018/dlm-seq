#!/bin/bash
set -euo pipefail

export PATH=/root/miniconda3/bin:$PATH
python_bin=/root/miniconda3/bin/python

llada_root=/root/autodl-tmp/LocalLeap/llada
attention_runner=/root/autodl-tmp/LocalLeap/scripts/llada/run_attention_stability_humaneval.sh
candidate_runner=/root/autodl-tmp/LocalLeap/scripts/llada/run_candidate_memory_humaneval.sh
current_sweep=${llada_root}/results/attention_stability/sweeps/attention_tau_sweep_20260713_v1
queue_id=${QUEUE_ID:-cross_step_full_queue_20260714_v1}
queue_root=${llada_root}/results/experiment_queues/${queue_id}
mkdir -p "${queue_root}"
manifest=${queue_root}/manifest.tsv
lock=${llada_root}/results/experiment_queues/cross_step_full_queue.lock

exec 9>"${lock}"
if ! flock -n 9; then
  echo "ERROR: another cross-step queue owns ${lock}" >&2
  exit 10
fi

if [[ ! -e "${manifest}" ]]; then
  printf 'stage\tvalue\tstatus\tstart\tfinish\trun_tag\texit_code\n' > "${manifest}"
fi

echo "queue_id=${queue_id}"
echo "start=$(date --iso-8601=seconds)"
echo "waiting_for=${current_sweep}/DONE"
while [[ ! -e "${current_sweep}/DONE" ]]; do
  if ! kill -0 133016 2>/dev/null; then
    echo "ERROR: prerequisite sweep stopped without DONE" >&2
    exit 11
  fi
  echo "[QUEUE] prerequisite still active at $(date --iso-8601=seconds)"
  sleep 60
done

validate_attention_run() {
  local tau=$1
  local tag=$2
  local root=${llada_root}/results/attention_stability/tau${tau}/${tag}
  "${python_bin}" - "${root}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
assert (root / "DONE").exists(), root
audit = json.loads((root / "audit/audit_summary.json").read_text())
steps = json.loads((root / "audit/step_diagnostics_summary.json").read_text())
assert audit["total"] == 164 and audit["nfe_min"] == audit["nfe_max"] == 256
assert audit["residual_mask_count"] == 0
assert steps["files"] == 164 and steps["total_steps"] == 41984
assert steps["residual_mask_count"] == 0
assert sum(1 for _ in (root / "trace/rank_0.jsonl").open()) == 164
PY
}

validate_candidate_run() {
  local mode=$1
  local topk=$2
  local delta=$3
  local tag=$4
  local delta_tag=${delta//./p}
  local root=${llada_root}/results/candidate_memory/${mode}/k${topk}_delta${delta_tag}/${tag}
  "${python_bin}" - "${root}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
assert (root / "DONE").exists(), root
audit = json.loads((root / "audit/audit_summary.json").read_text())
steps = json.loads((root / "audit/step_diagnostics_summary.json").read_text())
assert audit["total"] == 164 and audit["nfe_min"] == audit["nfe_max"] == 256
assert audit["residual_mask_count"] == 0
assert steps["files"] == 164 and steps["total_steps"] == 41984
assert steps["residual_mask_count"] == 0
assert sum(1 for _ in (root / "trace/rank_0.jsonl").open()) == 164
PY
}

run_stage() {
  local stage=$1
  local value=$2
  local tag=$3
  shift 3
  local start_time finish_time exit_code
  start_time=$(date --iso-8601=seconds)
  printf '%s\t%s\tSTARTED\t%s\t\t%s\t\n' "${stage}" "${value}" "${start_time}" "${tag}" >> "${manifest}"
  echo "[QUEUE] starting ${stage}=${value} tag=${tag} at ${start_time}"
  set +e
  "$@" > >(tee "${queue_root}/${tag}.log") 2>&1
  exit_code=$?
  set -e
  finish_time=$(date --iso-8601=seconds)
  if [[ "${exit_code}" -eq 0 ]]; then
    printf '%s\t%s\tCOMPLETED\t%s\t%s\t%s\t0\n' "${stage}" "${value}" "${start_time}" "${finish_time}" "${tag}" >> "${manifest}"
    echo "[QUEUE] completed ${stage}=${value} at ${finish_time}"
  else
    printf '%s\t%s\tFAILED\t%s\t%s\t%s\t%s\n' "${stage}" "${value}" "${start_time}" "${finish_time}" "${tag}" "${exit_code}" >> "${manifest}"
    echo "[QUEUE] failed ${stage}=${value} exit=${exit_code} at ${finish_time}" >&2
    exit "${exit_code}"
  fi
}

for tau in 0.004 0.0025 0.001 0.0005; do
  tau_tag=${tau//./p}
  tag=${queue_id}_tau${tau_tag}
  root=${llada_root}/results/attention_stability/tau${tau}/${tag}
  if [[ -e "${root}/DONE" ]] && validate_attention_run "${tau}" "${tag}"; then
    printf 'attention_tau\t%s\tSKIPPED_RESUME\t%s\t%s\t%s\t0\n' "${tau}" "$(date --iso-8601=seconds)" "$(date --iso-8601=seconds)" "${tag}" >> "${manifest}"
    continue
  fi
  run_stage attention_tau "${tau}" "${tag}" timeout --kill-after=5m 3h "${attention_runner}" "${tau}" full "${tag}"
  validate_attention_run "${tau}" "${tag}"
done

for mode in confidence frontier; do
  tag=${queue_id}_candidate_${mode}_k8_delta0
  root=${llada_root}/results/candidate_memory/${mode}/k8_delta0/${tag}
  if [[ -e "${root}/DONE" ]] && validate_candidate_run "${mode}" 8 0 "${tag}"; then
    printf 'candidate_memory\t%s\tSKIPPED_RESUME\t%s\t%s\t%s\t0\n' "${mode}" "$(date --iso-8601=seconds)" "$(date --iso-8601=seconds)" "${tag}" >> "${manifest}"
    continue
  fi
  run_stage candidate_memory "${mode}" "${tag}" timeout --kill-after=5m 4h "${candidate_runner}" "${mode}" 8 0 full "${tag}"
  validate_candidate_run "${mode}" 8 0 "${tag}"
done

echo "finish=$(date --iso-8601=seconds)"
touch "${queue_root}/DONE"
