#!/usr/bin/env bash
set -uo pipefail

queue_id=${ATTENTION_QUEUE_ID:-response_refine_he_math_20260718_v1}
llada_root=/root/autodl-tmp/LocalLeap/llada
source_root=/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability
runner=${source_root}/scripts/run_best_symmetric_benchmark.sh
queue_root=${llada_root}/results/experiment_queues/${queue_id}
run_base=${llada_root}/results/best_symmetric_benchmarks/${queue_id}
manifest=${queue_root}/formal_manifest.tsv
frozen_hashes=${queue_root}/FROZEN_SOURCE_SHA256
gpu0=${ATTENTION_GPU0:-0}
gpu1=${ATTENTION_GPU1:-1}
mkdir -p "${queue_root}"
exec > >(tee -a "${queue_root}/formal_controller.log") 2>&1

append_manifest() {
  local line=$1
  { flock 9; printf '%s\n' "${line}" >&9; } 9>>"${manifest}"
}

already_done() {
  local label=$1
  [[ -s "${manifest}" ]] && awk -F '\t' -v label="${label}" \
    '$1 == label && $2 == "DONE" {found=1} END {exit !found}' "${manifest}"
}

check_disk() {
  local free_kb
  free_kb=$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')
  if (( free_kb < 8 * 1024 * 1024 )); then
    touch "${queue_root}/FAILED_DISK"
    echo "[HARD STOP] disk below 8 GiB"
    exit 20
  fi
}

verify_source() {
  sha256sum -c "${frozen_hashes}" >/dev/null || {
    touch "${queue_root}/FAILED_SOURCE_DRIFT"
    echo "[HARD STOP] frozen source drift"
    exit 21
  }
}

run_stage() {
  local label=$1
  shift
  if already_done "${label}"; then
    echo "[SKIP-DONE] ${label}"
    return 0
  fi
  check_disk
  verify_source
  local start finish rc
  start=$(date --iso-8601=seconds)
  append_manifest "$(printf '%s\tSTARTED\t%s\t\t' "${label}" "${start}")"
  echo "[START] ${label} ${start}"
  set +e
  timeout --kill-after=5m 24h "$@"
  rc=$?
  set -e
  finish=$(date --iso-8601=seconds)
  if [[ ${rc} -eq 0 ]]; then
    append_manifest "$(printf '%s\tDONE\t%s\t%s\t0' "${label}" "${start}" "${finish}")"
    echo "[DONE] ${label} ${finish}"
  else
    append_manifest "$(printf '%s\tFAILED\t%s\t%s\t%s' "${label}" "${start}" "${finish}" "${rc}")"
    echo "[FAILED] ${label} rc=${rc} ${finish}"
  fi
  return 0
}

run_gpu() {
  local label=$1 gpu=$2
  shift 2
  run_stage "${label}" env CUDA_VISIBLE_DEVICES="${gpu}" "$@"
}

run_pair_same_task() {
  local task=$1 shots=$2 count=$3 profile0=$4 tag0=$5 profile1=$6 tag1=$7
  run_gpu "${tag0}" "${gpu0}" "${runner}" "${task}" "${shots}" 128 \
    "${profile0}" 0.004 trace "${tag0}" "${count}" 256 &
  local pid0=$!
  run_gpu "${tag1}" "${gpu1}" "${runner}" "${task}" "${shots}" 128 \
    "${profile1}" 0.004 trace "${tag1}" "${count}" 256 &
  local pid1=$!
  wait "${pid0}" || true
  wait "${pid1}" || true
}

audit_records() {
  local root=$1
  if [[ -s "${root}/audit/audit_records.jsonl" ]]; then
    echo "${root}/audit/audit_records.jsonl"
  elif [[ -s "${root}/audit/task_audit_records.jsonl" ]]; then
    echo "${root}/audit/task_audit_records.jsonl"
  else
    return 1
  fi
}

pair_runs() {
  local label=$1 task=$2 baseline_root=$3 method_root=$4
  local output=${queue_root}/paired/${label}
  if [[ ! -e "${baseline_root}/DONE" || ! -e "${method_root}/DONE" ]]; then
    append_manifest "$(printf '%s\tSKIPPED\t%s\t%s\tparent_incomplete' "${label}" "$(date --iso-8601=seconds)" "$(date --iso-8601=seconds)")"
    return 0
  fi
  local baseline_records method_records
  baseline_records=$(audit_records "${baseline_root}") || return 0
  method_records=$(audit_records "${method_root}") || return 0
  run_stage "${label}" /root/miniconda3/bin/python \
    "${llada_root}/compare_paired_task_runs.py" \
    "${baseline_records}" "${method_records}" \
    --baseline-config "${baseline_root}/run_config.txt" \
    --method-config "${method_root}/run_config.txt" \
    --method-log "${queue_root}/$(basename "${method_root}").log" \
    --output-dir "${output}"
}

healthy_at_least() {
  local label=$1 tolerance=$2
  /root/miniconda3/bin/python - "${queue_root}/paired/${label}/paired_summary.json" "${tolerance}" <<'PY'
import json, sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
tolerance = int(sys.argv[2])
healthy = (
    x.get("prompt_hash_mismatches") == 0
    and x.get("target_hash_mismatches") == 0
    and x.get("duplicate_or_missing_ids") == 0
)
raise SystemExit(0 if healthy and x["method_correct"] + tolerance >= x["baseline_correct"] else 1)
PY
}

summarize_trace() {
  local task=$1 tag=$2
  local root=${run_base}/${task}/${tag}
  if [[ -s "${root}/trace/rank_0.jsonl" ]]; then
    /root/miniconda3/bin/python "${llada_root}/summarize_response_refine.py" \
      "${root}/trace/rank_0.jsonl" \
      --output "${root}/audit/response_refine_summary.json" >/dev/null
  fi
}

export PATH=/root/miniconda3/bin:${PATH}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
cd "${llada_root}"

if [[ ! -s "${manifest}" ]]; then
  printf 'stage\tstatus\tstart\tfinish\texit_code_or_note\n' >"${manifest}"
fi

if [[ ! -s "${frozen_hashes}" ]]; then
  sha256sum \
    "${llada_root}/generate.py" \
    "${llada_root}/eval_llada.py" \
    "${llada_root}/differential_selector.py" \
    "${llada_root}/model/modeling_llada.py" \
    "${llada_root}/audit_attention_stability.py" \
    "${llada_root}/audit_lm_eval_task.py" \
    "${llada_root}/compare_paired_task_runs.py" \
    "${llada_root}/summarize_response_refine.py" \
    "${runner}" \
    "${source_root}/scripts/run_response_refine_he_math_queue.sh" \
    >"${frozen_hashes}"
fi

verify_source
/root/miniconda3/bin/python -m py_compile generate.py eval_llada.py summarize_response_refine.py
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_response_refine.py"
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_attention_stability.py"
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_response_credit_exchange.py"
bash -n "${runner}"

echo "queue_id=${queue_id}"
echo "source_commit=$(git -C /root/autodl-tmp/dlm-seq-flow rev-parse HEAD)"
echo "formal_baseline=original_llada_low_confidence"
echo "development_parent=symmetric_fast_tau_0.004"
echo "matched_method=64_nfe_full_draft_plus_64_nfe_directed_response_refine"
echo "extra_method=128_nfe_parent_draft_plus_64_nfe_directed_response_refine"
echo "horizontal=directed_attention_incidence_frontier_and_source_first_repair"
echo "vertical=conditioned_invalidation_frontier_and_revision_margin_repair"

# The already completed 2-example smoke runs are health checks only.
if [[ ! -e "${run_base}/humaneval/smoke_he_refine_fast/DONE" \
   || ! -e "${run_base}/localleap_math500/smoke_math_refine_fast/DONE" ]]; then
  touch "${queue_root}/FAILED_SMOKE"
  exit 22
fi
summarize_trace humaneval smoke_he_refine_fast
summarize_trace localleap_math500 smoke_math_refine_fast

# Fresh sampled controls prevent accidental reliance on list-position pairing.
run_pair_same_task humaneval 0 32 baseline he_baseline_128_n32 \
  symmetric_fast he_parent_128_n32
run_pair_same_task humaneval 0 32 response_refine_fast he_refine_fast_128_n32 \
  response_refine_extra he_refine_extra_128_n32

run_pair_same_task localleap_math500 0 50 baseline math_baseline_128_n50 \
  symmetric_fast math_parent_128_n50
run_pair_same_task localleap_math500 0 50 response_refine_fast math_refine_fast_128_n50 \
  response_refine_extra math_refine_extra_128_n50

for profile in fast extra; do
  pair_runs "he_${profile}_vs_base_n32" humaneval \
    "${run_base}/humaneval/he_baseline_128_n32" \
    "${run_base}/humaneval/he_refine_${profile}_128_n32"
  pair_runs "he_${profile}_vs_parent_n32" humaneval \
    "${run_base}/humaneval/he_parent_128_n32" \
    "${run_base}/humaneval/he_refine_${profile}_128_n32"
  pair_runs "math_${profile}_vs_base_n50" localleap_math500 \
    "${run_base}/localleap_math500/math_baseline_128_n50" \
    "${run_base}/localleap_math500/math_refine_${profile}_128_n50"
  pair_runs "math_${profile}_vs_parent_n50" localleap_math500 \
    "${run_base}/localleap_math500/math_parent_128_n50" \
    "${run_base}/localleap_math500/math_refine_${profile}_128_n50"
  summarize_trace humaneval "he_refine_${profile}_128_n32"
  summarize_trace localleap_math500 "math_refine_${profile}_128_n50"
done

selected=none
fast_ok=0
extra_ok=0
if healthy_at_least he_fast_vs_parent_n32 0 \
   && healthy_at_least math_fast_vs_base_n50 1; then fast_ok=1; fi
if healthy_at_least he_extra_vs_parent_n32 0 \
   && healthy_at_least math_extra_vs_base_n50 1; then extra_ok=1; fi
if (( fast_ok )); then selected=response_refine_fast; fi
if (( extra_ok && ! fast_ok )); then selected=response_refine_extra; fi
if (( fast_ok && extra_ok )); then
  selected=$(/root/miniconda3/bin/python - "${queue_root}" <<'PY'
import json, sys
r = sys.argv[1]
def score(profile):
    labels = [f"he_{profile}_vs_parent_n32", f"math_{profile}_vs_base_n50"]
    return sum(json.load(open(f"{r}/paired/{x}/paired_summary.json"))["method_correct"] for x in labels)
print("response_refine_extra" if score("extra") >= score("fast") + 2 else "response_refine_fast")
PY
  )
fi
echo "${selected}" >"${queue_root}/selected_profile.txt"
echo "selected_profile=${selected}"

if [[ "${selected}" == none ]]; then
  touch "${queue_root}/DONE_NO_PROMOTION"
  touch "${queue_root}/DONE"
  exit 0
fi

# Full confirmation runs use one evaluator per GPU and are paired by stable id
# against the immutable formal baseline and parent records from earlier runs.
run_gpu he_refine_selected_128_full "${gpu0}" "${runner}" humaneval 0 128 \
  "${selected}" 0.004 trace he_refine_selected_128_full full 256 &
pid0=$!
run_gpu math_refine_selected_128_full "${gpu1}" "${runner}" localleap_math500 0 128 \
  "${selected}" 0.004 trace math_refine_selected_128_full full 256 &
pid1=$!
wait "${pid0}" || true
wait "${pid1}" || true

he_baseline=/root/autodl-tmp/LocalLeap/llada/results/attention_recovery/humaneval/recovery_he_baseline_128
he_parent=/root/autodl-tmp/LocalLeap/llada/results/attention_recovery/humaneval/recovery_he_symmetric_fast_128
math_baseline=/root/autodl-tmp/LocalLeap/llada/results/best_symmetric_benchmarks/best_symmetric_long_20260716_v2/localleap_math500/math500_baseline_128
math_parent=/root/autodl-tmp/LocalLeap/llada/results/best_symmetric_benchmarks/best_symmetric_long_20260716_v2/localleap_math500/math500_symmetric_fast_128

pair_runs he_selected_vs_baseline_full humaneval "${he_baseline}" \
  "${run_base}/humaneval/he_refine_selected_128_full"
pair_runs he_selected_vs_parent_full humaneval "${he_parent}" \
  "${run_base}/humaneval/he_refine_selected_128_full"
pair_runs math_selected_vs_baseline_full localleap_math500 "${math_baseline}" \
  "${run_base}/localleap_math500/math_refine_selected_128_full"
pair_runs math_selected_vs_parent_full localleap_math500 "${math_parent}" \
  "${run_base}/localleap_math500/math_refine_selected_128_full"
summarize_trace humaneval he_refine_selected_128_full
summarize_trace localleap_math500 math_refine_selected_128_full

if [[ -e "${run_base}/humaneval/he_refine_selected_128_full/DONE" \
   && -e "${run_base}/localleap_math500/math_refine_selected_128_full/DONE" ]]; then
  touch "${queue_root}/DONE"
else
  touch "${queue_root}/FAILED_FULL"
fi
