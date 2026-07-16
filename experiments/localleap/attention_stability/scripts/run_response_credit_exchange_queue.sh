#!/usr/bin/env bash
set -uo pipefail

queue_id=${ATTENTION_QUEUE_ID:-response_credit_exchange_20260717_v1}
llada_root=/root/autodl-tmp/LocalLeap/llada
runner=/root/autodl-tmp/LocalLeap/scripts/llada/run_best_symmetric_benchmark.sh
queue_root=${llada_root}/results/experiment_queues/${queue_id}
run_base=${llada_root}/results/best_symmetric_benchmarks/${queue_id}
manifest=${queue_root}/manifest.tsv
mkdir -p "${queue_root}"
exec > >(tee -a "${queue_root}/controller.log") 2>&1

check_disk() {
  local free_kb
  free_kb=$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')
  if (( free_kb < 8 * 1024 * 1024 )); then
    echo "[QUEUE HARD STOP] disk below 8 GiB"
    touch "${queue_root}/FAILED_DISK"
    exit 20
  fi
}

verify_source() {
  cd "${llada_root}"
  sha256sum -c "${queue_root}/FROZEN_SOURCE_SHA256" >/dev/null || {
    echo "[QUEUE HARD STOP] source hash drift"
    touch "${queue_root}/FAILED_SOURCE_DRIFT"
    exit 21
  }
}

stage_already_done() {
  local label=$1
  awk -F '\t' -v label="${label}" '$1 == label && $2 == "DONE" {found=1} END {exit !found}' "${manifest}"
}

record_stage_status() {
  local label=$1 status=$2 note=${3:-}
  printf '%s\t%s\t%s\t%s\t%s\n' "${label}" "${status}" "$(date --iso-8601=seconds)" "$(date --iso-8601=seconds)" "${note}" >> "${manifest}"
}

run_stage() {
  local label=$1
  shift
  if stage_already_done "${label}"; then
    echo "[SKIP-DONE] ${label}"
    return 0
  fi
  check_disk
  verify_source
  local started finished rc
  started=$(date --iso-8601=seconds)
  printf '%s\tSTARTED\t%s\t\t\n' "${label}" "${started}" >> "${manifest}"
  echo "[START] ${label} ${started}"
  set +e
  timeout --kill-after=5m 24h "$@"
  rc=$?
  set -e
  finished=$(date --iso-8601=seconds)
  if [[ ${rc} -eq 0 ]]; then
    printf '%s\tDONE\t%s\t%s\t0\n' "${label}" "${started}" "${finished}" >> "${manifest}"
    echo "[DONE] ${label} ${finished}"
  else
    printf '%s\tFAILED\t%s\t%s\t%s\n' "${label}" "${started}" "${finished}" "${rc}" >> "${manifest}"
    echo "[FAILED-CONTINUE] ${label} rc=${rc} ${finished}"
  fi
}

audit_records() {
  local root=$1
  if [[ -e "${root}/audit/task_audit_records.jsonl" ]]; then
    echo "${root}/audit/task_audit_records.jsonl"
  elif [[ -e "${root}/audit/audit_records.jsonl" ]]; then
    echo "${root}/audit/audit_records.jsonl"
  else
    return 1
  fi
}

pair_stage() {
  local label=$1 task=$2 baseline_tag=$3 method_tag=$4
  local baseline_root=${5:-${run_base}/${task}/${baseline_tag}}
  local method_root=${run_base}/${task}/${method_tag}
  local output=${queue_root}/paired/${label}
  if [[ ! -e "${baseline_root}/DONE" || ! -e "${method_root}/DONE" ]]; then
    echo "[PAIR-SKIP] ${label}: a parent run is incomplete"
    record_stage_status "${label}" SKIPPED parent_incomplete
    return 0
  fi
  local baseline_records method_records
  baseline_records=$(audit_records "${baseline_root}") || {
    record_stage_status "${label}" FAILED baseline_records_missing
    return 0
  }
  method_records=$(audit_records "${method_root}") || {
    record_stage_status "${label}" FAILED method_records_missing
    return 0
  }
  run_stage "${label}" /root/miniconda3/bin/python compare_paired_task_runs.py \
    "${baseline_records}" "${method_records}" \
    --baseline-config "${baseline_root}/run_config.txt" \
    --method-config "${method_root}/run_config.txt" \
    --method-log "${queue_root}/${method_tag}.log" \
    --output-dir "${output}"
}

gate_pair() {
  local pair_label=$1 tolerance_correct=${2:-0}
  local summary=${queue_root}/paired/${pair_label}/paired_summary.json
  [[ -s "${summary}" ]] || return 1
  /root/miniconda3/bin/python - "${summary}" "${tolerance_correct}" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
tolerance = int(sys.argv[2])
healthy = (
    summary.get("prompt_hash_mismatches") == 0
    and summary.get("target_hash_mismatches") == 0
    and summary.get("duplicate_or_missing_ids") == 0
)
raise SystemExit(0 if healthy and summary["method_correct"] + tolerance >= summary["baseline_correct"] else 1)
PY
}

run_if_gate() {
  local gate_label=$1 tolerance=$2 stage_label=$3
  shift 3
  if gate_pair "${gate_label}" "${tolerance}"; then
    run_stage "${stage_label}" "$@"
  else
    echo "[GATE-SKIP] ${stage_label}: ${gate_label} did not pass tolerance=${tolerance}"
    record_stage_status "${stage_label}" SKIPPED "gate=${gate_label}"
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

if [[ ! -e "${manifest}" ]]; then
  printf 'stage\tstatus\tstart\tfinish\texit_code_or_note\n' > "${manifest}"
fi

echo "queue_id=${queue_id}"
echo "controller_start=$(date --iso-8601=seconds)"
echo "formal_baseline=original_llada_low_confidence"
echo "accuracy_parent=symmetric_attention_tau_0.004"
echo "new_vertical=response_credit_on_strong_conditioning_events"
echo "new_horizontal=two_draft_agreement_skeleton_and_disagreement_redenoising"
echo "code_selection=prompt_visible_examples_plus_behavior_consensus_no_hidden_tests"

/root/miniconda3/bin/python -m py_compile \
  generate.py eval_llada.py differential_selector.py audit_lm_eval_task.py \
  compare_paired_task_runs.py
PYTHONPATH=. /root/miniconda3/bin/python test_attention_stability.py
PYTHONPATH=. /root/miniconda3/bin/python test_differential_selector.py
PYTHONPATH=. /root/miniconda3/bin/python test_response_credit_exchange.py
PYTHONPATH=. /root/miniconda3/bin/python test_audit_lm_eval_task.py
/root/miniconda3/bin/python -m unittest discover -s tests -p 'test_*.py'

sha256sum generate.py eval_llada.py differential_selector.py model/modeling_llada.py \
  validate_step_diagnostics.py audit_attention_stability.py audit_lm_eval_task.py \
  compare_paired_task_runs.py postprocess_code.py sanitize.py \
  code_eval/code_eval.py code_eval/execute.py tasks/localleap_math500/math500.yaml \
  tasks/localleap_math500/utils.py "${runner}" "$0" \
  > "${queue_root}/FROZEN_SOURCE_SHA256"

# Real-model/evaluator smoke: the original baseline remains the formal control.
run_stage smoke_he_baseline "${runner}" humaneval 0 128 baseline 0.004 trace smoke_he_baseline 2 256
run_stage smoke_he_response_credit "${runner}" humaneval 0 128 response_credit 0.004 trace smoke_he_response_credit 2 256
run_stage smoke_he_response_credit_fast "${runner}" humaneval 0 128 response_credit_fast 0.004 trace smoke_he_response_credit_fast 2 256
run_stage smoke_he_draft_exchange_exec "${runner}" humaneval 0 128 draft_exchange_exec 0.004 trace smoke_he_draft_exchange_exec 2 256

# First code gate. All arms use identical examples, prompts, generation length and steps.
run_stage he_baseline_128_n32 "${runner}" humaneval 0 128 baseline 0.004 trace he_baseline_128_n32 32 256
for profile in response_credit response_credit_fast draft_exchange_exec; do
  tag="he_${profile}_128_n32"
  run_stage "${tag}" "${runner}" humaneval 0 128 "${profile}" 0.004 trace "${tag}" 32 256
  pair_stage "pair_${tag}" humaneval he_baseline_128_n32 "${tag}"
done

# The expensive two-draft branch gets a second gate before a full run.
run_if_gate pair_he_draft_exchange_exec_128_n32 1 he_baseline_128_n64 \
  "${runner}" humaneval 0 128 baseline 0.004 trace he_baseline_128_n64 64 256
run_if_gate pair_he_draft_exchange_exec_128_n32 1 he_draft_exchange_exec_128_n64 \
  "${runner}" humaneval 0 128 draft_exchange_exec 0.004 trace he_draft_exchange_exec_128_n64 64 256
pair_stage pair_he_draft_exchange_exec_128_n64 humaneval he_baseline_128_n64 he_draft_exchange_exec_128_n64

he_baseline_full=/root/autodl-tmp/LocalLeap/llada/results/attention_recovery/humaneval/recovery_he_baseline_128
run_if_gate pair_he_response_credit_128_n32 1 he_response_credit_128_full \
  "${runner}" humaneval 0 128 response_credit 0.004 trace he_response_credit_128_full full 256
pair_stage pair_he_response_credit_128_full humaneval unused he_response_credit_128_full "${he_baseline_full}"
run_if_gate pair_he_response_credit_fast_128_n32 1 he_response_credit_fast_128_full \
  "${runner}" humaneval 0 128 response_credit_fast 0.004 trace he_response_credit_fast_128_full full 256
pair_stage pair_he_response_credit_fast_128_full humaneval unused he_response_credit_fast_128_full "${he_baseline_full}"
run_if_gate pair_he_draft_exchange_exec_128_n64 0 he_draft_exchange_exec_128_full \
  "${runner}" humaneval 0 128 draft_exchange_exec 0.004 trace he_draft_exchange_exec_128_full full 256
pair_stage pair_he_draft_exchange_exec_128_full humaneval unused he_draft_exchange_exec_128_full "${he_baseline_full}"

# MBPP generalization uses a fresh 100-example baseline for timing, then promotes only non-negative arms.
run_stage mbpp_baseline_128_n100 "${runner}" mbpp 3 128 baseline 0.004 trace mbpp_baseline_128_n100 100 256
for profile in response_credit_fast draft_exchange_exec; do
  tag="mbpp_${profile}_128_n100"
  run_stage "${tag}" "${runner}" mbpp 3 128 "${profile}" 0.004 trace "${tag}" 100 256
  pair_stage "pair_${tag}" mbpp mbpp_baseline_128_n100 "${tag}"
done
mbpp_baseline_full=/root/autodl-tmp/LocalLeap/llada/results/b2_confirmatory/mbpp/baseline/stcc_b2_confirmatory_20260715_v1_mbpp_baseline
run_if_gate pair_mbpp_response_credit_fast_128_n100 0 mbpp_response_credit_fast_128_full \
  "${runner}" mbpp 3 128 response_credit_fast 0.004 trace mbpp_response_credit_fast_128_full full 256
pair_stage pair_mbpp_response_credit_fast_128_full mbpp unused mbpp_response_credit_fast_128_full "${mbpp_baseline_full}"
run_if_gate pair_mbpp_draft_exchange_exec_128_n100 0 mbpp_draft_exchange_exec_128_full \
  "${runner}" mbpp 3 128 draft_exchange_exec 0.004 trace mbpp_draft_exchange_exec_128_full full 256
pair_stage pair_mbpp_draft_exchange_exec_128_full mbpp unused mbpp_draft_exchange_exec_128_full "${mbpp_baseline_full}"

# Non-code generalization: no execution selector. These gates directly test the shared-skeleton repair.
for task_spec in "localleap_math500:0:100" "gsm8k:0:128"; do
  IFS=: read -r task shots count <<<"${task_spec}"
  run_stage "${task}_baseline_128_n${count}" "${runner}" "${task}" "${shots}" 128 baseline 0.004 trace "${task}_baseline_128_n${count}" "${count}" 256
  for profile in response_credit_fast draft_exchange; do
    tag="${task}_${profile}_128_n${count}"
    run_stage "${tag}" "${runner}" "${task}" "${shots}" 128 "${profile}" 0.004 trace "${tag}" "${count}" 256
    pair_stage "pair_${tag}" "${task}" "${task}_baseline_128_n${count}" "${tag}"
  done
done

echo "controller_finish=$(date --iso-8601=seconds)"
touch "${queue_root}/DONE"
