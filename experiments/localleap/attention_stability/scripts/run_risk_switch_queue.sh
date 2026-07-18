#!/usr/bin/env bash
set -uo pipefail

queue_id=${ATTENTION_QUEUE_ID:-confidence_switched_stability_20260718_v1}
llada_root=/root/autodl-tmp/LocalLeap/llada
source_root=/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability
runner=${source_root}/scripts/run_best_symmetric_benchmark.sh
controller=${source_root}/scripts/run_risk_switch_queue.sh
queue_root=${llada_root}/results/experiment_queues/${queue_id}
run_base=${llada_root}/results/best_symmetric_benchmarks/${queue_id}
manifest=${queue_root}/formal_manifest.tsv
frozen=${queue_root}/FROZEN_SOURCE_SHA256
mkdir -p "${queue_root}"
exec > >(tee -a "${queue_root}/formal_controller.log") 2>&1

append_manifest() { { flock 9; printf '%s\n' "$1" >&9; } 9>>"${manifest}"; }
done_stage() { [[ -s "${manifest}" ]] && awk -F '\t' -v x="$1" '$1==x && $2=="DONE"{ok=1} END{exit !ok}' "${manifest}"; }
verify() { sha256sum -c "${frozen}" >/dev/null || { touch "${queue_root}/FAILED_SOURCE_DRIFT"; exit 21; }; }
run_stage() {
  local label=$1; shift
  done_stage "${label}" && return 0
  verify
  local start finish rc
  start=$(date --iso-8601=seconds)
  append_manifest "$(printf '%s\tSTARTED\t%s\t\t' "${label}" "${start}")"
  set +e
  timeout --kill-after=5m 24h "$@"
  rc=$?
  set -e
  finish=$(date --iso-8601=seconds)
  if [[ ${rc} -eq 0 ]]; then
    append_manifest "$(printf '%s\tDONE\t%s\t%s\t0' "${label}" "${start}" "${finish}")"
  else
    append_manifest "$(printf '%s\tFAILED\t%s\t%s\t%s' "${label}" "${start}" "${finish}" "${rc}")"
  fi
  return "${rc}"
}
run_gpu() { local label=$1 gpu=$2; shift 2; run_stage "${label}" env CUDA_VISIBLE_DEVICES="${gpu}" "$@"; }
audit_records() {
  [[ -s "$1/audit/audit_records.jsonl" ]] && { echo "$1/audit/audit_records.jsonl"; return; }
  [[ -s "$1/audit/task_audit_records.jsonl" ]] && echo "$1/audit/task_audit_records.jsonl"
}
pair_runs() {
  local label=$1 base=$2 method=$3
  local br mr
  br=$(audit_records "${base}") || return 1
  mr=$(audit_records "${method}") || return 1
  run_stage "${label}" /root/miniconda3/bin/python "${llada_root}/compare_paired_task_runs.py" \
    "${br}" "${mr}" --baseline-config "${base}/run_config.txt" \
    --method-config "${method}/run_config.txt" \
    --method-log "${queue_root}/$(basename "${method}").log" \
    --output-dir "${queue_root}/paired/${label}"
}
field() { /root/miniconda3/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }
require_done() {
  local run
  for run in "$@"; do
    if [[ ! -e "${run}/DONE" ]]; then
      echo "required run did not complete: ${run}"
      touch "${queue_root}/FAILED_REQUIRED_RUN"
      exit 23
    fi
  done
}
gate_pair() {
  local he_summary=$1 math_summary=$2 max_nfe_pct=$3
  local hm hp mm mp hn hb mn mb
  hm=$(field "${he_summary}" method_correct); hp=$(field "${he_summary}" baseline_correct)
  mm=$(field "${math_summary}" method_correct); mp=$(field "${math_summary}" baseline_correct)
  hn=$(field "${he_summary}" method_total_nfe); hb=$(field "${he_summary}" baseline_total_nfe)
  mn=$(field "${math_summary}" method_total_nfe); mb=$(field "${math_summary}" baseline_total_nfe)
  echo "gate he=${hm}/${hp} math=${mm}/${mp} nfe_he=${hn}/${hb} nfe_math=${mn}/${mb}"
  (( hm + 1 >= hp && mm + 1 >= mp && hm + mm > hp + mp \
     && hn * 100 <= hb * max_nfe_pct && mn * 100 <= mb * max_nfe_pct ))
}

export PATH=/root/miniconda3/bin:${PATH}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=/root/autodl-tmp/.cache/huggingface HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_ALLOW_CODE_EVAL=1
cd "${llada_root}"
[[ -s "${manifest}" ]] || printf 'stage\tstatus\tstart\tfinish\texit_code_or_note\n' >"${manifest}"
if [[ ! -s "${frozen}" ]]; then
  sha256sum generate.py eval_llada.py compare_paired_task_runs.py audit_attention_stability.py \
    audit_lm_eval_task.py postprocess_code.py sanitize.py "${runner}" "${controller}" \
    "${source_root}/tests/test_attention_stability.py" \
    "${source_root}/tests/test_risk_switch_queue.py" >"${frozen}"
fi
verify
/root/miniconda3/bin/python -m py_compile generate.py eval_llada.py compare_paired_task_runs.py
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_attention_stability.py"
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_compare_paired_task_runs.py"
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_humaneval_evaluator.py"
/root/miniconda3/bin/python "${source_root}/tests/test_risk_switch_queue.py"
bash -n "${runner}" "${controller}"

echo "formal_baseline=original_llada_low_confidence"
echo "fast_parent=symmetric_fast_tau_0.004"
echo "accuracy_reference=symmetric_tau_0.004"
echo "method=symmetric_risk_switch_tau_0.004"
echo "switch=fill_stable_conflicts_wait_on_conditioned_rewrite"
echo "new_hyperparameters=none"

# Real-model two-example preflight on both task families.
run_gpu smoke_he_switch 0 "${runner}" humaneval 0 128 symmetric_risk_switch 0.004 trace smoke_he_switch 2 256 & p0=$!
run_gpu smoke_math_switch 1 "${runner}" localleap_math500 0 128 symmetric_risk_switch 0.004 trace smoke_math_switch 2 256 & p1=$!
wait "${p0}" || true; wait "${p1}" || true
[[ -e "${run_base}/humaneval/smoke_he_switch/DONE" && -e "${run_base}/localleap_math500/smoke_math_switch/DONE" ]] || { touch "${queue_root}/FAILED_SMOKE"; exit 22; }

# Discovery: regenerate every comparator from the same frozen source.
(
  run_gpu he_base_n32 0 "${runner}" humaneval 0 128 baseline 0 trace he_base_n32 32 256
  run_gpu he_fast_n32 0 "${runner}" humaneval 0 128 symmetric_fast 0.004 trace he_fast_n32 32 256
  run_gpu he_accuracy_n32 0 "${runner}" humaneval 0 128 symmetric 0.004 trace he_accuracy_n32 32 256
  run_gpu he_switch_n32 0 "${runner}" humaneval 0 128 symmetric_risk_switch 0.004 trace he_switch_n32 32 256
) & p0=$!
(
  run_gpu math_base_n50 1 "${runner}" localleap_math500 0 128 baseline 0 trace math_base_n50 50 256
  run_gpu math_fast_n50 1 "${runner}" localleap_math500 0 128 symmetric_fast 0.004 trace math_fast_n50 50 256
  run_gpu math_accuracy_n50 1 "${runner}" localleap_math500 0 128 symmetric 0.004 trace math_accuracy_n50 50 256
  run_gpu math_switch_n50 1 "${runner}" localleap_math500 0 128 symmetric_risk_switch 0.004 trace math_switch_n50 50 256
) & p1=$!
wait "${p0}" || true; wait "${p1}" || true
require_done \
  "${run_base}/humaneval/he_base_n32" \
  "${run_base}/humaneval/he_fast_n32" \
  "${run_base}/humaneval/he_accuracy_n32" \
  "${run_base}/humaneval/he_switch_n32" \
  "${run_base}/localleap_math500/math_base_n50" \
  "${run_base}/localleap_math500/math_fast_n50" \
  "${run_base}/localleap_math500/math_accuracy_n50" \
  "${run_base}/localleap_math500/math_switch_n50"

pair_runs he_switch_vs_fast_n32 "${run_base}/humaneval/he_fast_n32" "${run_base}/humaneval/he_switch_n32"
pair_runs he_switch_vs_base_n32 "${run_base}/humaneval/he_base_n32" "${run_base}/humaneval/he_switch_n32"
pair_runs he_switch_vs_accuracy_n32 "${run_base}/humaneval/he_accuracy_n32" "${run_base}/humaneval/he_switch_n32"
pair_runs math_switch_vs_fast_n50 "${run_base}/localleap_math500/math_fast_n50" "${run_base}/localleap_math500/math_switch_n50"
pair_runs math_switch_vs_base_n50 "${run_base}/localleap_math500/math_base_n50" "${run_base}/localleap_math500/math_switch_n50"
pair_runs math_switch_vs_accuracy_n50 "${run_base}/localleap_math500/math_accuracy_n50" "${run_base}/localleap_math500/math_switch_n50"

if ! gate_pair "${queue_root}/paired/he_switch_vs_fast_n32/paired_summary.json" \
  "${queue_root}/paired/math_switch_vs_fast_n50/paired_summary.json" 135; then
  touch "${queue_root}/DONE_NO_EXPANSION" "${queue_root}/DONE"
  exit 0
fi

# Larger, disjoint-cost gate.
(
  run_gpu he_base_n64 0 "${runner}" humaneval 0 128 baseline 0 trace he_base_n64 64 256
  run_gpu he_fast_n64 0 "${runner}" humaneval 0 128 symmetric_fast 0.004 trace he_fast_n64 64 256
  run_gpu he_switch_n64 0 "${runner}" humaneval 0 128 symmetric_risk_switch 0.004 trace he_switch_n64 64 256
) & p0=$!
(
  run_gpu math_base_n100 1 "${runner}" localleap_math500 0 128 baseline 0 trace math_base_n100 100 256
  run_gpu math_fast_n100 1 "${runner}" localleap_math500 0 128 symmetric_fast 0.004 trace math_fast_n100 100 256
  run_gpu math_switch_n100 1 "${runner}" localleap_math500 0 128 symmetric_risk_switch 0.004 trace math_switch_n100 100 256
) & p1=$!
wait "${p0}" || true; wait "${p1}" || true
require_done \
  "${run_base}/humaneval/he_base_n64" \
  "${run_base}/humaneval/he_fast_n64" \
  "${run_base}/humaneval/he_switch_n64" \
  "${run_base}/localleap_math500/math_base_n100" \
  "${run_base}/localleap_math500/math_fast_n100" \
  "${run_base}/localleap_math500/math_switch_n100"

pair_runs he_switch_vs_fast_n64 "${run_base}/humaneval/he_fast_n64" "${run_base}/humaneval/he_switch_n64"
pair_runs he_switch_vs_base_n64 "${run_base}/humaneval/he_base_n64" "${run_base}/humaneval/he_switch_n64"
pair_runs math_switch_vs_fast_n100 "${run_base}/localleap_math500/math_fast_n100" "${run_base}/localleap_math500/math_switch_n100"
pair_runs math_switch_vs_base_n100 "${run_base}/localleap_math500/math_base_n100" "${run_base}/localleap_math500/math_switch_n100"

if ! gate_pair "${queue_root}/paired/he_switch_vs_fast_n64/paired_summary.json" \
  "${queue_root}/paired/math_switch_vs_fast_n100/paired_summary.json" 135; then
  touch "${queue_root}/DONE_NO_PROMOTION" "${queue_root}/DONE"
  exit 0
fi

# Full HumanEval and MATH-500 confirmation in parallel.
(
  run_gpu he_base_full 0 "${runner}" humaneval 0 128 baseline 0 trace he_base_full full 256
  run_gpu he_fast_full 0 "${runner}" humaneval 0 128 symmetric_fast 0.004 trace he_fast_full full 256
  run_gpu he_switch_full 0 "${runner}" humaneval 0 128 symmetric_risk_switch 0.004 trace he_switch_full full 256
) & p0=$!
(
  run_gpu math_base_full 1 "${runner}" localleap_math500 0 128 baseline 0 trace math_base_full full 256
  run_gpu math_fast_full 1 "${runner}" localleap_math500 0 128 symmetric_fast 0.004 trace math_fast_full full 256
  run_gpu math_switch_full 1 "${runner}" localleap_math500 0 128 symmetric_risk_switch 0.004 trace math_switch_full full 256
) & p1=$!
wait "${p0}" || true; wait "${p1}" || true
require_done \
  "${run_base}/humaneval/he_base_full" \
  "${run_base}/humaneval/he_fast_full" \
  "${run_base}/humaneval/he_switch_full" \
  "${run_base}/localleap_math500/math_base_full" \
  "${run_base}/localleap_math500/math_fast_full" \
  "${run_base}/localleap_math500/math_switch_full"

pair_runs he_switch_vs_fast_full "${run_base}/humaneval/he_fast_full" "${run_base}/humaneval/he_switch_full"
pair_runs he_switch_vs_base_full "${run_base}/humaneval/he_base_full" "${run_base}/humaneval/he_switch_full"
pair_runs math_switch_vs_fast_full "${run_base}/localleap_math500/math_fast_full" "${run_base}/localleap_math500/math_switch_full"
pair_runs math_switch_vs_base_full "${run_base}/localleap_math500/math_base_full" "${run_base}/localleap_math500/math_switch_full"

# Cross-task sampled generalization follows only a successful HE/MATH promotion.
(
  run_gpu mbpp_base_n100 0 "${runner}" mbpp 0 128 baseline 0 trace mbpp_base_n100 100 256
  run_gpu mbpp_fast_n100 0 "${runner}" mbpp 0 128 symmetric_fast 0.004 trace mbpp_fast_n100 100 256
  run_gpu mbpp_switch_n100 0 "${runner}" mbpp 0 128 symmetric_risk_switch 0.004 trace mbpp_switch_n100 100 256
  pair_runs mbpp_switch_vs_fast_n100 "${run_base}/mbpp/mbpp_fast_n100" "${run_base}/mbpp/mbpp_switch_n100"
  pair_runs mbpp_switch_vs_base_n100 "${run_base}/mbpp/mbpp_base_n100" "${run_base}/mbpp/mbpp_switch_n100"
) & p0=$!
(
  run_gpu gsm_base_n128 1 "${runner}" gsm8k 0 128 baseline 0 trace gsm_base_n128 128 256
  run_gpu gsm_fast_n128 1 "${runner}" gsm8k 0 128 symmetric_fast 0.004 trace gsm_fast_n128 128 256
  run_gpu gsm_switch_n128 1 "${runner}" gsm8k 0 128 symmetric_risk_switch 0.004 trace gsm_switch_n128 128 256
  pair_runs gsm_switch_vs_fast_n128 "${run_base}/gsm8k/gsm_fast_n128" "${run_base}/gsm8k/gsm_switch_n128"
  pair_runs gsm_switch_vs_base_n128 "${run_base}/gsm8k/gsm_base_n128" "${run_base}/gsm8k/gsm_switch_n128"
) & p1=$!
wait "${p0}" || true; wait "${p1}" || true
touch "${queue_root}/DONE"
