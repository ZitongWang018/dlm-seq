#!/usr/bin/env bash
set -uo pipefail

queue_id=${ATTENTION_QUEUE_ID:-trajectory_disagreement_evidence_20260719_v1}
llada_root=/root/autodl-tmp/LocalLeap/llada
source_root=/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability
runner=${source_root}/scripts/run_best_symmetric_benchmark.sh
controller=${source_root}/scripts/run_disagreement_evidence_queue.sh
queue_root=${llada_root}/results/experiment_queues/${queue_id}
run_base=${llada_root}/results/best_symmetric_benchmarks/${queue_id}
manifest=${queue_root}/formal_manifest.tsv
frozen=${queue_root}/FROZEN_SOURCE_SHA256
mkdir -p "${queue_root}"
exec > >(tee -a "${queue_root}/formal_controller.log") 2>&1

append_manifest() { { flock 9; printf '%s\n' "$1" >&9; } 9>>"${manifest}"; }
done_stage() { [[ -s "${manifest}" ]] && awk -F '\t' -v x="$1" '$1==x && $2=="DONE"{ok=1} END{exit !ok}' "${manifest}"; }
verify() { (cd "${llada_root}" && sha256sum -c "${frozen}" >/dev/null) || { touch "${queue_root}/FAILED_SOURCE_DRIFT"; exit 21; }; }
run_stage() {
  local label=$1; shift
  done_stage "${label}" && return 0
  verify
  local start finish rc
  start=$(date --iso-8601=seconds)
  append_manifest "$(printf '%s\tSTARTED\t%s\t\t' "${label}" "${start}")"
  set +e
  timeout --kill-after=5m 12h "$@"
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
  local label=$1 baseline=$2 method=$3 br mr
  br=$(audit_records "${baseline}") || return 1
  mr=$(audit_records "${method}") || return 1
  run_stage "${label}" /root/miniconda3/bin/python "${llada_root}/compare_paired_task_runs.py" \
    "${br}" "${mr}" --baseline-config "${baseline}/run_config.txt" \
    --method-config "${method}/run_config.txt" \
    --method-log "${queue_root}/$(basename "${method}").log" \
    --output-dir "${queue_root}/paired/${label}"
}
field() { /root/miniconda3/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }
require_done() {
  local run
  for run in "$@"; do
    [[ -e "${run}/DONE" ]] || { touch "${queue_root}/FAILED_REQUIRED_RUN"; exit 23; }
  done
}

export PATH=/root/miniconda3/bin:${PATH}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=/root/autodl-tmp/.cache/huggingface HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_ALLOW_CODE_EVAL=1
cd "${llada_root}"
[[ -s "${manifest}" ]] || printf 'stage\tstatus\tstart\tfinish\texit_code_or_note\n' >"${manifest}"
if [[ ! -s "${frozen}" ]]; then
  sha256sum generate.py eval_llada.py compare_paired_task_runs.py audit_attention_stability.py \
    audit_lm_eval_task.py postprocess_code.py humaneval_execution.py sanitize.py \
    "${runner}" "${controller}" "${source_root}/slice_audit_records.py" \
    "${source_root}/tests/test_attention_stability.py" \
    "${source_root}/tests/test_slice_audit_records.py" >"${frozen}"
fi
verify
/root/miniconda3/bin/python -m py_compile generate.py eval_llada.py "${source_root}/slice_audit_records.py"
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_attention_stability.py"
/root/miniconda3/bin/python "${source_root}/tests/test_slice_audit_records.py"
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_compare_paired_task_runs.py"
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_humaneval_evaluator.py"
bash -n "${runner}" "${controller}"

echo "formal_baseline=original_llada_low_confidence"
echo "method=trajectory_disagreement_evidence_tau_0.004"
echo "selection=max_commit_log_evidence_on_final_disagreement_positions"
echo "fitted_hyperparameters=none"

run_gpu smoke_he_disagreement 0 "${runner}" humaneval 0 128 trajectory_disagreement_evidence 0.004 trace smoke_he_disagreement 2 256 & p0=$!
run_gpu smoke_he_fast 1 "${runner}" humaneval 0 128 symmetric_fast 0.004 trace smoke_he_fast 2 256 & p1=$!
wait "${p0}" || true; wait "${p1}" || true
require_done "${run_base}/humaneval/smoke_he_disagreement" "${run_base}/humaneval/smoke_he_fast"

# GPU0 evaluates the two-path method while GPU1 regenerates both comparators
# from the same frozen source. HumanEval/64..95 is untouched validation data.
run_gpu he_disagreement_n96 0 "${runner}" humaneval 0 128 trajectory_disagreement_evidence 0.004 trace he_disagreement_n96 96 256 & p0=$!
(
  run_gpu he_fast_n96 1 "${runner}" humaneval 0 128 symmetric_fast 0.004 trace he_fast_n96 96 256
  run_gpu he_base_n96 1 "${runner}" humaneval 0 128 baseline 0 trace he_base_n96 96 256
) & p1=$!
wait "${p0}" || true; wait "${p1}" || true
require_done "${run_base}/humaneval/he_disagreement_n96" "${run_base}/humaneval/he_fast_n96" \
  "${run_base}/humaneval/he_base_n96"

pair_runs he_disagreement_vs_fast_n96 "${run_base}/humaneval/he_fast_n96" "${run_base}/humaneval/he_disagreement_n96"
pair_runs he_disagreement_vs_base_n96 "${run_base}/humaneval/he_base_n96" "${run_base}/humaneval/he_disagreement_n96"
for name in fast base disagreement; do
  input=$(audit_records "${run_base}/humaneval/he_${name}_n96")
  /root/miniconda3/bin/python "${source_root}/slice_audit_records.py" "${input}" \
    "${queue_root}/holdout/he_${name}_64_96.jsonl" --start 64 --end 96
done
run_stage he_holdout_vs_fast /root/miniconda3/bin/python "${llada_root}/compare_paired_task_runs.py" \
  "${queue_root}/holdout/he_fast_64_96.jsonl" "${queue_root}/holdout/he_disagreement_64_96.jsonl" \
  --baseline-config "${run_base}/humaneval/he_fast_n96/run_config.txt" \
  --method-config "${run_base}/humaneval/he_disagreement_n96/run_config.txt" \
  --method-log "${queue_root}/he_disagreement_n96.log" --output-dir "${queue_root}/paired/he_holdout_vs_fast"
run_stage he_holdout_vs_base /root/miniconda3/bin/python "${llada_root}/compare_paired_task_runs.py" \
  "${queue_root}/holdout/he_base_64_96.jsonl" "${queue_root}/holdout/he_disagreement_64_96.jsonl" \
  --baseline-config "${run_base}/humaneval/he_base_n96/run_config.txt" \
  --method-config "${run_base}/humaneval/he_disagreement_n96/run_config.txt" \
  --method-log "${queue_root}/he_disagreement_n96.log" --output-dir "${queue_root}/paired/he_holdout_vs_base"

hf="${queue_root}/paired/he_holdout_vs_fast/paired_summary.json"
hb="${queue_root}/paired/he_holdout_vs_base/paired_summary.json"
hm=$(field "${hf}" method_correct); hfast=$(field "${hf}" baseline_correct)
hbase=$(field "${hb}" baseline_correct); hrecover=$(field "${hf}" method_only)
echo "holdout_gate method=${hm} fast=${hfast} base=${hbase} recoveries=${hrecover}"
if ! (( hm > hfast && hm > hbase && hrecover > 0 )); then
  touch "${queue_root}/DONE_NO_MATH" "${queue_root}/DONE"
  exit 0
fi

# Cross-domain promotion only after an unseen HumanEval suffix gain.
(
  run_gpu math_disagreement_n50 0 "${runner}" localleap_math500 0 128 trajectory_disagreement_evidence 0.004 trace math_disagreement_n50 50 256
  run_gpu math_base_n50 0 "${runner}" localleap_math500 0 128 baseline 0 trace math_base_n50 50 256
) & p0=$!
(
  run_gpu math_fast_n50 1 "${runner}" localleap_math500 0 128 symmetric_fast 0.004 trace math_fast_n50 50 256
) & p1=$!
wait "${p0}" || true; wait "${p1}" || true
require_done "${run_base}/localleap_math500/math_disagreement_n50" "${run_base}/localleap_math500/math_base_n50" \
  "${run_base}/localleap_math500/math_fast_n50"
pair_runs math_disagreement_vs_fast_n50 "${run_base}/localleap_math500/math_fast_n50" "${run_base}/localleap_math500/math_disagreement_n50"
pair_runs math_disagreement_vs_base_n50 "${run_base}/localleap_math500/math_base_n50" "${run_base}/localleap_math500/math_disagreement_n50"
mf="${queue_root}/paired/math_disagreement_vs_fast_n50/paired_summary.json"
mb="${queue_root}/paired/math_disagreement_vs_base_n50/paired_summary.json"
mm=$(field "${mf}" method_correct); mfast=$(field "${mf}" baseline_correct); mbase=$(field "${mb}" baseline_correct)
echo "math_gate method=${mm} fast=${mfast} base=${mb}"
if ! (( mm >= mfast && mm >= mbase )); then
  touch "${queue_root}/DONE_NO_GENERALIZATION" "${queue_root}/DONE"
  exit 0
fi

# Small cross-task discovery keeps overnight iteration bounded.
(
  run_gpu mbpp_base_n50 0 "${runner}" mbpp 0 128 baseline 0 trace mbpp_base_n50 50 256
  run_gpu mbpp_fast_n50 0 "${runner}" mbpp 0 128 symmetric_fast 0.004 trace mbpp_fast_n50 50 256
  run_gpu mbpp_disagreement_n50 0 "${runner}" mbpp 0 128 trajectory_disagreement_evidence 0.004 trace mbpp_disagreement_n50 50 256
  pair_runs mbpp_disagreement_vs_base_n50 "${run_base}/mbpp/mbpp_base_n50" "${run_base}/mbpp/mbpp_disagreement_n50"
  pair_runs mbpp_disagreement_vs_fast_n50 "${run_base}/mbpp/mbpp_fast_n50" "${run_base}/mbpp/mbpp_disagreement_n50"
) & p0=$!
(
  run_gpu gsm_base_n64 1 "${runner}" gsm8k 0 128 baseline 0 trace gsm_base_n64 64 256
  run_gpu gsm_fast_n64 1 "${runner}" gsm8k 0 128 symmetric_fast 0.004 trace gsm_fast_n64 64 256
  run_gpu gsm_disagreement_n64 1 "${runner}" gsm8k 0 128 trajectory_disagreement_evidence 0.004 trace gsm_disagreement_n64 64 256
  pair_runs gsm_disagreement_vs_base_n64 "${run_base}/gsm8k/gsm_base_n64" "${run_base}/gsm8k/gsm_disagreement_n64"
  pair_runs gsm_disagreement_vs_fast_n64 "${run_base}/gsm8k/gsm_fast_n64" "${run_base}/gsm8k/gsm_disagreement_n64"
) & p1=$!
wait "${p0}" || true; wait "${p1}" || true
touch "${queue_root}/DONE"
