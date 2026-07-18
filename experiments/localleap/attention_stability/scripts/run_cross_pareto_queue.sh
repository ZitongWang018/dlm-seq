#!/usr/bin/env bash
set -uo pipefail

queue_id=${ATTENTION_QUEUE_ID:-response_refine_cross_pareto_20260718_v1}
llada_root=/root/autodl-tmp/LocalLeap/llada
source_root=/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability
runner=${source_root}/scripts/run_best_symmetric_benchmark.sh
controller=${source_root}/scripts/run_cross_pareto_queue.sh
summarizer=${source_root}/summarize_response_refine.py
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
  set +e; timeout --kill-after=5m 24h "$@"; rc=$?; set -e
  finish=$(date --iso-8601=seconds)
  if [[ ${rc} -eq 0 ]]; then
    append_manifest "$(printf '%s\tDONE\t%s\t%s\t0' "${label}" "${start}" "${finish}")"
  else
    append_manifest "$(printf '%s\tFAILED\t%s\t%s\t%s' "${label}" "${start}" "${finish}" "${rc}")"
  fi
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
correct_count() { /root/miniconda3/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["method_correct"])' "$1"; }
baseline_count() { /root/miniconda3/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["baseline_correct"])' "$1"; }
summarize_run() {
  local label=$1 run=$2
  run_stage "${label}" /root/miniconda3/bin/python "${summarizer}" \
    "${run}/trace/rank_0.jsonl" --output "${run}/trace_summary_v3.json"
}

export PATH=/root/miniconda3/bin:${PATH}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=/root/autodl-tmp/.cache/huggingface HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_ALLOW_CODE_EVAL=1
cd "${llada_root}"
[[ -s "${manifest}" ]] || printf 'stage\tstatus\tstart\tfinish\texit_code_or_note\n' >"${manifest}"
if [[ ! -s "${frozen}" ]]; then
  sha256sum generate.py eval_llada.py differential_selector.py compare_paired_task_runs.py \
    audit_attention_stability.py audit_lm_eval_task.py "${runner}" "${controller}" \
    "${summarizer}" >"${frozen}"
fi
verify
/root/miniconda3/bin/python -m py_compile generate.py eval_llada.py differential_selector.py compare_paired_task_runs.py
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_response_refine.py"
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_differential_selector.py"
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_compare_paired_task_runs.py"
bash -n "${runner}" "${controller}"

echo "formal_baseline=original_llada_low_confidence"
echo "development_parent=symmetric_fast_tau_0.004"
echo "method_core=response_refine_cross_pareto"
echo "code_safety_child=response_refine_cross_pareto_exec"
echo "pairing_policy=same_queue_same_frozen_source_only"

# Two-GPU smoke, including the batch-two cross-condition verifier on code.
run_gpu smoke_he_cross_exec 0 "${runner}" humaneval 0 128 response_refine_cross_pareto_exec 0.004 trace smoke_he_cross_exec 2 256 & p0=$!
run_gpu smoke_math_cross 1 "${runner}" localleap_math500 0 128 response_refine_cross_pareto 0.004 trace smoke_math_cross 2 256 & p1=$!
wait "${p0}" || true; wait "${p1}" || true
[[ -e "${run_base}/humaneval/smoke_he_cross_exec/DONE" && -e "${run_base}/localleap_math500/smoke_math_cross/DONE" ]] || { touch "${queue_root}/FAILED_SMOKE"; exit 22; }

# Discovery gate.  Every baseline and parent is regenerated from this frozen source.
(
  run_gpu he_base_n32 0 "${runner}" humaneval 0 128 baseline 0 trace he_base_n32 32 256
  run_gpu he_parent_n32 0 "${runner}" humaneval 0 128 symmetric_fast 0.004 trace he_parent_n32 32 256
  run_gpu he_cross_n32 0 "${runner}" humaneval 0 128 response_refine_cross_pareto 0.004 trace he_cross_n32 32 256
  run_gpu he_cross_exec_n32 0 "${runner}" humaneval 0 128 response_refine_cross_pareto_exec 0.004 trace he_cross_exec_n32 32 256
) & p0=$!
(
  run_gpu math_base_n50 1 "${runner}" localleap_math500 0 128 baseline 0 trace math_base_n50 50 256
  run_gpu math_parent_n50 1 "${runner}" localleap_math500 0 128 symmetric_fast 0.004 trace math_parent_n50 50 256
  run_gpu math_cross_n50 1 "${runner}" localleap_math500 0 128 response_refine_cross_pareto 0.004 trace math_cross_n50 50 256
) & p1=$!
wait "${p0}" || true; wait "${p1}" || true

summarize_run summarize_he_cross_n32 "${run_base}/humaneval/he_cross_n32"
summarize_run summarize_he_cross_exec_n32 "${run_base}/humaneval/he_cross_exec_n32"
summarize_run summarize_math_cross_n50 "${run_base}/localleap_math500/math_cross_n50"

pair_runs he_core_vs_parent_n32 "${run_base}/humaneval/he_parent_n32" "${run_base}/humaneval/he_cross_n32"
pair_runs he_exec_vs_parent_n32 "${run_base}/humaneval/he_parent_n32" "${run_base}/humaneval/he_cross_exec_n32"
pair_runs he_exec_vs_base_n32 "${run_base}/humaneval/he_base_n32" "${run_base}/humaneval/he_cross_exec_n32"
pair_runs math_vs_parent_n50 "${run_base}/localleap_math500/math_parent_n50" "${run_base}/localleap_math500/math_cross_n50"
pair_runs math_vs_base_n50 "${run_base}/localleap_math500/math_base_n50" "${run_base}/localleap_math500/math_cross_n50"

hs=${queue_root}/paired/he_exec_vs_parent_n32/paired_summary.json
ms=${queue_root}/paired/math_vs_parent_n50/paired_summary.json
hm=$(correct_count "${hs}"); hp=$(baseline_count "${hs}")
mm=$(correct_count "${ms}"); mp=$(baseline_count "${ms}")
echo "discovery_gate he=${hm}/${hp} math=${mm}/${mp} combined=$((hm+mm))/$((hp+mp))"
expand=0
if (( hm >= hp && mm >= mp && hm + mm > hp + mp )); then
  expand=1
elif (( hm + 1 >= hp && mm + 1 >= mp && hm + mm >= hp + mp )); then
  expand=1
fi
if (( ! expand )); then touch "${queue_root}/DONE_NO_EXPANSION" "${queue_root}/DONE"; exit 0; fi

# Larger matched-source gate; original LLaDA remains a formal comparator.
(
  run_gpu he_base_n64 0 "${runner}" humaneval 0 128 baseline 0 trace he_base_n64 64 256
  run_gpu he_parent_n64 0 "${runner}" humaneval 0 128 symmetric_fast 0.004 trace he_parent_n64 64 256
  run_gpu he_cross_exec_n64 0 "${runner}" humaneval 0 128 response_refine_cross_pareto_exec 0.004 trace he_cross_exec_n64 64 256
) & p0=$!
(
  run_gpu math_base_n100 1 "${runner}" localleap_math500 0 128 baseline 0 trace math_base_n100 100 256
  run_gpu math_parent_n100 1 "${runner}" localleap_math500 0 128 symmetric_fast 0.004 trace math_parent_n100 100 256
  run_gpu math_cross_n100 1 "${runner}" localleap_math500 0 128 response_refine_cross_pareto 0.004 trace math_cross_n100 100 256
) & p1=$!
wait "${p0}" || true; wait "${p1}" || true

summarize_run summarize_he_cross_exec_n64 "${run_base}/humaneval/he_cross_exec_n64"
summarize_run summarize_math_cross_n100 "${run_base}/localleap_math500/math_cross_n100"

pair_runs he_exec_vs_parent_n64 "${run_base}/humaneval/he_parent_n64" "${run_base}/humaneval/he_cross_exec_n64"
pair_runs he_exec_vs_base_n64 "${run_base}/humaneval/he_base_n64" "${run_base}/humaneval/he_cross_exec_n64"
pair_runs math_vs_parent_n100 "${run_base}/localleap_math500/math_parent_n100" "${run_base}/localleap_math500/math_cross_n100"
pair_runs math_vs_base_n100 "${run_base}/localleap_math500/math_base_n100" "${run_base}/localleap_math500/math_cross_n100"

hs=${queue_root}/paired/he_exec_vs_parent_n64/paired_summary.json
ms=${queue_root}/paired/math_vs_parent_n100/paired_summary.json
hm=$(correct_count "${hs}"); hp=$(baseline_count "${hs}")
mm=$(correct_count "${ms}"); mp=$(baseline_count "${ms}")
echo "expanded_gate he=${hm}/${hp} math=${mm}/${mp} combined=$((hm+mm))/$((hp+mp))"
if (( hm < hp || mm < mp || hm + mm <= hp + mp )); then
  touch "${queue_root}/DONE_NO_PROMOTION" "${queue_root}/DONE"
  exit 0
fi

# Full formal confirmation.  These are deliberately regenerated rather than
# paired to historical outputs from a different source snapshot.
(
  run_gpu he_base_full 0 "${runner}" humaneval 0 128 baseline 0 trace he_base_full full 256
  run_gpu he_parent_full 0 "${runner}" humaneval 0 128 symmetric_fast 0.004 trace he_parent_full full 256
  run_gpu he_cross_exec_full 0 "${runner}" humaneval 0 128 response_refine_cross_pareto_exec 0.004 trace he_cross_exec_full full 256
) & p0=$!
(
  run_gpu math_base_full 1 "${runner}" localleap_math500 0 128 baseline 0 trace math_base_full full 256
  run_gpu math_parent_full 1 "${runner}" localleap_math500 0 128 symmetric_fast 0.004 trace math_parent_full full 256
  run_gpu math_cross_full 1 "${runner}" localleap_math500 0 128 response_refine_cross_pareto 0.004 trace math_cross_full full 256
) & p1=$!
wait "${p0}" || true; wait "${p1}" || true

summarize_run summarize_he_cross_exec_full "${run_base}/humaneval/he_cross_exec_full"
summarize_run summarize_math_cross_full "${run_base}/localleap_math500/math_cross_full"

pair_runs he_full_vs_parent "${run_base}/humaneval/he_parent_full" "${run_base}/humaneval/he_cross_exec_full"
pair_runs he_full_vs_base "${run_base}/humaneval/he_base_full" "${run_base}/humaneval/he_cross_exec_full"
pair_runs math_full_vs_parent "${run_base}/localleap_math500/math_parent_full" "${run_base}/localleap_math500/math_cross_full"
pair_runs math_full_vs_base "${run_base}/localleap_math500/math_base_full" "${run_base}/localleap_math500/math_cross_full"
touch "${queue_root}/DONE"
