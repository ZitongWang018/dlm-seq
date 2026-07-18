#!/usr/bin/env bash
set -uo pipefail

queue_id=${ATTENTION_QUEUE_ID:-response_refine_gated_20260718_v2}
method_profile=${ATTENTION_METHOD_PROFILE:-response_refine_gated}
method_label=${ATTENTION_METHOD_LABEL:-gated}
llada_root=/root/autodl-tmp/LocalLeap/llada
source_root=/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability
runner=${source_root}/scripts/run_best_symmetric_benchmark.sh
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
pair_runs_subset() {
  local label=$1 base=$2 method=$3
  local br mr subset
  br=$(audit_records "${base}") || return 1
  mr=$(audit_records "${method}") || return 1
  subset=${queue_root}/subsets/${label}_baseline.jsonl
  mkdir -p "$(dirname "${subset}")"
  /root/miniconda3/bin/python - "${br}" "${mr}" "${subset}" <<'PY'
import json, sys
from pathlib import Path
def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def key(row):
    value = row.get("stable_task_id", row.get("task_id"))
    if value is None: raise ValueError("missing stable identity")
    return str(value)
source = {key(x): x for x in rows(sys.argv[1])}
method_ids = [key(x) for x in rows(sys.argv[2])]
if len(method_ids) != len(set(method_ids)): raise ValueError("duplicate method id")
missing = [x for x in method_ids if x not in source]
if missing: raise ValueError(f"missing baseline ids: {missing[:3]}")
Path(sys.argv[3]).write_text("".join(json.dumps(source[x], sort_keys=True)+"\n" for x in method_ids))
PY
  run_stage "${label}" /root/miniconda3/bin/python "${llada_root}/compare_paired_task_runs.py" \
    "${subset}" "${mr}" --baseline-config "${base}/run_config.txt" \
    --method-config "${method}/run_config.txt" \
    --method-log "${queue_root}/$(basename "${method}").log" \
    --output-dir "${queue_root}/paired/${label}"
}
correct_count() { /root/miniconda3/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["method_correct"])' "$1"; }
baseline_count() { /root/miniconda3/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["baseline_correct"])' "$1"; }

export PATH=/root/miniconda3/bin:${PATH}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=/root/autodl-tmp/.cache/huggingface HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_ALLOW_CODE_EVAL=1
cd "${llada_root}"
[[ -s "${manifest}" ]] || printf 'stage\tstatus\tstart\tfinish\texit_code_or_note\n' >"${manifest}"
if [[ ! -s "${frozen}" ]]; then
  sha256sum generate.py eval_llada.py compare_paired_task_runs.py audit_attention_stability.py audit_lm_eval_task.py \
    "${runner}" "${source_root}/scripts/run_response_refine_gated_queue.sh" >"${frozen}"
fi
verify
/root/miniconda3/bin/python -m py_compile generate.py eval_llada.py compare_paired_task_runs.py
PYTHONPATH=. /root/miniconda3/bin/python "${source_root}/tests/test_response_refine.py"
bash -n "${runner}" "${source_root}/scripts/run_response_refine_gated_queue.sh"

echo "formal_baseline=original_llada_low_confidence"
echo "development_parent=symmetric_fast_tau_0.004"
echo "method_profile=${method_profile}"
echo "method_label=${method_label}"

# Health smoke uses both GPUs concurrently.
run_gpu "smoke_he_${method_label}" 0 "${runner}" humaneval 0 128 "${method_profile}" 0.004 trace "smoke_he_${method_label}" 2 256 & p0=$!
run_gpu "smoke_math_${method_label}" 1 "${runner}" localleap_math500 0 128 "${method_profile}" 0.004 trace "smoke_math_${method_label}" 2 256 & p1=$!
wait "${p0}" || true; wait "${p1}" || true
[[ -e "${run_base}/humaneval/smoke_he_${method_label}/DONE" && -e "${run_base}/localleap_math500/smoke_math_${method_label}/DONE" ]] || { touch "${queue_root}/FAILED_SMOKE"; exit 22; }

# First formal sampled gate: one benchmark per GPU, identical prompt/evaluator setup.
run_gpu "he_${method_label}_n32" 0 "${runner}" humaneval 0 128 "${method_profile}" 0.004 trace "he_${method_label}_n32" 32 256 & p0=$!
run_gpu "math_${method_label}_n50" 1 "${runner}" localleap_math500 0 128 "${method_profile}" 0.004 trace "math_${method_label}_n50" 50 256 & p1=$!
wait "${p0}" || true; wait "${p1}" || true

v1=/root/autodl-tmp/LocalLeap/llada/results/best_symmetric_benchmarks/response_refine_he_math_20260718_v1
pair_runs he_gate_vs_base_n32 "${v1}/humaneval/he_baseline_128_n32" "${run_base}/humaneval/he_${method_label}_n32"
pair_runs he_gate_vs_parent_n32 "${v1}/humaneval/he_parent_128_n32" "${run_base}/humaneval/he_${method_label}_n32"
pair_runs math_gate_vs_base_n50 "${v1}/localleap_math500/math_baseline_128_n50" "${run_base}/localleap_math500/math_${method_label}_n50"
pair_runs math_gate_vs_parent_n50 "${v1}/localleap_math500/math_parent_128_n50" "${run_base}/localleap_math500/math_${method_label}_n50"

he_summary=${queue_root}/paired/he_gate_vs_parent_n32/paired_summary.json
math_summary=${queue_root}/paired/math_gate_vs_parent_n50/paired_summary.json
he_method=$(correct_count "${he_summary}"); he_parent=$(baseline_count "${he_summary}")
math_method=$(correct_count "${math_summary}"); math_parent=$(baseline_count "${math_summary}")
combined_method=$((he_method + math_method)); combined_parent=$((he_parent + math_parent))
echo "sample_gate method=${combined_method} parent=${combined_parent} he=${he_method}/${he_parent} math=${math_method}/${math_parent}"

# Clear failures stop. Borderline results receive a larger paired sample before promotion.
promote=0
if (( he_method + 1 >= he_parent && math_method + 1 >= math_parent && combined_method > combined_parent )); then
  promote=1
elif (( he_method + 1 >= he_parent && math_method + 1 >= math_parent && combined_method + 1 >= combined_parent )); then
  run_gpu "he_${method_label}_n64" 0 "${runner}" humaneval 0 128 "${method_profile}" 0.004 trace "he_${method_label}_n64" 64 256 & p0=$!
  run_gpu "math_${method_label}_n100" 1 "${runner}" localleap_math500 0 128 "${method_profile}" 0.004 trace "math_${method_label}_n100" 100 256 & p1=$!
  wait "${p0}" || true; wait "${p1}" || true
  he_full_base=/root/autodl-tmp/LocalLeap/llada/results/attention_recovery/humaneval/recovery_he_symmetric_fast_128
  math_full_base=/root/autodl-tmp/LocalLeap/llada/results/best_symmetric_benchmarks/best_symmetric_long_20260716_v2/localleap_math500/math500_symmetric_fast_128
  pair_runs_subset he_gate_vs_parent_n64 "${he_full_base}" "${run_base}/humaneval/he_${method_label}_n64"
  pair_runs_subset math_gate_vs_parent_n100 "${math_full_base}" "${run_base}/localleap_math500/math_${method_label}_n100"
  hs=${queue_root}/paired/he_gate_vs_parent_n64/paired_summary.json; ms=${queue_root}/paired/math_gate_vs_parent_n100/paired_summary.json
  hm=$(correct_count "${hs}"); hp=$(baseline_count "${hs}"); mm=$(correct_count "${ms}"); mp=$(baseline_count "${ms}")
  (( hm + 1 >= hp && mm + 1 >= mp && hm + mm > hp + mp )) && promote=1
fi

if (( ! promote )); then touch "${queue_root}/DONE_NO_PROMOTION" "${queue_root}/DONE"; exit 0; fi

run_gpu "he_${method_label}_full" 0 "${runner}" humaneval 0 128 "${method_profile}" 0.004 trace "he_${method_label}_full" full 256 & p0=$!
run_gpu "math_${method_label}_full" 1 "${runner}" localleap_math500 0 128 "${method_profile}" 0.004 trace "math_${method_label}_full" full 256 & p1=$!
wait "${p0}" || true; wait "${p1}" || true

he_base=/root/autodl-tmp/LocalLeap/llada/results/attention_recovery/humaneval/recovery_he_baseline_128
he_parent_full=/root/autodl-tmp/LocalLeap/llada/results/attention_recovery/humaneval/recovery_he_symmetric_fast_128
math_base=/root/autodl-tmp/LocalLeap/llada/results/best_symmetric_benchmarks/best_symmetric_long_20260716_v2/localleap_math500/math500_baseline_128
math_parent_full=/root/autodl-tmp/LocalLeap/llada/results/best_symmetric_benchmarks/best_symmetric_long_20260716_v2/localleap_math500/math500_symmetric_fast_128
pair_runs he_full_vs_base "${he_base}" "${run_base}/humaneval/he_${method_label}_full"
pair_runs he_full_vs_parent "${he_parent_full}" "${run_base}/humaneval/he_${method_label}_full"
pair_runs math_full_vs_base "${math_base}" "${run_base}/localleap_math500/math_${method_label}_full"
pair_runs math_full_vs_parent "${math_parent_full}" "${run_base}/localleap_math500/math_${method_label}_full"
touch "${queue_root}/DONE"
