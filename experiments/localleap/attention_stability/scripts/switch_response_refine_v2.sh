#!/usr/bin/env bash
set -uo pipefail
old_q=/root/autodl-tmp/LocalLeap/llada/results/experiment_queues/response_refine_he_math_20260718_v1
old_b=/root/autodl-tmp/LocalLeap/llada/results/best_symmetric_benchmarks/response_refine_he_math_20260718_v1
stage=/root/autodl-tmp/response_refine_v2_staging
repo=/root/autodl-tmp/dlm-seq-flow
src=${repo}/experiments/localleap/attention_stability
live=/root/autodl-tmp/LocalLeap/llada
new_id=response_refine_gated_20260718_v2
new_q=${live}/results/experiment_queues/${new_id}
log=${old_q}/v2_switcher.log
exec >>"${log}" 2>&1
echo "switcher_start=$(date --iso-8601=seconds)"

while [[ ! -e "${old_b}/localleap_math500/math_refine_fast_128_n50/DONE" \
      || ! -e "${old_b}/localleap_math500/math_refine_extra_128_n50/DONE" ]]; do
  echo "waiting $(date --iso-8601=seconds)"
  sleep 60
done

if [[ -s "${old_q}/controller.pid" ]]; then
  old_pid=$(cat "${old_q}/controller.pid")
  kill -TERM "${old_pid}" 2>/dev/null || true
  kill -CONT "${old_pid}" 2>/dev/null || true
fi
touch "${old_q}/CANCELLED_BEFORE_FULL" "${old_q}/DONE_NO_PROMOTION" "${old_q}/DONE"

records() {
  [[ -s "$1/audit/audit_records.jsonl" ]] && { echo "$1/audit/audit_records.jsonl"; return; }
  echo "$1/audit/task_audit_records.jsonl"
}
pair_one() {
  local label=$1 base=$2 method=$3
  local out=${old_q}/paired_final_v2/${label}
  rm -rf "${out}"
  /root/miniconda3/bin/python "${live}/compare_paired_task_runs.py" \
    "$(records "${base}")" "$(records "${method}")" \
    --baseline-config "${base}/run_config.txt" --method-config "${method}/run_config.txt" \
    --method-log "${old_q}/$(basename "${method}").log" --output-dir "${out}"
}
for profile in fast extra; do
  pair_one "he_${profile}_vs_base_n32" "${old_b}/humaneval/he_baseline_128_n32" "${old_b}/humaneval/he_refine_${profile}_128_n32"
  pair_one "he_${profile}_vs_parent_n32" "${old_b}/humaneval/he_parent_128_n32" "${old_b}/humaneval/he_refine_${profile}_128_n32"
  pair_one "math_${profile}_vs_base_n50" "${old_b}/localleap_math500/math_baseline_128_n50" "${old_b}/localleap_math500/math_refine_${profile}_128_n50"
  pair_one "math_${profile}_vs_parent_n50" "${old_b}/localleap_math500/math_parent_128_n50" "${old_b}/localleap_math500/math_refine_${profile}_128_n50"
done
awk -F '\t' 'NF==5' "${old_q}/formal_manifest.tsv" >"${old_q}/formal_manifest_v2.tsv"
printf 'pair_evaluator_v2\tDONE\t2026-07-18T02:15:40+08:00\t2026-07-18T02:15:40+08:00\tcommit_6c0811c\n' >>"${old_q}/formal_manifest_v2.tsv"

install -m 0644 "${stage}/generate.py" "${src}/generate.py"
install -m 0644 "${stage}/eval_llada.py" "${src}/eval_llada.py"
install -m 0644 "${stage}/generate.py" "${live}/generate.py"
install -m 0644 "${stage}/eval_llada.py" "${live}/eval_llada.py"
install -m 0644 "${stage}/tests/test_response_refine.py" "${src}/tests/test_response_refine.py"
install -m 0755 "${stage}/scripts/run_best_symmetric_benchmark.sh" "${src}/scripts/run_best_symmetric_benchmark.sh"
install -m 0755 "${stage}/scripts/run_response_refine_gated_queue.sh" "${src}/scripts/run_response_refine_gated_queue.sh"
install -m 0755 "${stage}/scripts/monitor_response_refine_queue.sh" "${src}/scripts/monitor_response_refine_queue.sh"
install -m 0755 "${stage}/scripts/switch_response_refine_v2.sh" "${src}/scripts/switch_response_refine_v2.sh"

cd "${repo}"
/root/miniconda3/bin/python -m py_compile "${src}/generate.py" "${src}/eval_llada.py"
PYTHONPATH="${src}:${live}" /root/miniconda3/bin/python "${src}/tests/test_response_refine.py"
bash -n "${src}/scripts/run_best_symmetric_benchmark.sh" "${src}/scripts/run_response_refine_gated_queue.sh" "${src}/scripts/monitor_response_refine_queue.sh"
git diff --check
git add experiments/localleap/attention_stability/generate.py \
  experiments/localleap/attention_stability/eval_llada.py \
  experiments/localleap/attention_stability/tests/test_response_refine.py \
  experiments/localleap/attention_stability/scripts/run_best_symmetric_benchmark.sh \
  experiments/localleap/attention_stability/scripts/run_response_refine_gated_queue.sh \
  experiments/localleap/attention_stability/scripts/monitor_response_refine_queue.sh \
  experiments/localleap/attention_stability/scripts/switch_response_refine_v2.sh
git commit -m Add-risk-gated-response-retention

mkdir -p "${new_q}"
nohup env ATTENTION_QUEUE_ID="${new_id}" bash "${src}/scripts/run_response_refine_gated_queue.sh" >"${new_q}/controller_launcher.log" 2>&1 &
echo $! >"${new_q}/controller.pid"
nohup bash "${src}/scripts/monitor_response_refine_queue.sh" "${new_q}" >"${new_q}/monitor_launcher.log" 2>&1 &
echo $! >"${new_q}/monitor.pid"
echo "new_queue_started=$(date --iso-8601=seconds) pid=$(cat "${new_q}/controller.pid")"
