#!/usr/bin/env bash
set -euo pipefail

old_controller_pid=${1:?old controller pid required}
active_stage_pid=${2:?active stage pid required}
old_queue=${3:?old queue root required}
canonical=/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability
localleap=/root/autodl-tmp/LocalLeap
llada=${localleap}/llada
new_queue_id=${ATTENTION_QUEUE_ID:-attention_retention_v2_20260716}
new_queue=${llada}/results/experiment_queues/${new_queue_id}
switch_log=${old_queue}/switch_to_${new_queue_id}.log

exec >> "${switch_log}" 2>&1
echo "switch_wait_start=$(date --iso-8601=seconds)"
echo "old_controller_pid=${old_controller_pid} active_stage_pid=${active_stage_pid}"

while kill -0 "${active_stage_pid}" 2>/dev/null; do
  sleep 5
done
echo "active_stage_finished=$(date --iso-8601=seconds)"

# The old controller was SIGSTOPed while its active child completed, so it
# cannot advance into the superseded 64-step method branches.
for child in $(pgrep -P "${old_controller_pid}" 2>/dev/null || true); do
  kill -TERM "${child}" 2>/dev/null || true
done
kill -TERM "${old_controller_pid}" 2>/dev/null || true
kill -CONT "${old_controller_pid}" 2>/dev/null || true
sleep 2
kill -KILL "${old_controller_pid}" 2>/dev/null || true
touch "${old_queue}/SUPERSEDED_BY_${new_queue_id}"

cp "${canonical}/generate.py" "${llada}/generate.py"
cp "${canonical}/eval_llada.py" "${llada}/eval_llada.py"
cp "${canonical}/validate_step_diagnostics.py" "${llada}/validate_step_diagnostics.py"
cp "${canonical}/audit_lm_eval_task.py" "${llada}/audit_lm_eval_task.py"
cp "${canonical}/tests/test_attention_stability.py" "${llada}/test_attention_stability.py"
cp "${canonical}/tests/test_validate_step_diagnostics.py" "${llada}/test_validate_step_diagnostics.py"
cp "${canonical}/tests/test_audit_lm_eval_task.py" "${llada}/test_audit_lm_eval_task.py"
cp "${canonical}/scripts/run_attention_benchmark_v2.sh" "${localleap}/scripts/llada/run_attention_benchmark_v2.sh"
cp "${canonical}/scripts/run_attention_retention_v2_queue.sh" "${localleap}/scripts/llada/run_attention_retention_v2_queue.sh"
chmod +x "${localleap}/scripts/llada/run_attention_benchmark_v2.sh" \
  "${localleap}/scripts/llada/run_attention_retention_v2_queue.sh"

cd "${llada}"
/root/miniconda3/bin/python -m py_compile generate.py eval_llada.py \
  validate_step_diagnostics.py audit_lm_eval_task.py test_attention_stability.py \
  test_validate_step_diagnostics.py test_audit_lm_eval_task.py
/root/miniconda3/bin/python test_attention_stability.py
/root/miniconda3/bin/python test_validate_step_diagnostics.py
/root/miniconda3/bin/python test_audit_lm_eval_task.py

mkdir -p "${new_queue}"
ATTENTION_QUEUE_ID="${new_queue_id}" nohup \
  "${localleap}/scripts/llada/run_attention_retention_v2_queue.sh" \
  > "${new_queue}/launcher.log" 2>&1 < /dev/null &
new_pid=$!
echo "${new_pid}" > "${new_queue}/controller.pid"
echo "new_queue_started=$(date --iso-8601=seconds) pid=${new_pid}"
