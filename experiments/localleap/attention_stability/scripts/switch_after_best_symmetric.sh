#!/usr/bin/env bash
set -euo pipefail

old_queue=/root/autodl-tmp/LocalLeap/llada/results/experiment_queues/best_symmetric_long_20260716_v2
new_queue=/root/autodl-tmp/LocalLeap/llada/results/experiment_queues/response_credit_exchange_20260717_v1
repo=/root/autodl-tmp/dlm-seq-flow/experiments/localleap/attention_stability
live=/root/autodl-tmp/LocalLeap/llada
live_scripts=/root/autodl-tmp/LocalLeap/scripts/llada
source_commit=a0c53a7d3e93642cc6aef3a70adddffb21a80e76
switch_log=${old_queue}/response_credit_switcher.log

exec >>"${switch_log}" 2>&1
echo "switcher_start=$(date --iso-8601=seconds)"
echo "source_commit=${source_commit}"

if [[ -e /etc/network_turbo ]]; then
  # Optional proxy requested by the user; all benchmark datasets remain offline-cached.
  source /etc/network_turbo || true
fi

old_pid=$(cat "${old_queue}/controller.pid")
while kill -0 "${old_pid}" 2>/dev/null; do
  state=$(ps -o stat= -p "${old_pid}" 2>/dev/null | tr -d ' ' || true)
  [[ "${state}" == Z* ]] && break
  echo "waiting_old_controller pid=${old_pid} state=${state} at=$(date --iso-8601=seconds)"
  sleep 60
done

# A controller can exit just before its evaluator child. Never deploy into a live generation.
while pgrep -f 'accelerate.*eval_llada.py|python.*eval_llada.py' >/dev/null 2>&1; do
  echo "waiting_evaluator_children at=$(date --iso-8601=seconds)"
  sleep 30
done

echo "old_queue_done=$(test -e "${old_queue}/DONE" && echo true || echo false)"
echo "old_queue_failed=$(find "${old_queue}" -maxdepth 1 -type f -name 'FAILED*' -print -quit | grep -q . && echo true || echo false)"

cd "${repo}"
git cat-file -e "${source_commit}^{commit}"
for path in generate.py eval_llada.py differential_selector.py \
  scripts/run_best_symmetric_benchmark.sh scripts/run_response_credit_exchange_queue.sh \
  tests/test_attention_stability.py tests/test_differential_selector.py \
  tests/test_response_credit_exchange.py; do
  git diff --quiet "${source_commit}" -- "${path}" || {
    echo "source drift relative to pinned commit: ${path}"
    touch "${old_queue}/RESPONSE_CREDIT_SWITCH_FAILED_SOURCE"
    exit 31
  }
done

export PYTHONPATH=${repo}
/root/miniconda3/bin/python -m py_compile \
  generate.py eval_llada.py differential_selector.py \
  tests/test_attention_stability.py tests/test_differential_selector.py \
  tests/test_response_credit_exchange.py
/root/miniconda3/bin/python tests/test_attention_stability.py
/root/miniconda3/bin/python tests/test_differential_selector.py
/root/miniconda3/bin/python tests/test_response_credit_exchange.py
bash -n scripts/run_best_symmetric_benchmark.sh
bash -n scripts/run_response_credit_exchange_queue.sh

mkdir -p "${live_scripts}" "${new_queue}"
install -m 0644 generate.py "${live}/generate.py"
install -m 0644 eval_llada.py "${live}/eval_llada.py"
install -m 0644 differential_selector.py "${live}/differential_selector.py"
install -m 0644 tests/test_attention_stability.py "${live}/test_attention_stability.py"
install -m 0644 tests/test_differential_selector.py "${live}/test_differential_selector.py"
install -m 0644 tests/test_response_credit_exchange.py "${live}/test_response_credit_exchange.py"
install -m 0755 scripts/run_best_symmetric_benchmark.sh "${live_scripts}/run_best_symmetric_benchmark.sh"
install -m 0755 scripts/run_response_credit_exchange_queue.sh "${live_scripts}/run_response_credit_exchange_queue.sh"

cd "${live}"
export PYTHONPATH=${live}
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
/root/miniconda3/bin/python -m py_compile generate.py eval_llada.py differential_selector.py
/root/miniconda3/bin/python test_attention_stability.py
/root/miniconda3/bin/python test_differential_selector.py
/root/miniconda3/bin/python test_response_credit_exchange.py
timeout 5m /root/miniconda3/bin/python test_humaneval_evaluator.py
/root/miniconda3/bin/python test_audit_lm_eval_task.py

sha256sum generate.py eval_llada.py differential_selector.py \
  "${live_scripts}/run_best_symmetric_benchmark.sh" \
  "${live_scripts}/run_response_credit_exchange_queue.sh" \
  > "${new_queue}/DEPLOYED_SOURCE_SHA256"
echo "${source_commit}" > "${new_queue}/SOURCE_COMMIT"

nohup env ATTENTION_QUEUE_ID=response_credit_exchange_20260717_v1 \
  bash "${live_scripts}/run_response_credit_exchange_queue.sh" \
  > "${new_queue}/launcher.log" 2>&1 < /dev/null &
new_pid=$!
echo "${new_pid}" > "${new_queue}/controller.pid"
sleep 5
kill -0 "${new_pid}"
echo "new_controller_pid=${new_pid}"
echo "switcher_finish=$(date --iso-8601=seconds)"
touch "${old_queue}/RESPONSE_CREDIT_SWITCH_DONE"
