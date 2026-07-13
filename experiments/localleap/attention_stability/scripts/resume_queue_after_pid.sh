#!/bin/bash
set -euo pipefail

watched_pid=${1:?queue pid required}
queue_id=${2:?queue id required}
queue_done=/root/autodl-tmp/LocalLeap/llada/results/experiment_queues/${queue_id}/DONE
queue_script=/root/autodl-tmp/LocalLeap/scripts/llada/run_full_experiment_queue.sh
log=/root/autodl-tmp/LocalLeap/llada/results/experiment_queues/${queue_id}.resume.log

while kill -0 "${watched_pid}" 2>/dev/null; do
  sleep 60
done

if [[ -e "${queue_done}" ]]; then
  echo "queue already completed; no resume needed" >> "${log}"
  exit 0
fi

echo "restarting ${queue_id} after pid ${watched_pid} exited at $(date --iso-8601=seconds)" >> "${log}"
cd /root/autodl-tmp/LocalLeap
exec env QUEUE_ID="${queue_id}" "${queue_script}" >> "${log}" 2>&1
