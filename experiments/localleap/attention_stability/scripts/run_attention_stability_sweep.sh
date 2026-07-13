#!/bin/bash
set -euo pipefail

runner=/root/autodl-tmp/LocalLeap/scripts/llada/run_attention_stability_humaneval.sh
llada_root=/root/autodl-tmp/LocalLeap/llada

if [[ "$#" -gt 0 ]]; then
  taus=("$@")
else
  taus=(0.005 0.02 0.05)
fi

sweep_id=${SWEEP_ID:-attention_tau_sweep_$(date +%Y%m%d_%H%M%S)}
sweep_root=${llada_root}/results/attention_stability/sweeps/${sweep_id}
mkdir -p "${sweep_root}"
manifest=${sweep_root}/manifest.tsv
printf 'tau\tstatus\tstart\tfinish\trun_tag\texit_code\n' > "${manifest}"

echo "sweep_id=${sweep_id}"
echo "taus=${taus[*]}"
echo "start=$(date --iso-8601=seconds)"

for tau in "${taus[@]}"; do
  tau_tag=${tau//./p}
  run_tag=${sweep_id}_tau${tau_tag}
  start_time=$(date --iso-8601=seconds)
  printf '%s\tSTARTED\t%s\t\t%s\t\n' "${tau}" "${start_time}" "${run_tag}" >> "${manifest}"
  echo "[SWEEP] starting tau=${tau} run_tag=${run_tag} at ${start_time}"
  if "${runner}" "${tau}" full "${run_tag}"; then
    finish_time=$(date --iso-8601=seconds)
    printf '%s\tCOMPLETED\t%s\t%s\t%s\t0\n' "${tau}" "${start_time}" "${finish_time}" "${run_tag}" >> "${manifest}"
    echo "[SWEEP] completed tau=${tau} at ${finish_time}"
  else
    exit_code=$?
    finish_time=$(date --iso-8601=seconds)
    printf '%s\tFAILED\t%s\t%s\t%s\t%s\n' "${tau}" "${start_time}" "${finish_time}" "${run_tag}" "${exit_code}" >> "${manifest}"
    echo "[SWEEP] failed tau=${tau} exit_code=${exit_code} at ${finish_time}" >&2
    exit "${exit_code}"
  fi
done

echo "finish=$(date --iso-8601=seconds)"
touch "${sweep_root}/DONE"
