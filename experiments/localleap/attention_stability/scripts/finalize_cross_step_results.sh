#!/bin/bash
set -euo pipefail

export PATH=/root/miniconda3/bin:$PATH
llada=/root/autodl-tmp/LocalLeap/llada
python_bin=/root/miniconda3/bin/python
queue_id=${1:-cross_step_full_queue_20260714_v1}
queue_root=${llada}/results/experiment_queues/${queue_id}
baseline=${llada}/results/baseline/humaneval_len256_blen32_0shot/__root__autodl-tmp__model__LLaDA__instruct/samples_humaneval_2026-07-13T15-29-41.470324.jsonl

test -e "${queue_root}/DONE"
cd "${llada}"

roots=(
  "results/attention_stability/tau0.004/${queue_id}_tau0p004"
  "results/attention_stability/tau0.0025/${queue_id}_tau0p0025"
  "results/attention_stability/tau0.001/${queue_id}_tau0p001"
  "results/attention_stability/tau0.0005/${queue_id}_tau0p0005"
  "results/candidate_memory/confidence/k8_delta0/${queue_id}_candidate_confidence_k8_delta0"
  "results/candidate_memory/frontier/k8_delta0/${queue_id}_candidate_frontier_k8_delta0"
)

for root in "${roots[@]}"; do
  test -e "${root}/DONE"
  sample=$(find "${root}" -type f -name 'samples_humaneval_*.jsonl' | sort | tail -1)
  test -s "${sample}"
  test -s "${sample}.cleaned"
  "${python_bin}" analyze_paired_humaneval.py \
    "${baseline}" "${sample}" "${root}/trace/rank_0.jsonl" \
    --output-dir "${root}/paired"
done

confidence_root=${roots[4]}
frontier_root=${roots[5]}
confidence_sample=$(find "${confidence_root}" -type f -name 'samples_humaneval_*.jsonl' | sort | tail -1)
frontier_sample=$(find "${frontier_root}" -type f -name 'samples_humaneval_*.jsonl' | sort | tail -1)
"${python_bin}" analyze_paired_humaneval.py \
  "${confidence_sample}" "${frontier_sample}" "${frontier_root}/trace/rank_0.jsonl" \
  --output-dir "${frontier_root}/paired_vs_candidate_stability"

"${python_bin}" finalize_cross_step_results.py \
  --llada-root "${llada}" \
  --queue-id "${queue_id}" \
  --output "${queue_root}/final_summary.json"
touch "${queue_root}/FINALIZED"
