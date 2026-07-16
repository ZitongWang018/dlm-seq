#!/usr/bin/env bash
set -euo pipefail

queue=/root/autodl-tmp/LocalLeap/llada/results/experiment_queues/best_symmetric_long_20260716_v2
run_base=/root/autodl-tmp/LocalLeap/llada/results/best_symmetric_benchmarks/best_symmetric_long_20260716_v2/gsm8k
llada=/root/autodl-tmp/LocalLeap/llada
log=${queue}/gsm8k_flexible_reaudit.log
exec >>"${log}" 2>&1

controller=$(cat "${queue}/controller.pid")

wait_for_run() {
  local tag=$1
  local root=${run_base}/${tag}
  while [[ ! -e "${root}/DONE" ]]; do
    if ! kill -0 "${controller}" 2>/dev/null; then
      echo "controller exited before ${tag} completed"
      return 1
    fi
    sleep 60
  done
}

audit_run() {
  local tag=$1 expected=$2 mode=$3
  local root=${run_base}/${tag}
  local out=${root}/audit_flexible_v3
  if [[ -e "${out}/DONE" ]]; then
    echo "[SKIP-DONE] ${tag}"
    return 0
  fi
  local samples results
  samples=$(find "${root}/lm_eval" -type f -name 'samples_gsm8k_*.jsonl' | sort | tail -1)
  results=$(find "${root}/lm_eval" -type f -name 'results_*.json' | sort | tail -1)
  [[ -s "${samples}" && -s "${results}" ]]
  mkdir -p "${out}"
  local nfe_args
  if [[ "${mode}" == baseline ]]; then
    nfe_args=(--constant-nfe 128)
  else
    nfe_args=(--trace "${root}/trace/rank_0.jsonl")
  fi
  cd "${llada}"
  /root/miniconda3/bin/python audit_lm_eval_task.py "${samples}" "${results}" \
    --task gsm8k --primary-metric 'exact_match,flexible-extract' \
    "${nfe_args[@]}" --filter flexible-extract --expected-records "${expected}" \
    --output-dir "${out}"
  /root/miniconda3/bin/python - "${samples}" "${out}/task_audit_summary.json" <<'PY'
import json, sys
samples = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
selected = [row for row in samples if row.get("filter") == "flexible-extract"]
summary = json.load(open(sys.argv[2], encoding="utf-8"))
correct = sum(float(row["exact_match"]) == 1.0 for row in selected)
assert len(selected) == summary["total"]
assert correct == summary["correct"]
assert summary["prompt_hash_mismatches"] == 0
assert summary["duplicate_ids"] == 0
print({"second_path_correct": correct, "total": len(selected)})
PY
  touch "${out}/DONE"
}

pair_runs() {
  local label=$1 baseline_tag=$2 method_tag=$3
  local baseline=${run_base}/${baseline_tag}
  local method=${run_base}/${method_tag}
  local out=${queue}/paired_flexible_v3/${label}
  [[ -e "${out}/DONE" ]] && return 0
  mkdir -p "${out}"
  cd "${llada}"
  /root/miniconda3/bin/python compare_paired_task_runs.py \
    "${baseline}/audit_flexible_v3/task_audit_records.jsonl" \
    "${method}/audit_flexible_v3/task_audit_records.jsonl" \
    --baseline-config "${baseline}/run_config.txt" \
    --method-config "${method}/run_config.txt" \
    --method-log "${queue}/${method_tag}.log" \
    --output-dir "${out}"
  touch "${out}/DONE"
}

echo "start=$(date --iso-8601=seconds)"
cd "${llada}"
sha256sum audit_lm_eval_task.py compare_paired_task_runs.py > "${queue}/gsm8k_flexible_reaudit_source.sha256"

audit_run gsm8k_baseline_0shot_128 1319 baseline
wait_for_run gsm8k_symmetric_fast_0shot_128
audit_run gsm8k_symmetric_fast_0shot_128 1319 method
pair_runs pair_gsm8k_0shot_fast gsm8k_baseline_0shot_128 gsm8k_symmetric_fast_0shot_128

wait_for_run gsm8k_symmetric_0shot_128
audit_run gsm8k_symmetric_0shot_128 1319 method
pair_runs pair_gsm8k_0shot_accuracy gsm8k_baseline_0shot_128 gsm8k_symmetric_0shot_128

wait_for_run gsm8k_baseline_4shot_128_n500
audit_run gsm8k_baseline_4shot_128_n500 500 baseline
wait_for_run gsm8k_symmetric_fast_4shot_128_n500
audit_run gsm8k_symmetric_fast_4shot_128_n500 500 method
pair_runs pair_gsm8k_4shot_fast_n500 gsm8k_baseline_4shot_128_n500 gsm8k_symmetric_fast_4shot_128_n500

echo "finish=$(date --iso-8601=seconds)"
touch "${queue}/GSM8K_FLEXIBLE_REAUDIT_DONE"
