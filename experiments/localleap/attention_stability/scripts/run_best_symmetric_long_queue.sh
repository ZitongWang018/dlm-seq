#!/usr/bin/env bash
set -uo pipefail

queue_id=${ATTENTION_QUEUE_ID:-best_symmetric_long_20260716_v1}
llada_root=/root/autodl-tmp/LocalLeap/llada
runner=/root/autodl-tmp/LocalLeap/scripts/llada/run_best_symmetric_benchmark.sh
queue_root=${llada_root}/results/experiment_queues/${queue_id}
run_base=${llada_root}/results/best_symmetric_benchmarks
manifest=${queue_root}/manifest.tsv
mkdir -p "${queue_root}"
exec > >(tee -a "${queue_root}/controller.log") 2>&1

check_disk() {
  local free_kb
  free_kb=$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')
  if (( free_kb < 8 * 1024 * 1024 )); then
    echo "[QUEUE HARD STOP] disk below 8 GiB"
    touch "${queue_root}/FAILED_DISK"
    exit 20
  fi
}

verify_source() {
  cd "${llada_root}"
  sha256sum -c "${queue_root}/FROZEN_SOURCE_SHA256" >/dev/null || {
    echo "[QUEUE HARD STOP] source hash drift"
    touch "${queue_root}/FAILED_SOURCE_DRIFT"
    exit 21
  }
}

stage_already_done() {
  local label=$1
  awk -F '\t' -v label="${label}" '$1 == label && $2 == "DONE" {found=1} END {exit !found}' "${manifest}"
}

run_stage() {
  local label=$1
  shift
  if stage_already_done "${label}"; then
    echo "[SKIP-DONE] ${label}"
    return 0
  fi
  check_disk
  verify_source
  local started finished rc
  started=$(date --iso-8601=seconds)
  printf '%s\tSTARTED\t%s\t\t\n' "${label}" "${started}" >> "${manifest}"
  echo "[START] ${label} ${started}"
  set +e
  timeout --kill-after=5m 24h "$@"
  rc=$?
  set -e
  finished=$(date --iso-8601=seconds)
  if [[ ${rc} -eq 0 ]]; then
    printf '%s\tDONE\t%s\t%s\t0\n' "${label}" "${started}" "${finished}" >> "${manifest}"
    echo "[DONE] ${label} ${finished}"
  else
    printf '%s\tFAILED\t%s\t%s\t%s\n' "${label}" "${started}" "${finished}" "${rc}" >> "${manifest}"
    echo "[FAILED-CONTINUE] ${label} rc=${rc} ${finished}"
  fi
}

pair_stage() {
  local label=$1
  local task=$2
  local baseline_tag=$3
  local method_tag=$4
  local baseline_root=${5:-${run_base}/${task}/${baseline_tag}}
  local method_root=${run_base}/${task}/${method_tag}
  local output=${queue_root}/paired/${label}
  if [[ ! -e "${baseline_root}/DONE" || ! -e "${method_root}/DONE" ]]; then
    echo "[PAIR-SKIP] ${label}: a parent run is incomplete"
    return 0
  fi
  run_stage "${label}" /root/miniconda3/bin/python compare_paired_task_runs.py \
    "${baseline_root}/audit/task_audit_records.jsonl" \
    "${method_root}/audit/task_audit_records.jsonl" \
    --baseline-config "${baseline_root}/run_config.txt" \
    --method-config "${method_root}/run_config.txt" \
    --method-log "${queue_root}/${method_tag}.log" \
    --output-dir "${output}"
}

export PATH=/root/miniconda3/bin:${PATH}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
cd "${llada_root}"

if [[ ! -e "${manifest}" ]]; then
  printf 'stage\tstatus\tstart\tfinish\texit_code\n' > "${manifest}"
fi

echo "queue_id=${queue_id}"
echo "controller_start=$(date --iso-8601=seconds)"
echo "baseline=original_llada_low_confidence"
echo "method_parent=symmetric_attention_tau_0.004"
echo "fixed_budget_child=symmetric_fast_tau_0.004"

/root/miniconda3/bin/python -m py_compile \
  generate.py eval_llada.py audit_lm_eval_task.py compare_paired_task_runs.py \
  tasks/localleap_math500/utils.py
PYTHONPATH=. /root/miniconda3/bin/python test_attention_stability.py
PYTHONPATH=. /root/miniconda3/bin/python test_audit_lm_eval_task.py
PYTHONPATH=. /root/miniconda3/bin/python -m unittest discover -s tests -p 'test_*.py'

/root/miniconda3/bin/python - <<'PY'
from datasets import load_dataset
assert len(load_dataset("HuggingFaceH4/MATH-500", split="test")) == 500
assert len(load_dataset("gsm8k", "main", split="test")) == 1319
print("offline datasets ready")
PY

sha256sum generate.py eval_llada.py model/modeling_llada.py validate_step_diagnostics.py \
  audit_attention_stability.py audit_lm_eval_task.py compare_paired_task_runs.py \
  postprocess_code.py sanitize.py code_eval/code_eval.py code_eval/execute.py \
  tasks/localleap_math500/math500.yaml \
  tasks/localleap_math500/utils.py "${runner}" "$0" \
  > "${queue_root}/FROZEN_SOURCE_SHA256"

# Real-model and evaluator preflights. Full diagnostics are limited to a few records.
run_stage smoke_math500_baseline "${runner}" localleap_math500 0 128 baseline 0.004 trace smoke_math500_baseline 2 256
run_stage smoke_math500_method "${runner}" localleap_math500 0 128 symmetric_fast 0.004 full smoke_math500_method 2 256
run_stage smoke_gsm8k_baseline "${runner}" gsm8k 0 128 baseline 0.004 trace smoke_gsm8k_baseline 2 256
run_stage smoke_gsm8k_method "${runner}" gsm8k 0 128 symmetric_fast 0.004 full smoke_gsm8k_method 2 256

# OTS-aligned MATH-500: LLaDA-Instruct, zero-shot, length 256, 128 steps, block 32.
run_stage math500_baseline_128 "${runner}" localleap_math500 0 128 baseline 0.004 trace math500_baseline_128 full 256
run_stage math500_symmetric_fast_128 "${runner}" localleap_math500 0 128 symmetric_fast 0.004 trace math500_symmetric_fast_128 full 256
pair_stage pair_math500_fast localleap_math500 math500_baseline_128 math500_symmetric_fast_128

# Accuracy-first parent. This arm may use more NFE; the audit reports the exact ratio.
run_stage math500_symmetric_128 "${runner}" localleap_math500 0 128 symmetric 0.004 trace math500_symmetric_128 full 256
pair_stage pair_math500_accuracy localleap_math500 math500_baseline_128 math500_symmetric_128

# SOAR-aligned prompting for MBPP. Reuse the formally audited original-LLaDA baseline.
run_stage mbpp_symmetric_fast_128 "${runner}" mbpp 3 128 symmetric_fast 0.004 trace mbpp_symmetric_fast_128 full 256
mbpp_baseline=/root/autodl-tmp/LocalLeap/llada/results/b2_confirmatory/mbpp/baseline/stcc_b2_confirmatory_20260715_v1_mbpp_baseline
pair_stage pair_mbpp_fast mbpp unused mbpp_symmetric_fast_128 "${mbpp_baseline}"

# Prism/OTS-aligned zero-shot GSM8K at length 256 and 128 steps.
run_stage gsm8k_baseline_0shot_128 "${runner}" gsm8k 0 128 baseline 0.004 trace gsm8k_baseline_0shot_128 full 256
run_stage gsm8k_symmetric_fast_0shot_128 "${runner}" gsm8k 0 128 symmetric_fast 0.004 trace gsm8k_symmetric_fast_0shot_128 full 256
pair_stage pair_gsm8k_0shot_fast gsm8k gsm8k_baseline_0shot_128 gsm8k_symmetric_fast_0shot_128
run_stage gsm8k_symmetric_0shot_128 "${runner}" gsm8k 0 128 symmetric 0.004 trace gsm8k_symmetric_0shot_128 full 256
pair_stage pair_gsm8k_0shot_accuracy gsm8k gsm8k_baseline_0shot_128 gsm8k_symmetric_0shot_128

# SOAR uses four-shot GSM8K. Run a 500-example paired subset because its backbone is Base, not Instruct.
run_stage gsm8k_baseline_4shot_128_n500 "${runner}" gsm8k 4 128 baseline 0.004 trace gsm8k_baseline_4shot_128_n500 500 256
run_stage gsm8k_symmetric_fast_4shot_128_n500 "${runner}" gsm8k 4 128 symmetric_fast 0.004 trace gsm8k_symmetric_fast_4shot_128_n500 500 256
pair_stage pair_gsm8k_4shot_fast_n500 gsm8k gsm8k_baseline_4shot_128_n500 gsm8k_symmetric_fast_4shot_128_n500

# Long-length robustness, matching the 512-token column used by recent work.
run_stage math500_baseline_len512_steps256_n250 "${runner}" localleap_math500 0 256 baseline 0.004 trace math500_baseline_len512_steps256_n250 250 512
run_stage math500_symmetric_fast_len512_steps256_n250 "${runner}" localleap_math500 0 256 symmetric_fast 0.004 trace math500_symmetric_fast_len512_steps256_n250 250 512
pair_stage pair_math500_len512_fast_n250 localleap_math500 math500_baseline_len512_steps256_n250 math500_symmetric_fast_len512_steps256_n250

echo "controller_finish=$(date --iso-8601=seconds)"
touch "${queue_root}/DONE"
