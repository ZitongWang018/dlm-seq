#!/usr/bin/env bash
set -uo pipefail

queue_id=${ATTENTION_QUEUE_ID:-attention_recovery_long_20260715_v1}
llada_root=/root/autodl-tmp/LocalLeap/llada
runner=/root/autodl-tmp/LocalLeap/scripts/llada/run_attention_benchmark.sh
queue_root=${llada_root}/results/experiment_queues/${queue_id}
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

run_stage() {
  local label=$1
  shift
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

export PATH=/root/miniconda3/bin:${PATH}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd "${llada_root}"

if [[ ! -e "${manifest}" ]]; then
  printf 'stage\tstatus\tstart\tfinish\texit_code\n' > "${manifest}"
fi

echo "queue_id=${queue_id}"
echo "controller_start=$(date --iso-8601=seconds)"
echo "anchor=attention_stability_v1 tau=0.004"

/root/miniconda3/bin/python -m py_compile generate.py eval_llada.py validate_step_diagnostics.py test_attention_stability.py
/root/miniconda3/bin/python test_attention_stability.py

sha256sum generate.py eval_llada.py model/modeling_llada.py validate_step_diagnostics.py \
  audit_attention_stability.py audit_lm_eval_task.py analyze_paired_humaneval.py \
  postprocess_code.py sanitize.py code_eval/code_eval.py code_eval/execute.py \
  "${runner}" "$0" > "${queue_root}/FROZEN_SOURCE_SHA256"

# Fast real-model and evaluator preflights.
run_stage smoke_he_anchor "${runner}" humaneval 0 256 symmetric 0.004 full recovery_smoke_he_anchor 1
run_stage smoke_mbpp_fast "${runner}" mbpp 3 64 symmetric_fast 0.004 trace recovery_smoke_mbpp_fast 1
run_stage smoke_minerva_baseline "${runner}" minerva_math_counting_and_prob 4 128 baseline 0.004 trace recovery_smoke_minerva_baseline 1
run_stage smoke_gsm8k_baseline "${runner}" gsm8k 5 128 baseline 0.004 trace recovery_smoke_gsm8k_baseline 1

# HumanEval: exact old best first, then asymmetry and fixed-budget speed arms.
run_stage he_anchor_symmetric_256 "${runner}" humaneval 0 256 symmetric 0.004 full recovery_he_anchor_symmetric_256
run_stage he_directed_256 "${runner}" humaneval 0 256 directed 0.004 trace recovery_he_directed_256
run_stage he_baseline_128 "${runner}" humaneval 0 128 baseline 0.004 trace recovery_he_baseline_128
run_stage he_symmetric_fast_128 "${runner}" humaneval 0 128 symmetric_fast 0.004 trace recovery_he_symmetric_fast_128
run_stage he_directed_fast_128 "${runner}" humaneval 0 128 directed_fast 0.004 trace recovery_he_directed_fast_128
run_stage he_baseline_64 "${runner}" humaneval 0 64 baseline 0.004 trace recovery_he_baseline_64
run_stage he_symmetric_fast_64 "${runner}" humaneval 0 64 symmetric_fast 0.004 trace recovery_he_symmetric_fast_64
run_stage he_directed_fast_64 "${runner}" humaneval 0 64 directed_fast 0.004 trace recovery_he_directed_fast_64

# MBPP: the completed 128-step baseline (17.8%) remains the fair anchor.
run_stage mbpp_symmetric_128 "${runner}" mbpp 3 128 symmetric 0.004 full recovery_mbpp_symmetric_128
run_stage mbpp_directed_128 "${runner}" mbpp 3 128 directed 0.004 trace recovery_mbpp_directed_128
run_stage mbpp_baseline_64 "${runner}" mbpp 3 64 baseline 0.004 trace recovery_mbpp_baseline_64
run_stage mbpp_symmetric_fast_64 "${runner}" mbpp 3 64 symmetric_fast 0.004 trace recovery_mbpp_symmetric_fast_64
run_stage mbpp_directed_fast_64 "${runner}" mbpp 3 64 directed_fast 0.004 trace recovery_mbpp_directed_fast_64

# Mathematical generalization with identical prompts, shots, lengths and seeds per paired arm.
run_stage minerva_baseline_128 "${runner}" minerva_math_counting_and_prob 4 128 baseline 0.004 trace recovery_minerva_baseline_128
run_stage minerva_symmetric_128 "${runner}" minerva_math_counting_and_prob 4 128 symmetric 0.004 full recovery_minerva_symmetric_128
run_stage minerva_directed_128 "${runner}" minerva_math_counting_and_prob 4 128 directed 0.004 trace recovery_minerva_directed_128
run_stage minerva_baseline_64 "${runner}" minerva_math_counting_and_prob 4 64 baseline 0.004 trace recovery_minerva_baseline_64
run_stage minerva_symmetric_fast_64 "${runner}" minerva_math_counting_and_prob 4 64 symmetric_fast 0.004 trace recovery_minerva_symmetric_fast_64
run_stage minerva_directed_fast_64 "${runner}" minerva_math_counting_and_prob 4 64 directed_fast 0.004 trace recovery_minerva_directed_fast_64

# Long tail: GSM8K keeps the GPU occupied and tests broad arithmetic generalization.
run_stage gsm8k_baseline_128 "${runner}" gsm8k 5 128 baseline 0.004 trace recovery_gsm8k_baseline_128
run_stage gsm8k_symmetric_128 "${runner}" gsm8k 5 128 symmetric 0.004 trace recovery_gsm8k_symmetric_128
run_stage gsm8k_directed_128 "${runner}" gsm8k 5 128 directed 0.004 trace recovery_gsm8k_directed_128
run_stage gsm8k_baseline_64 "${runner}" gsm8k 5 64 baseline 0.004 trace recovery_gsm8k_baseline_64
run_stage gsm8k_symmetric_fast_64 "${runner}" gsm8k 5 64 symmetric_fast 0.004 trace recovery_gsm8k_symmetric_fast_64
run_stage gsm8k_directed_fast_64 "${runner}" gsm8k 5 64 directed_fast 0.004 trace recovery_gsm8k_directed_fast_64

echo "controller_finish=$(date --iso-8601=seconds)"
touch "${queue_root}/DONE"
