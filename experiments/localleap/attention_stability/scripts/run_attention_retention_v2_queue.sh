#!/usr/bin/env bash
set -uo pipefail

queue_id=${ATTENTION_QUEUE_ID:-attention_retention_v2_20260716}
llada_root=/root/autodl-tmp/LocalLeap/llada
runner=/root/autodl-tmp/LocalLeap/scripts/llada/run_attention_benchmark_v2.sh
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
echo "baseline=original_llada"
echo "parent=attention_stability_symmetric_tau_0.004"
echo "new_temporal_rule=topk_overlap_lexicographic_k4"

/root/miniconda3/bin/python -m py_compile generate.py eval_llada.py \
  validate_step_diagnostics.py audit_lm_eval_task.py test_attention_stability.py \
  test_validate_step_diagnostics.py test_audit_lm_eval_task.py
/root/miniconda3/bin/python test_attention_stability.py
/root/miniconda3/bin/python test_validate_step_diagnostics.py
/root/miniconda3/bin/python test_audit_lm_eval_task.py

sha256sum generate.py eval_llada.py model/modeling_llada.py \
  validate_step_diagnostics.py audit_attention_stability.py audit_lm_eval_task.py \
  analyze_paired_humaneval.py postprocess_code.py sanitize.py \
  code_eval/code_eval.py code_eval/execute.py "${runner}" "$0" \
  > "${queue_root}/FROZEN_SOURCE_SHA256"

# Functional checks only; K=4 is fixed a priori and is not selected on benchmark accuracy.
run_stage smoke_he_retention "${runner}" humaneval 0 256 symmetric_retention 0.004 full retention_v2_smoke_he 1
run_stage smoke_gsm8k_auditor "${runner}" gsm8k 5 128 baseline 0.004 trace retention_v2_smoke_gsm8k 1

# HumanEval: compare the new vertical rule with the registered original-LLaDA and best-parent runs.
run_stage he_retention_256 "${runner}" humaneval 0 256 symmetric_retention 0.004 full retention_v2_he_256
run_stage he_retention_fast_128 "${runner}" humaneval 0 128 symmetric_retention_fast 0.004 full retention_v2_he_fast_128

# MBPP is the development benchmark for horizontal parallel decoding.
# Existing formal anchors: original LLaDA 128 = 17.8%, parent symmetric = 24.2%.
run_stage mbpp_retention_128 "${runner}" mbpp 3 128 symmetric_retention 0.004 full retention_v2_mbpp_128
run_stage mbpp_retention_fast_128 "${runner}" mbpp 3 128 symmetric_retention_fast 0.004 full retention_v2_mbpp_fast_128

# GSM8K is an untouched cross-domain generalization check with aligned prompts/shots/seeds.
run_stage gsm8k_original_llada_128 "${runner}" gsm8k 5 128 baseline 0.004 trace retention_v2_gsm8k_baseline_128
run_stage gsm8k_parent_symmetric_128 "${runner}" gsm8k 5 128 symmetric 0.004 trace retention_v2_gsm8k_parent_128
run_stage gsm8k_retention_128 "${runner}" gsm8k 5 128 symmetric_retention 0.004 trace retention_v2_gsm8k_128
run_stage gsm8k_retention_fast_128 "${runner}" gsm8k 5 128 symmetric_retention_fast 0.004 trace retention_v2_gsm8k_fast_128

echo "controller_finish=$(date --iso-8601=seconds)"
touch "${queue_root}/DONE"
