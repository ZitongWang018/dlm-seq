#!/usr/bin/env bash
set -euo pipefail

queue_id=stcc_overnight_20260715_v1
queue_root=/root/autodl-tmp/LocalLeap/llada/results/experiment_queues/${queue_id}
initial_mbpp_root=/root/autodl-tmp/LocalLeap/llada/results/b2_confirmatory/mbpp/baseline/stcc_b2_confirmatory_20260715_v1_mbpp_baseline
runner=/root/autodl-tmp/LocalLeap/scripts/llada/run_stcc_experiment.sh
baseline_runner=/root/autodl-tmp/LocalLeap/scripts/llada/run_b2_baseline.sh
mkdir -p "${queue_root}"
exec > >(tee -a "${queue_root}/controller.log") 2>&1

fail() {
  echo "[QUEUE FAILED] $*"
  touch "${queue_root}/FAILED"
  exit 1
}

check_disk() {
  available_kb=$(df --output=avail /root/autodl-tmp | tail -1)
  if (( available_kb < 8 * 1024 * 1024 )); then
    fail "less than 8 GiB free under /root/autodl-tmp"
  fi
}

run_checked() {
  local label=$1
  shift
  check_disk
  if [[ -e "${queue_root}/FROZEN_SOURCE_SHA256" ]]; then
    sha256sum -c "${queue_root}/FROZEN_SOURCE_SHA256" >/dev/null \
      || fail "source drift before ${label}"
  fi
  echo "[START] ${label} $(date --iso-8601=seconds)"
  timeout --kill-after=5m 12h "$@" || fail "${label}"
  echo "[DONE] ${label} $(date --iso-8601=seconds)"
}

echo "queue_id=${queue_id}"
echo "controller_start=$(date --iso-8601=seconds)"
echo "waiting for already-running MBPP b2 baseline"
while [[ ! -e "${initial_mbpp_root}/DONE" && ! -e "${initial_mbpp_root}/FAILED" ]]; do
  sleep 60
done
[[ -e "${initial_mbpp_root}/DONE" ]] || fail "initial MBPP baseline failed"

export PATH=/root/miniconda3/bin:${PATH}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/LocalLeap/llada

# The initial launcher predates the generic record auditor; audit it before any
# method result is allowed to run.
mbpp_samples=$(find "${initial_mbpp_root}/lm_eval" -type f -name 'samples_mbpp_*.jsonl' | sort | tail -1)
mbpp_results=$(find "${initial_mbpp_root}/lm_eval" -type f -name 'results_*.json' | sort | tail -1)
/root/miniconda3/bin/python audit_lm_eval_task.py \
  "${mbpp_samples}" "${mbpp_results}" \
  --task mbpp --primary-metric pass_at_1 --expected-records 500 \
  --output-dir "${initial_mbpp_root}/audit" || fail "initial MBPP evaluator audit"

/root/miniconda3/bin/python -m py_compile \
  stcc_generate.py eval_llada.py validate_stcc_trace.py \
  validate_stcc_diagnostics.py audit_lm_eval_task.py
/root/miniconda3/bin/python - <<'PY'
import test_candidate_memory as old
import test_stcc_generate as new
for module in (old, new):
    for name, fn in sorted(vars(module).items()):
        if name.startswith("test_") and callable(fn):
            fn()
print("direct_regression_tests=16")
PY

sha256sum generate.py stcc_generate.py eval_llada.py model/modeling_llada.py \
  validate_stcc_trace.py validate_stcc_diagnostics.py audit_lm_eval_task.py \
  "${runner}" "${baseline_runner}" > "${queue_root}/FROZEN_SOURCE_SHA256"
cp -f stcc_generate.py eval_llada.py validate_stcc_trace.py \
  validate_stcc_diagnostics.py audit_lm_eval_task.py "${queue_root}/"

run_checked real_model_smoke \
  "${runner}" humaneval 0 256 none 0.01 0.004 1 1 full he_stcc_real_smoke1 1

# HumanEval is explicitly exploratory: test a compact response threshold family
# and one two-step streak without adding confidence gates or frontier heuristics.
run_checked he_eps0p005_s1 \
  "${runner}" humaneval 0 256 none 0.005 0.004 1 1 trace he_vertical_eps0p005_s1
run_checked he_eps0p01_s1 \
  "${runner}" humaneval 0 256 none 0.01 0.004 1 1 full he_vertical_eps0p01_s1
run_checked he_eps0p02_s1 \
  "${runner}" humaneval 0 256 none 0.02 0.004 1 1 trace he_vertical_eps0p02_s1
run_checked he_eps0p01_s2 \
  "${runner}" humaneval 0 256 none 0.01 0.004 1 2 trace he_vertical_eps0p01_s2

/root/miniconda3/bin/python select_stcc_humaneval_winner.py \
  --results-root /root/autodl-tmp/LocalLeap/llada/results/stcc/humaneval \
  --output-json "${queue_root}/humaneval_winner.json" \
  --output-env "${queue_root}/humaneval_winner.env"
source "${queue_root}/humaneval_winner.env"

# Speed/performance arms use the selected longitudinal response rule.  They
# never reduce the baseline commit count; only verified low-response positions
# can be added.
run_checked he_accel2 \
  "${runner}" humaneval 0 256 none "${WINNER_EPS}" 0.004 2 "${WINNER_STREAK}" trace he_accel2
run_checked he_accel4 \
  "${runner}" humaneval 0 256 none "${WINNER_EPS}" 0.004 4 "${WINNER_STREAK}" trace he_accel4

# MBPP is the first full b=2 development benchmark.  Baseline is the completed
# initial run above; all quality arms preserve exactly 128 NFE per task.
run_checked mbpp_vertical \
  "${runner}" mbpp 3 128 none "${WINNER_EPS}" 0.004 1 "${WINNER_STREAK}" trace mbpp_vertical_b2
run_checked mbpp_symmetric \
  "${runner}" mbpp 3 128 symmetric "${WINNER_EPS}" 0.004 1 "${WINNER_STREAK}" trace mbpp_symmetric_b2
run_checked mbpp_directed \
  "${runner}" mbpp 3 128 directed "${WINNER_EPS}" 0.004 1 "${WINNER_STREAK}" full mbpp_directed_b2
run_checked mbpp_accel2 \
  "${runner}" mbpp 3 128 directed "${WINNER_EPS}" 0.004 2 "${WINNER_STREAK}" trace mbpp_directed_accel2
run_checked mbpp_accel4 \
  "${runner}" mbpp 3 128 directed "${WINNER_EPS}" 0.004 4 "${WINNER_STREAK}" trace mbpp_directed_accel4

# Minerva counting/probability is kept untouched until the HumanEval family and
# all MBPP arms finish.  It receives the same frozen method family.
run_checked minerva_baseline \
  "${baseline_runner}" minerva_math_counting_and_prob 4 minerva_counting_prob_baseline_b2
run_checked minerva_vertical \
  "${runner}" minerva_math_counting_and_prob 4 128 none "${WINNER_EPS}" 0.004 1 "${WINNER_STREAK}" trace minerva_vertical_b2
run_checked minerva_symmetric \
  "${runner}" minerva_math_counting_and_prob 4 128 symmetric "${WINNER_EPS}" 0.004 1 "${WINNER_STREAK}" trace minerva_symmetric_b2
run_checked minerva_directed \
  "${runner}" minerva_math_counting_and_prob 4 128 directed "${WINNER_EPS}" 0.004 1 "${WINNER_STREAK}" full minerva_directed_b2
run_checked minerva_accel2 \
  "${runner}" minerva_math_counting_and_prob 4 128 directed "${WINNER_EPS}" 0.004 2 "${WINNER_STREAK}" trace minerva_directed_accel2
run_checked minerva_accel4 \
  "${runner}" minerva_math_counting_and_prob 4 128 directed "${WINNER_EPS}" 0.004 4 "${WINNER_STREAK}" trace minerva_directed_accel4

touch "${queue_root}/DONE"
echo "controller_finish=$(date --iso-8601=seconds)"
