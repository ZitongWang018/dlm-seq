#!/usr/bin/env bash
set -uo pipefail

queue_id=${ATTENTION_QUEUE_ID:-response_credit_exchange_20260717_v1}
llada_root=/root/autodl-tmp/LocalLeap/llada
runner=/root/autodl-tmp/LocalLeap/scripts/llada/run_best_symmetric_benchmark.sh
queue_root=${llada_root}/results/experiment_queues/${queue_id}
run_base=${llada_root}/results/best_symmetric_benchmarks/${queue_id}
manifest=${queue_root}/manifest.tsv
gpu0=${ATTENTION_GPU0:-0}
gpu1=${ATTENTION_GPU1:-1}
mkdir -p "${queue_root}"
exec > >(tee -a "${queue_root}/controller.log") 2>&1

append_manifest() {
  local line=$1
  {
    flock 9
    printf '%s\n' "${line}" >&9
  } 9>>"${manifest}"
}

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

record_stage_status() {
  local label=$1 status=$2 note=${3:-}
  append_manifest "$(printf '%s\t%s\t%s\t%s\t%s' "${label}" "${status}" "$(date --iso-8601=seconds)" "$(date --iso-8601=seconds)" "${note}")"
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
  append_manifest "$(printf '%s\tSTARTED\t%s\t\t' "${label}" "${started}")"
  echo "[START] ${label} ${started}"
  set +e
  timeout --kill-after=5m 24h "$@"
  rc=$?
  set -e
  finished=$(date --iso-8601=seconds)
  if [[ ${rc} -eq 0 ]]; then
    append_manifest "$(printf '%s\tDONE\t%s\t%s\t0' "${label}" "${started}" "${finished}")"
    echo "[DONE] ${label} ${finished}"
  else
    append_manifest "$(printf '%s\tFAILED\t%s\t%s\t%s' "${label}" "${started}" "${finished}" "${rc}")"
    echo "[FAILED-CONTINUE] ${label} rc=${rc} ${finished}"
  fi
}

run_stage_gpu() {
  local label=$1 gpu=$2
  shift 2
  run_stage "${label}" env CUDA_VISIBLE_DEVICES="${gpu}" "$@"
}

# Run two independent benchmark arms concurrently.  Each process sees exactly
# one physical GPU as cuda:0, which keeps the custom attention hooks unchanged.
# Manifest writes are serialized by append_manifest.
run_profile_pair() {
  local task=$1 shots=$2 steps=$3 count=$4 gen_length=$5
  local profile0=$6 tag0=$7 profile1=$8 tag1=$9
  local pid0 pid1 rc0 rc1
  echo "[DUAL-GPU] gpu=${gpu0} ${tag0} | gpu=${gpu1} ${tag1}"
  run_stage_gpu "${tag0}" "${gpu0}" "${runner}" \
    "${task}" "${shots}" "${steps}" "${profile0}" 0.004 trace "${tag0}" "${count}" "${gen_length}" &
  pid0=$!
  run_stage_gpu "${tag1}" "${gpu1}" "${runner}" \
    "${task}" "${shots}" "${steps}" "${profile1}" 0.004 trace "${tag1}" "${count}" "${gen_length}" &
  pid1=$!
  set +e
  wait "${pid0}"; rc0=$?
  wait "${pid1}"; rc1=$?
  set -e
  echo "[DUAL-GPU-DONE] ${tag0} rc=${rc0}; ${tag1} rc=${rc1}"
  return 0
}

run_profile_single() {
  local gpu=$1 task=$2 shots=$3 steps=$4 profile=$5 tag=$6 count=$7 gen_length=$8
  run_stage_gpu "${tag}" "${gpu}" "${runner}" \
    "${task}" "${shots}" "${steps}" "${profile}" 0.004 trace "${tag}" "${count}" "${gen_length}"
}

audit_records() {
  local root=$1
  if [[ -e "${root}/audit/task_audit_records.jsonl" ]]; then
    echo "${root}/audit/task_audit_records.jsonl"
  elif [[ -e "${root}/audit/audit_records.jsonl" ]]; then
    echo "${root}/audit/audit_records.jsonl"
  else
    return 1
  fi
}

pair_stage() {
  local label=$1 task=$2 baseline_tag=$3 method_tag=$4
  local baseline_root=${5:-${run_base}/${task}/${baseline_tag}}
  local method_root=${run_base}/${task}/${method_tag}
  local output=${queue_root}/paired/${label}
  if [[ ! -e "${baseline_root}/DONE" || ! -e "${method_root}/DONE" ]]; then
    echo "[PAIR-SKIP] ${label}: a parent run is incomplete"
    record_stage_status "${label}" SKIPPED parent_incomplete
    return 0
  fi
  local baseline_records method_records
  baseline_records=$(audit_records "${baseline_root}") || {
    record_stage_status "${label}" FAILED baseline_records_missing
    return 0
  }
  method_records=$(audit_records "${method_root}") || {
    record_stage_status "${label}" FAILED method_records_missing
    return 0
  }
  run_stage "${label}" /root/miniconda3/bin/python compare_paired_task_runs.py \
    "${baseline_records}" "${method_records}" \
    --baseline-config "${baseline_root}/run_config.txt" \
    --method-config "${method_root}/run_config.txt" \
    --method-log "${queue_root}/${method_tag}.log" \
    --output-dir "${output}"
}

gate_pair() {
  local pair_label=$1 tolerance_correct=${2:-0}
  local summary=${queue_root}/paired/${pair_label}/paired_summary.json
  [[ -s "${summary}" ]] || return 1
  /root/miniconda3/bin/python - "${summary}" "${tolerance_correct}" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
tolerance = int(sys.argv[2])
healthy = (
    summary.get("prompt_hash_mismatches") == 0
    and summary.get("target_hash_mismatches") == 0
    and summary.get("duplicate_or_missing_ids") == 0
)
raise SystemExit(0 if healthy and summary["method_correct"] + tolerance >= summary["baseline_correct"] else 1)
PY
}

choose_best_profile() {
  local output_json=$1
  shift
  local args=()
  for candidate in "$@"; do args+=(--candidate "${candidate}"); done
  /root/miniconda3/bin/python "${llada_root}/select_queue_profile.py" choose \
    --output "${output_json}" "${args[@]}"
}

pair_beats_pair() {
  local candidate_label=$1 parent_label=$2 required_gain=${3:-1}
  /root/miniconda3/bin/python "${llada_root}/select_queue_profile.py" beats \
    "${queue_root}/paired/${candidate_label}/paired_summary.json" \
    "${queue_root}/paired/${parent_label}/paired_summary.json" \
    --required-gain "${required_gain}"
}

run_if_gate() {
  local gate_label=$1 tolerance=$2 stage_label=$3
  shift 3
  if gate_pair "${gate_label}" "${tolerance}"; then
    run_stage "${stage_label}" "$@"
  else
    echo "[GATE-SKIP] ${stage_label}: ${gate_label} did not pass tolerance=${tolerance}"
    record_stage_status "${stage_label}" SKIPPED "gate=${gate_label}"
  fi
}

export PATH=/root/miniconda3/bin:${PATH}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
cd "${llada_root}"

if [[ ! -e "${manifest}" ]]; then
  printf 'stage\tstatus\tstart\tfinish\texit_code_or_note\n' > "${manifest}"
fi

echo "queue_id=${queue_id}"
echo "controller_start=$(date --iso-8601=seconds)"
echo "formal_baseline=original_llada_low_confidence"
echo "accuracy_parent=symmetric_attention_tau_0.004"
echo "new_vertical=response_credit_on_strong_conditioning_events"
echo "new_vertical_minimal=conditioned_revision_margin_in_unstable_tail"
echo "new_horizontal=two_draft_agreement_skeleton_and_disagreement_redenoising"
echo "code_selection=prompt_visible_examples_plus_behavior_consensus_no_hidden_tests"
echo "gpu_assignment=${gpu0},${gpu1}"
echo "parallelism=one_independent_evaluator_per_gpu"

/root/miniconda3/bin/python -m py_compile \
  generate.py eval_llada.py differential_selector.py select_queue_profile.py audit_lm_eval_task.py \
  compare_paired_task_runs.py
PYTHONPATH=. /root/miniconda3/bin/python test_attention_stability.py
PYTHONPATH=. /root/miniconda3/bin/python test_differential_selector.py
PYTHONPATH=. /root/miniconda3/bin/python test_response_credit_exchange.py
PYTHONPATH=. /root/miniconda3/bin/python test_audit_lm_eval_task.py
PYTHONPATH=. /root/miniconda3/bin/python test_queue_profile_selection.py
/root/miniconda3/bin/python -m unittest discover -s tests -p 'test_*.py'

sha256sum generate.py eval_llada.py differential_selector.py select_queue_profile.py model/modeling_llada.py \
  validate_step_diagnostics.py audit_attention_stability.py audit_lm_eval_task.py \
  compare_paired_task_runs.py postprocess_code.py sanitize.py \
  code_eval/code_eval.py code_eval/execute.py tasks/localleap_math500/math500.yaml \
  tasks/localleap_math500/utils.py "${runner}" "$0" \
  > "${queue_root}/FROZEN_SOURCE_SHA256"

# Real-model/evaluator smoke: original LLaDA is the formal baseline and the
# established symmetric_fast arm is the development parent.
run_profile_pair humaneval 0 128 2 256 \
  baseline smoke_he_baseline symmetric_fast smoke_he_symmetric_fast
run_profile_pair humaneval 0 128 2 256 \
  response_credit_fast smoke_he_response_credit_fast \
  revision_margin_fast smoke_he_revision_margin_fast
run_profile_single "${gpu0}" humaneval 0 128 draft_exchange_exec smoke_he_draft_exchange_exec 2 256

# HumanEval is the development set. Explore on 32 examples, confirm on 64, and
# promote only a new single-trajectory rule that strictly beats symmetric_fast.
run_profile_pair humaneval 0 128 32 256 \
  baseline he_baseline_128_n32 symmetric_fast he_symmetric_fast_128_n32
run_profile_pair humaneval 0 128 32 256 \
  response_credit_fast he_response_credit_fast_128_n32 \
  revision_margin_fast he_revision_margin_fast_128_n32
run_profile_single "${gpu0}" humaneval 0 128 draft_exchange_exec he_draft_exchange_exec_128_n32 32 256
for profile in symmetric_fast response_credit_fast revision_margin_fast draft_exchange_exec; do
  tag="he_${profile}_128_n32"
  pair_stage "pair_${tag}" humaneval he_baseline_128_n32 "${tag}"
done

run_profile_pair humaneval 0 128 64 256 \
  baseline he_baseline_128_n64 symmetric_fast he_symmetric_fast_128_n64
for profile in symmetric_fast response_credit_fast revision_margin_fast; do
  tag="he_${profile}_128_n64"
  if [[ "${profile}" != symmetric_fast ]]; then
    run_if_gate "pair_he_${profile}_128_n32" 1 "${tag}" \
      env CUDA_VISIBLE_DEVICES="${gpu0}" "${runner}" humaneval 0 128 "${profile}" 0.004 trace "${tag}" 64 256
  fi
  pair_stage "pair_${tag}" humaneval he_baseline_128_n64 "${tag}"
done
run_if_gate pair_he_draft_exchange_exec_128_n32 1 he_draft_exchange_exec_128_n64 \
  env CUDA_VISIBLE_DEVICES="${gpu1}" "${runner}" humaneval 0 128 draft_exchange_exec 0.004 trace he_draft_exchange_exec_128_n64 64 256
pair_stage pair_he_draft_exchange_exec_128_n64 humaneval he_baseline_128_n64 he_draft_exchange_exec_128_n64

selection_json=${queue_root}/selection/humaneval_single_n64.json
selected_single=$(choose_best_profile "${selection_json}" \
  "symmetric_fast=${queue_root}/paired/pair_he_symmetric_fast_128_n64/paired_summary.json" \
  "response_credit_fast=${queue_root}/paired/pair_he_response_credit_fast_128_n64/paired_summary.json" \
  "revision_margin_fast=${queue_root}/paired/pair_he_revision_margin_fast_128_n64/paired_summary.json")
echo "selected_single_profile=${selected_single}"
echo "${selected_single}" > "${queue_root}/selection/selected_single_profile.txt"

promoted_single=none
if [[ "${selected_single}" != symmetric_fast && "${selected_single}" != none ]] \
  && pair_beats_pair "pair_he_${selected_single}_128_n64" pair_he_symmetric_fast_128_n64 1; then
  promoted_single=${selected_single}
fi
promote_draft=false
if pair_beats_pair pair_he_draft_exchange_exec_128_n64 pair_he_symmetric_fast_128_n64 1; then
  promote_draft=true
fi
echo "promoted_single=${promoted_single}" | tee "${queue_root}/selection/promotion.txt"
echo "promote_draft=${promote_draft}" | tee -a "${queue_root}/selection/promotion.txt"

if [[ "${promoted_single}" != none || "${promote_draft}" == true ]]; then
  run_profile_pair humaneval 0 128 full 256 \
    baseline he_baseline_128_full_fresh \
    symmetric_fast he_symmetric_fast_128_full_fresh
  pair_stage pair_he_symmetric_fast_128_full_fresh humaneval he_baseline_128_full_fresh he_symmetric_fast_128_full_fresh
else
  record_stage_status he_full_promotions SKIPPED no_new_arm_beat_parent
fi

if [[ "${promoted_single}" != none ]]; then
  tag="he_${promoted_single}_128_full"
  run_stage "${tag}" "${runner}" humaneval 0 128 "${promoted_single}" 0.004 trace "${tag}" full 256
  pair_stage "pair_${tag}" humaneval he_baseline_128_full_fresh "${tag}"
  pair_stage "pair_${tag}_vs_parent" humaneval unused "${tag}" \
    "${run_base}/humaneval/he_symmetric_fast_128_full_fresh"
fi
if [[ "${promote_draft}" == true ]]; then
  run_stage he_draft_exchange_exec_128_full \
    "${runner}" humaneval 0 128 draft_exchange_exec 0.004 trace he_draft_exchange_exec_128_full full 256
  pair_stage pair_he_draft_exchange_exec_128_full humaneval he_baseline_128_full_fresh he_draft_exchange_exec_128_full
  pair_stage pair_he_draft_exchange_exec_128_full_vs_parent humaneval unused he_draft_exchange_exec_128_full \
    "${run_base}/humaneval/he_symmetric_fast_128_full_fresh"
fi

# The HumanEval-selected rule is tested without retuning on held-out benchmarks.
# If no new rule wins, revision_margin_fast remains a clearly labelled exploratory arm.
generalization_profile=${promoted_single}
if [[ "${generalization_profile}" == none ]]; then generalization_profile=revision_margin_fast; fi
echo "${generalization_profile}" > "${queue_root}/selection/generalization_profile.txt"

# MBPP uses 100-example gates. Full promotion requires at least +3/100 over
# symmetric_fast, which avoids spending hours on a marginal exploratory tie.
run_profile_pair mbpp 3 128 100 256 \
  baseline mbpp_baseline_128_n100 symmetric_fast mbpp_symmetric_fast_128_n100
if [[ "${generalization_profile}" != symmetric_fast ]]; then
  run_profile_pair mbpp 3 128 100 256 \
    "${generalization_profile}" "mbpp_${generalization_profile}_128_n100" \
    draft_exchange_exec mbpp_draft_exchange_exec_128_n100
else
  run_profile_single "${gpu1}" mbpp 3 128 draft_exchange_exec mbpp_draft_exchange_exec_128_n100 100 256
fi
for profile in symmetric_fast "${generalization_profile}" draft_exchange_exec; do
  tag="mbpp_${profile}_128_n100"
  pair_stage "pair_${tag}" mbpp mbpp_baseline_128_n100 "${tag}"
done

promote_mbpp_single=false
promote_mbpp_draft=false
if pair_beats_pair "pair_mbpp_${generalization_profile}_128_n100" pair_mbpp_symmetric_fast_128_n100 3; then
  promote_mbpp_single=true
fi
if pair_beats_pair pair_mbpp_draft_exchange_exec_128_n100 pair_mbpp_symmetric_fast_128_n100 3; then
  promote_mbpp_draft=true
fi
if [[ "${promote_mbpp_single}" == true || "${promote_mbpp_draft}" == true ]]; then
  run_profile_pair mbpp 3 128 full 256 \
    baseline mbpp_baseline_128_full_fresh \
    symmetric_fast mbpp_symmetric_fast_128_full_fresh
  pair_stage pair_mbpp_symmetric_fast_128_full_fresh mbpp mbpp_baseline_128_full_fresh mbpp_symmetric_fast_128_full_fresh
fi
if [[ "${promote_mbpp_single}" == true ]]; then
  tag="mbpp_${generalization_profile}_128_full"
  run_stage "${tag}" "${runner}" mbpp 3 128 "${generalization_profile}" 0.004 trace "${tag}" full 256
  pair_stage "pair_${tag}" mbpp mbpp_baseline_128_full_fresh "${tag}"
  pair_stage "pair_${tag}_vs_parent" mbpp unused "${tag}" \
    "${run_base}/mbpp/mbpp_symmetric_fast_128_full_fresh"
fi
if [[ "${promote_mbpp_draft}" == true ]]; then
  run_stage mbpp_draft_exchange_exec_128_full \
    "${runner}" mbpp 3 128 draft_exchange_exec 0.004 trace mbpp_draft_exchange_exec_128_full full 256
  pair_stage pair_mbpp_draft_exchange_exec_128_full mbpp mbpp_baseline_128_full_fresh mbpp_draft_exchange_exec_128_full
  pair_stage pair_mbpp_draft_exchange_exec_128_full_vs_parent mbpp unused mbpp_draft_exchange_exec_128_full \
    "${run_base}/mbpp/mbpp_symmetric_fast_128_full_fresh"
fi

# Larger non-code datasets stay sampled in this round. Execution selection is
# disabled; these runs test the vertical rule and shared-skeleton repair directly.
for task_spec in "localleap_math500:0:100" "gsm8k:0:128"; do
  IFS=: read -r task shots count <<<"${task_spec}"
  baseline_tag="${task}_baseline_128_n${count}"
  parent_tag="${task}_symmetric_fast_128_n${count}"
  run_profile_pair "${task}" "${shots}" 128 "${count}" 256 \
    baseline "${baseline_tag}" symmetric_fast "${parent_tag}"
  if [[ "${generalization_profile}" != symmetric_fast ]]; then
    run_profile_pair "${task}" "${shots}" 128 "${count}" 256 \
      "${generalization_profile}" "${task}_${generalization_profile}_128_n${count}" \
      draft_exchange "${task}_draft_exchange_128_n${count}"
  else
    run_profile_single "${gpu1}" "${task}" "${shots}" 128 \
      draft_exchange "${task}_draft_exchange_128_n${count}" "${count}" 256
  fi
  for profile in symmetric_fast "${generalization_profile}" draft_exchange; do
    tag="${task}_${profile}_128_n${count}"
    pair_stage "pair_${tag}" "${task}" "${baseline_tag}" "${tag}"
  done
done

echo "controller_finish=$(date --iso-8601=seconds)"
touch "${queue_root}/DONE"
