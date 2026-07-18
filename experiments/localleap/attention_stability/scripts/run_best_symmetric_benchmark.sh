#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
  echo "usage: $0 TASK SHOTS STEPS PROFILE TAU DIAGNOSTICS RUN_TAG [LIMIT] [GEN_LENGTH]" >&2
  exit 2
fi

task=$1
shots=$2
steps=$3
profile=$4
tau=$5
diagnostics_mode=$6
run_tag=$7
limit=${8:-full}
gen_length=${9:-256}

case "${task}" in
  humaneval) expected_full=164; primary_metric=postprocess ;;
  mbpp) expected_full=500; primary_metric=pass_at_1 ;;
  gsm8k) expected_full=1319; primary_metric='exact_match,flexible-extract' ;;
  localleap_math500) expected_full=500; primary_metric=exact_match ;;
  *) echo "unsupported task: ${task}" >&2; exit 2 ;;
esac

if (( gen_length % steps != 0 )); then
  echo "generation length ${gen_length} must be divisible by steps ${steps}" >&2
  exit 2
fi

case "${profile}" in
  baseline)
    dependency_args=""
    ;;
  symmetric)
    dependency_args=",dependency_threshold=${tau},dependency_mode=symmetric,dependency_prune_stable_conflicts=False,dependency_fill_budget=False"
    ;;
  symmetric_fast)
    dependency_args=",dependency_threshold=${tau},dependency_mode=symmetric,dependency_prune_stable_conflicts=True,dependency_fill_budget=True"
    ;;
  symmetric_risk_switch)
    # Confidence-switched decoding: stable conflicts are treated as dense,
    # low-information edges and retain the native parallel budget.  A conflict
    # involving a candidate rewritten by the newly committed condition is not
    # force-filled, so only risky steps spend additional denoising iterations.
    dependency_args=",dependency_threshold=${tau},dependency_mode=symmetric,dependency_prune_stable_conflicts=True,dependency_fill_budget=False"
    ;;
  trajectory_likelihood)
    # Two verified parent paths expose complementary horizontal schedules.
    # Select with vertical evidence accumulated exactly when tokens commit.
    dependency_args=",dependency_threshold=${tau},dependency_mode=symmetric,dependency_likelihood_selection=True"
    ;;
  trajectory_block_evidence)
    # Keep the parallel parent unless the slower path accumulates at least one
    # nat of extra commit evidence per existing generation block.
    dependency_args=",dependency_threshold=${tau},dependency_mode=symmetric,dependency_likelihood_selection=True,dependency_likelihood_selection_mode=block_evidence"
    ;;
  response_credit)
    dependency_args=",dependency_threshold=${tau},dependency_mode=symmetric,dependency_temporal_mode=response_credit,dependency_prune_stable_conflicts=False,dependency_fill_budget=False"
    ;;
  response_credit_fast)
    dependency_args=",dependency_threshold=${tau},dependency_mode=symmetric,dependency_temporal_mode=response_credit,dependency_prune_stable_conflicts=True,dependency_fill_budget=True"
    ;;
  revision_margin)
    dependency_args=",dependency_threshold=${tau},dependency_mode=symmetric,dependency_temporal_mode=revision_margin,dependency_prune_stable_conflicts=False,dependency_fill_budget=False"
    ;;
  revision_margin_fast)
    dependency_args=",dependency_threshold=${tau},dependency_mode=symmetric,dependency_temporal_mode=revision_margin,dependency_prune_stable_conflicts=True,dependency_fill_budget=True"
    ;;
  response_refine_fast)
    dependency_args=",dependency_threshold=${tau},dependency_response_refine=True,dependency_response_refine_budget_mode=matched"
    ;;
  response_refine_extra)
    dependency_args=",dependency_threshold=${tau},dependency_response_refine=True,dependency_response_refine_budget_mode=extra"
    ;;
  response_refine_gated)
    dependency_args=",dependency_threshold=${tau},dependency_response_refine=True,dependency_response_refine_budget_mode=gated"
    ;;
  response_refine_causal_pareto)
    dependency_args=",dependency_threshold=${tau},dependency_response_refine=True,dependency_response_refine_budget_mode=causal_pareto"
    ;;
  response_refine_cross_pareto)
    dependency_args=",dependency_threshold=${tau},dependency_response_refine=True,dependency_response_refine_budget_mode=causal_cross_pareto"
    ;;
  response_refine_cross_pareto_exec)
    if [[ "${task}" != "humaneval" && "${task}" != "mbpp" ]]; then
      echo "response_refine_cross_pareto_exec is code-only; got task=${task}" >&2
      exit 2
    fi
    dependency_args=",dependency_threshold=${tau},dependency_response_refine=True,dependency_response_refine_budget_mode=causal_cross_pareto,dependency_differential_selection=True"
    ;;
  draft_exchange)
    dependency_args=",dependency_threshold=${tau},dependency_mode=symmetric,dependency_draft_exchange=True,dependency_differential_selection=False,dependency_prune_stable_conflicts=False,dependency_fill_budget=False"
    ;;
  draft_exchange_exec)
    if [[ "${task}" != "humaneval" && "${task}" != "mbpp" ]]; then
      echo "draft_exchange_exec is code-only; got task=${task}" >&2
      exit 2
    fi
    dependency_args=",dependency_threshold=${tau},dependency_mode=symmetric,dependency_draft_exchange=True,dependency_differential_selection=True,dependency_prune_stable_conflicts=False,dependency_fill_budget=False"
    ;;
  *) echo "unsupported profile: ${profile}" >&2; exit 2 ;;
esac

if [[ "${diagnostics_mode}" == "full" && ( "${profile}" == draft_exchange* || "${profile}" == response_refine* ) ]]; then
  echo "draft/refine profiles expose complete trace summaries, not per-step tensor dumps" >&2
  exit 2
fi

expected_records=${expected_full}
limit_args=()
if [[ "${limit}" != "full" ]]; then
  expected_records=${limit}
  limit_args=(--limit "${limit}")
fi

queue_id=${ATTENTION_QUEUE_ID:-best_symmetric_long_20260716_v2}
llada_root=/root/autodl-tmp/LocalLeap/llada
queue_root=${llada_root}/results/experiment_queues/${queue_id}
run_root=${llada_root}/results/best_symmetric_benchmarks/${queue_id}/${task}/${run_tag}
trace_dir=${run_root}/trace
diagnostics_dir=${run_root}/step_diagnostics
audit_dir=${run_root}/audit
output_dir=${run_root}/lm_eval
log_file=${queue_root}/${run_tag}.log
task_path=${llada_root}/tasks

if [[ -e "${run_root}/DONE" ]]; then
  echo "completed run already exists; skipping: ${run_root}"
  exit 0
fi
mkdir -p "${queue_root}" "${run_root}" "${audit_dir}" "${output_dir}"

export PATH=/root/miniconda3/bin:${PATH}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
unset TRANSFORMERS_CACHE || true
unset HF_ENDPOINT || true
cd "${llada_root}"

trace_args=""
if [[ "${profile}" != "baseline" ]]; then
  mkdir -p "${trace_dir}"
  trace_args=",dependency_trace_dir=${trace_dir}"
  if [[ "${diagnostics_mode}" == "full" ]]; then
    mkdir -p "${diagnostics_dir}"
    trace_args+=",dependency_diagnostics_dir=${diagnostics_dir}"
  elif [[ "${diagnostics_mode}" != "trace" ]]; then
    echo "diagnostics must be trace or full" >&2
    exit 2
  fi
fi

model_args="model_path=/root/autodl-tmp/model/LLaDA/instruct,gen_length=${gen_length},steps=${steps},block_length=32,remasking=low_confidence,early_stop=False,show_speed=True,integrate_speed=False${dependency_args}${trace_args}"
{
  echo "schema=best_symmetric_benchmark_v2"
  echo "queue_id=${queue_id}"
  echo "task=${task}"
  echo "profile=${profile}"
  echo "tau=${tau}"
  echo "steps=${steps}"
  echo "gen_length=${gen_length}"
  echo "baseline_budget_per_step=$((gen_length / steps))"
  echo "expected_records=${expected_records}"
  echo "num_fewshot=${shots}"
  echo "seeds=0,1234,1234,1234"
  echo "model_args=${model_args}"
  echo "start=$(date --iso-8601=seconds)"
  git -C /root/autodl-tmp/dlm-seq-flow rev-parse HEAD
  sha256sum generate.py eval_llada.py differential_selector.py model/modeling_llada.py validate_step_diagnostics.py \
    audit_attention_stability.py audit_lm_eval_task.py compare_paired_task_runs.py \
    postprocess_code.py humaneval_execution.py sanitize.py
} > "${run_root}/run_config.txt"

include_args=()
if [[ "${task}" == "localleap_math500" ]]; then
  include_args=(--include_path "${task_path}")
fi

set +e
accelerate launch --num_processes 1 --num_machines 1 --mixed_precision no eval_llada.py \
  --model llada_dist --model_args "${model_args}" "${include_args[@]}" --tasks "${task}" \
  --num_fewshot "${shots}" --seed 0,1234,1234,1234 --confirm_run_unsafe_code \
  --output_path "${output_dir}" --log_samples "${limit_args[@]}" 2>&1 | tee "${log_file}"
rc=${PIPESTATUS[0]}
set -e
if [[ ${rc} -ne 0 ]]; then
  echo "${rc}" > "${run_root}/EXITCODE"
  touch "${run_root}/FAILED"
  exit "${rc}"
fi

samples=$(find "${output_dir}" -type f -name "samples_${task}_*.jsonl" | sort | tail -1)
results_json=$(find "${output_dir}" -type f -name 'results_*.json' | sort | tail -1)
[[ -n "${samples}" && -n "${results_json}" ]]

if [[ "${profile}" != "baseline" ]]; then
  trace=${trace_dir}/rank_0.jsonl
  [[ $(wc -l < "${trace}") -eq ${expected_records} ]]
  if [[ "${diagnostics_mode}" == "full" ]]; then
    /root/miniconda3/bin/python validate_step_diagnostics.py "${diagnostics_dir}" \
      --expected-count "${expected_records}" --output "${audit_dir}/step_diagnostics_summary.json"
  fi
fi

if [[ "${task}" == "humaneval" ]]; then
  /root/miniconda3/bin/python postprocess_code.py "${samples}" | tee "${run_root}/postprocess.txt"
  if [[ "${profile}" != "baseline" ]]; then
    /root/miniconda3/bin/python audit_attention_stability.py "${samples}" "${trace}" \
      --postprocess "${run_root}/postprocess.txt" --output-dir "${audit_dir}"
  else
    /root/miniconda3/bin/python audit_attention_stability.py "${samples}" \
      --constant-nfe "${steps}" --postprocess "${run_root}/postprocess.txt" \
      --output-dir "${audit_dir}"
  fi
else
  audit_args=()
  if [[ "${profile}" != "baseline" ]]; then
    audit_args=(--trace "${trace}")
  else
    audit_args=(--constant-nfe "${steps}")
  fi
  filter_args=()
  if [[ "${task}" == "gsm8k" ]]; then filter_args=(--filter flexible-extract); fi
  /root/miniconda3/bin/python audit_lm_eval_task.py "${samples}" "${results_json}" \
    --task "${task}" --primary-metric "${primary_metric}" "${audit_args[@]}" \
    "${filter_args[@]}" --expected-records "${expected_records}" --output-dir "${audit_dir}"
fi

echo 0 > "${run_root}/EXITCODE"
echo "finish=$(date --iso-8601=seconds)" >> "${run_root}/run_config.txt"
touch "${run_root}/DONE"
echo "run_root=${run_root}"
