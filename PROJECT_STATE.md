# Project State

Date: 2026-07-13

Remote worktrees:

- Research code: `/root/autodl-tmp/dlm-seq-flow`
- **LocalLeap reproduction (protected): `/root/autodl-tmp/LocalLeap`**

## Current focus

Reproduce and extend **LocalLeap** on LLaDA-Instruct (local weights), with correct HumanEval scoring via `postprocess_code.py`.

See `experiments/localleap/README.md` for configs, scripts, and HE numbers.

## Deprecated (cleaned 2026-07-13)

Trajectory / lateral-response / agreement / ceiling-bug evals were removed from `results/` (local + AutoDL `dlm-seq-flow`).  
Audit notes kept under `cursorfeedback/`. Archived narrative docs under `cursorfeedback/archive/`.

## Kept baselines in this repo

- `results/round0_baseline/gsm8k_lcr.json` — LCR GSM8K 62%
- `results/round_block32_*` — region decoding positive evidence
- `experiments/localleap/` — LocalLeap scripts + protocol (mirror of AutoDL)
