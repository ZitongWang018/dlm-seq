# Project State

Date: 2026-07-13

Remote worktrees:

- Research code: `/root/autodl-tmp/dlm-seq-flow`
- **LocalLeap reproduction (protected): `/root/autodl-tmp/LocalLeap`**

## Current focus

Reproduce and extend **LocalLeap** on LLaDA-Instruct (local weights), with correct HumanEval scoring via `postprocess_code.py`.

See `experiments/localleap/README.md` for configs, scripts, and HE numbers.

The attention-stability decoder has now been implemented and evaluated with the
exact HumanEval baseline configuration. At tau 0.01 it scored 66/164 (40.24%)
versus baseline 67/164 (40.85%), with identical total NFE (41984). Treat this as
negative evidence, not a completed improvement. Implementation and audit scripts
are mirrored under `experiments/localleap/attention_stability/`.

The next sensitivity run is configured as a strict sequential sweep over tau
`0.005`, `0.02`, and `0.05`. Each run records compressed per-example, per-step
attention/candidate diagnostics and is validated before the next threshold can
start. The runner and schema validator are in the mirrored extension directory.

## Deprecated (cleaned 2026-07-13)

Trajectory / lateral-response / agreement / ceiling-bug evals were removed from `results/` (local + AutoDL `dlm-seq-flow`).  
Audit notes kept under `cursorfeedback/`. Archived narrative docs under `cursorfeedback/archive/`.

## Kept baselines in this repo

- `results/round0_baseline/gsm8k_lcr.json` — LCR GSM8K 62%
- `results/round_block32_*` — region decoding positive evidence
- `experiments/localleap/` — LocalLeap scripts + protocol (mirror of AutoDL)
