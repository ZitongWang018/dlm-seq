# Project State

Date: 2026-07-14

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

The first strict tau sweep is complete: tau 0.005 scored 69/164 (42.07%), tau
0.02 scored 66/164 (40.24%), and tau 0.05 scored 67/164 (40.85%), all at the
same 41984 total NFE and with full evaluator/diagnostic audit. The +2 result at
tau 0.005 is not statistically significant (paired exact McNemar p=0.7539).

Candidate diagnostics show weak and inconsistent association between symmetric
attention strength and adjacent-step top-1 changes. The active experiment is
therefore a direct cross-step candidate-memory decoder plus a conservative
confidence-frontier variant. Both preserve the baseline per-step budget; store
Top-8+OTHER state for still-masked positions; use coarsened JSD for the formal
O(|M|K) method and offer exact JSD only as an off-by-default diagnostic; record
directional attention arrival, entropy, ranks and selection reasons; and use no
JSD/attention mixing weight. A locked fail-fast queue is running tau
`0.004, 0.0025, 0.001, 0.0005`, then direct stability and frontier stability on
full HumanEval. Ten selector/generator tests pass, and each candidate full run is gated
by an exact-source-hash one-task end-to-end preflight. Implementation, tests,
runners and validators are mirrored
under `experiments/localleap/attention_stability/`.

## Deprecated (cleaned 2026-07-13)

Trajectory / lateral-response / agreement / ceiling-bug evals were removed from `results/` (local + AutoDL `dlm-seq-flow`).  
Audit notes kept under `cursorfeedback/`. Archived narrative docs under `cursorfeedback/archive/`.

## Kept baselines in this repo

- `results/round0_baseline/gsm8k_lcr.json` — LCR GSM8K 62%
- `results/round_block32_*` — region decoding positive evidence
- `experiments/localleap/` — LocalLeap scripts + protocol (mirror of AutoDL)
