# Project State

Date: 2026-07-15

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

The completed low-threshold and candidate-memory family established that the
best HumanEval result is 71/164 (43.29%) for tau 0.004 through 0.0005 and for
direct candidate stability, versus 67/164 baseline. Because this HumanEval
configuration has b=1, rejected horizontal pairs are identically zero; the gain
is longitudinal and not evidence for same-batch horizontal control. At tau
0.0005 every historical position is strong and the rule reduces to adjacent
top-1 stability. The gain is exploratory and not significant (7 method-only,
3 baseline-only, exact McNemar p=0.34375).

The STCC round `stcc_overnight_20260715_v1` was cancelled after its first two
full HumanEval arms scored 47/164 and 46/164, well below the 67/164 baseline.
The active round is now `attention_recovery_long_20260715_v1`, anchored to the
previous best `attention_stability_v1` result at tau 0.004 (71/164). The exact
old symmetric path remains the default. Optional directed-read dependencies
test attention asymmetry, while stable-conflict pruning plus fixed-budget fill
provides parallel 128/64-step speed arms without changing the anchor path.
Eight selector tests pass. A resumable detached queue runs HumanEval, MBPP,
Minerva counting/probability and GSM8K baseline, symmetric, directed and speed
profiles, with per-run evaluator audits, source hashes, an 8 GiB disk gate and
continue-on-run-failure behavior. Queue root:
`/root/autodl-tmp/LocalLeap/llada/results/experiment_queues/attention_recovery_long_20260715_v1`.

On 2026-07-16 the queue produced a formally audited MBPP directed result of
123/500 (24.6%) versus the original-LLaDA 128-step anchor of 89/500 (17.8%).
The conservative selector used a mean of about 150 NFE, so this is an
accuracy-first result rather than a compute-matched conclusion. Correct MBPP
samples had roughly half as many unstable candidates as incorrect samples,
showing that binary top-1 instability spends extra NFE mainly on hard cases.

The next registered method is therefore a single-change descendant of the
best symmetric tau-0.004 decoder: cross-step top-K overlap creates a three-level
temporal order (top-1 stable, candidate-set continuous, candidate-set broken),
while the horizontal symmetric exclusion is unchanged. K=4 is fixed a priori,
not selected on benchmark accuracy. A fixed-budget variant fills only after the
same selector runs. Ten selector tests and two tests each for the variable-NFE
validator and GSM8K filter-aware auditor pass. The replacement queue is
`attention_retention_v2_20260716`; it drops full 64-step method branches and
tests HumanEval, MBPP, and GSM8K against original LLaDA and the best parent.

## Deprecated (cleaned 2026-07-13)

Trajectory / lateral-response / agreement / ceiling-bug evals were removed from `results/` (local + AutoDL `dlm-seq-flow`).  
Audit notes kept under `cursorfeedback/`. Archived narrative docs under `cursorfeedback/archive/`.

## Kept baselines in this repo

- `results/round0_baseline/gsm8k_lcr.json` — LCR GSM8K 62%
- `results/round_block32_*` — region decoding positive evidence
- `experiments/localleap/` — LocalLeap scripts + protocol (mirror of AutoDL)
