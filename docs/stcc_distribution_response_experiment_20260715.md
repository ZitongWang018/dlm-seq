# STCC distribution-response experiment, 2026-07-15

## Scope

This round replaces top-1-only maturity with a compact distribution-response
state.  It does not inject historical KV or alter model parameters.  For every
position that remains masked, the decoder stores the previous Top-8 ids and
probabilities, an OTHER mass, and a short stability streak.  The persistent
complexity remains O(|M|K).

HumanEval is exploratory because it has already been used to inspect threshold
behavior.  MBPP is the first full b=2 development benchmark.  The complete
Minerva counting-and-probability split is held until all HumanEval and MBPP arms
finish.  No partial MBPP or Minerva accuracy is used to change a running arm.

## Longitudinal response

The response divergence is JSD on the previous Top-K fixed partition plus an
OTHER bucket.  Top-1 identity is retained but is no longer the sole maturity
criterion.  Positions are ordered by four response classes:

1. stable top-1 and low JSD;
2. changed top-1 and low JSD, representing a small near-boundary flip;
3. stable top-1 and high JSD, representing substantial hidden redistribution;
4. changed top-1 and high JSD.

Within a class, lower JSD, larger Top-K overlap, longer low-response streak, and
higher current confidence are preferred.  The first step of every block remains
the original confidence TopB bootstrap.  The compact exploratory family uses
JSD thresholds 0.005, 0.01, and 0.02 with one verified transition, plus a single
two-transition streak arm at JSD 0.01.  It deliberately removes the failed hard
confidence threshold and the failed b+1 candidate frontier.

## Horizontal response

Horizontal comparisons use b=2: generation length 256, steps 128, and block
length 32.  The symmetric arm rejects a pair when the averaged attention exceeds
tau.  The directed arm tests A[target, source] in each direction before forming
the pair conflict.  A directed read is active only when its target has high
cross-step response.  Dense reads into a low-JSD target are counted and pruned as
low-information edges.  Tau is frozen at 0.004 for this round.

Quality arms always commit exactly the baseline budget.  If a greedy horizontal
scan cannot fill the budget, a deterministic response-order fill is used and
recorded as a forced conflict fill.  Thus baseline, vertical, symmetric, and
directed quality arms have exactly 128 NFE per b=2 task.

## Acceleration

Acceleration never removes a baseline commit.  After the fixed base selection,
an extra position can be committed only when it has history, stable top-1, JSD
below the frozen threshold, Top-8 overlap of at least 7, the required stability
streak, and no active pair conflict with any selected position.  Multipliers two
and four cap the total commits at 2b and 4b.  No extra commit is forced.  Actual
NFE, end-to-end time, summed synchronized forward time, tokens per second, peak
memory, base/extra commits, and the accuracy-NFE pair are reported.

## Frozen tasks

- HumanEval: full 164, 0-shot, length 256, steps 256, block 32.  This is an
  exploratory longitudinal and acceleration family; b=1 cannot validate
  horizontal exclusion.
- MBPP: full test 500, 3-shot, length 256, steps 128, block 32, seed tuple
  0/1234/1234/1234.  The initial baseline starts before method code is deployed
  and records the pre-deployment source hashes.
- Minerva counting and probability: full split 474, 4-shot, length 256, steps
  128, block 32, with the same seed tuple.

All tasks use the local LLaDA-Instruct weights, temperature zero,
low-confidence remasking, early_stop false, and the same lm-eval task templates
as the LocalLeap checkout.

## Evaluation and stop conditions

Every method trace contains a stable task id, absolute index, prompt hash, raw
gold, decoded generation, NFE, generation settings, source hashes, and decode
summary.  Quality arms must have fixed NFE; acceleration arms report actual NFE.
Full diagnostic runs additionally replay mask accounting, response classes,
Top-K overlap, extra-commit eligibility, active-edge membership, and conflict
checks.

The queue stops on source drift, evaluator aggregate disagreement, duplicate or
missing ids, prompt/generation mismatch, non-finite diagnostics, residual masks,
quality-budget mismatch, invalid extra commits, CUDA errors, or less than 8 GiB
free disk.  HumanEval uses the existing sanitized postprocessor plus independent
record aggregation.  MBPP uses pass_at_1 and Minerva uses math_verify as the
primary metric; the complete sample records are independently re-aggregated.

## Queue

The active controller is
`/root/autodl-tmp/LocalLeap/scripts/llada/run_stcc_overnight_queue.sh`.  Its queue
root is
`/root/autodl-tmp/LocalLeap/llada/results/experiment_queues/stcc_overnight_20260715_v1`.
It first waits for the already-running full MBPP b=2 baseline, audits that run,
runs direct regression tests and a one-example real-model smoke, freezes source
hashes, and then executes the HumanEval, MBPP, and Minerva families sequentially.
