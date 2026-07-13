# Memory

## 2026-07-14

- Completed the first attention-stability sensitivity sweep. Exact-baseline
  HumanEval results were tau 0.005 = 69/164 (42.07%), tau 0.02 = 66/164
  (40.24%), and tau 0.05 = 67/164 (40.85%), versus baseline 67/164. The tau
  0.005 paired gain was 6 method-only versus 4 baseline-only tasks (exact
  McNemar p=0.7539), so it is exploratory rather than conclusive.

- Audited 650,752 cross-step candidates per threshold. Tau 0.005 had 10.903%
  changed candidates and `P(change|strong)=11.395%`; tau 0.02 had 10.750%
  changed candidates but `P(change|strong)=8.337%`, lower than its weak-pair
  rate of 11.114%. Win/loss samples were separated more by candidate volatility
  than by attention strength, motivating direct stability and a conservative
  confidence-frontier rule.

- Implemented `candidate_memory_stability_v2` in the protected LocalLeap tree
  and mirrored it in this repository. It preserves the baseline TopB budget,
  stores Top-8 candidates only for remaining masks, compares adjacent top-1,
  uses previous-Top-K-plus-OTHER partition JSD, and records directional
  attention arrival plus extensive per-candidate diagnostics. Exact JSD is an
  explicit off-by-default diagnostic only; formal runs retain zero full-vocab
  history elements and never write full distributions to disk.

- Added a no-extra-size-parameter frontier improvement: only confidence rank
  `<= b_t+1` may replace the baseline choice based on stability. Ten
  selector/generator tests pass, including argmax/Top-K ties and a true
  three-candidate frontier. Candidate full runs require a same-source-hash
  one-task end-to-end preflight before the 164-task run. Started locked,
  fail-fast full queue `cross_step_full_queue_20260714_v1` for tau 0.004,
  0.0025, 0.001, 0.0005, direct stability, and frontier stability.

## 2026-07-13

- Added a fail-fast sequential tau sweep (`0.005`, `0.02`, `0.05`) for the
  attention-stability HumanEval experiment. Each example now saves all 256
  decoding steps in `attention_stability_steps_v1` files: directional and
  symmetric 32x32 attention, candidate top-1/confidence/history, dependency,
  maturity, ordering, selected/rejected positions, budgets, masks, and fallback
  state. A validator checks counts, shapes, finite values, symmetry, contiguous
  NFE, budgets, stable ids, and residual masks before a run is marked complete.

- Implemented the attention-stability decoder in the protected remote LocalLeap
  worktree and mirrored it under `experiments/localleap/attention_stability/`.
  Exact-baseline HumanEval at tau 0.01 gave 66/164 (40.24%) versus baseline
  67/164 (40.85%), with 4 method-only and 5 baseline-only tasks (McNemar p=1.0).
  Total NFE stayed 41984 and TPS was 9.439 versus baseline 9.730. The method
  changed 121 generations but did not improve accuracy.

- Cleaned misleading eval artifacts (failed traj/response/agreement/ceiling) from local + AutoDL `dlm-seq-flow/results`.
- **Did not touch** `/root/autodl-tmp/LocalLeap` (code, configs, baseline/LocalLeap HE results).
- Mirrored LocalLeap scripts into `experiments/localleap/` for future tests.
- HumanEval (postprocess): baseline 40.85%, LocalLeap 40.24%; TPS ~9.7 → ~41 on one GPU.
- Official HE score requires `postprocess_code.py`; raw lm_eval underestimates.

## Earlier (summary)

- Old “recover gold from past logits / direct response timing” directions failed on math.
- MATH/HumanEval ceiling scripts had metric bugs (see `cursorfeedback/0712grokproblem.md`).
- top-k distance support-index bug was fixed in `src/distribution.py` (tests cover it).
