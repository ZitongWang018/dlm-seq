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

## 2026-07-15

- Finalized the formal HumanEval cross-step family. Baseline is 67/164;
  attention stability tau 0.004, 0.0025, 0.001, and 0.0005 plus direct
  candidate stability all score 71/164. Candidate frontier and the hard
  confidence threshold 0.8 both score 67/164. The best paired comparison is
  7 method-only versus 3 baseline-only, exact McNemar p=0.34375. HumanEval has
  b=1, so these runs validate longitudinal scheduling only; horizontal rejected
  pairs are zero.

- Implemented `stcc_distribution_response_v1`. It stores Top-8+OTHER and a
  short streak for remaining masks, orders candidates using partition JSD,
  Top-K overlap and top-1 change, and removes the failed hard-confidence and
  b+1-frontier mechanisms. Added symmetric and per-direction horizontal b=2
  constraints. Directed reads into low-JSD targets are pruned as dense
  low-information edges.

- Added acceleration multipliers 2 and 4. Baseline commits are never removed;
  extra commits require stable top-1, low JSD, Top-8 overlap at least 7, the
  frozen streak, and no active conflict. Quality arms assert fixed baseline NFE;
  acceleration arms report actual NFE and synchronized step wall time.

- Five STCC unit/generator tests and all eleven candidate-memory regression
  tests pass. Started full MBPP b=2 baseline and a 72-hour fail-fast controller
  `stcc_overnight_20260715_v1` covering HumanEval exploration, MBPP b=2
  symmetric/directed comparisons, Minerva counting/probability confirmation,
  and speed/performance arms. Added record, trace and full-step validators plus
  an 8 GiB disk stop gate and frozen-source hash checks.

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
