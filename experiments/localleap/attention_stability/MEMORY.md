# Memory

## 2026-07-19 — Trajectory-likelihood selection

- Formally rejected confidence-switched stability after matched discovery:
  HumanEval 12/32 versus `symmetric_fast` 13/32 and MATH-500 12/50 versus
  15/50.  It changed 17/32 and 29/50 generations respectively but recovered
  zero fast-parent errors, so no larger run was launched.
- The fast/accuracy-first oracle unions remain 16/32 HumanEval and 18/50
  MATH-500, three above the best individual path on each slice.  This supports
  complete-trajectory selection rather than more single-path waiting logic.
- Added a two-parent selector that accumulates the mean log top-1 confidence
  at each token's actual commit step and selects the higher-likelihood complete
  path.  It introduces no threshold, blending weight, execution feedback, or
  token-level crossover.  Added 19 selector tests, two queue-order/gate tests,
  a method note, and a dual-GPU HumanEval/MATH discovery queue.
- Fixed a new-host HumanEval evaluator deadlock.  Hugging Face code_eval's
  ThreadPool/Manager/default-fork wrapper stalled after runtime thread
  initialization on Python 3.12.  Versioned the same official execution guard
  and timeout with spawn plus shared status as
  `humaneval_spawn_official_checker_v3`; five synthetic correct/error/timeout
  cases now pass in 11 seconds.  Formal scoring semantics remain one candidate
  pass@1 per task.

## 2026-07-18 — Confidence-switched stability retrospective

- Re-audited the method lineage before starting another experiment.  The only
  parents retained are original LLaDA, symmetric tau 0.004, and its fixed-
  budget `symmetric_fast` child.  Global Top-K retention, response-credit draft
  exchange, response refinement, cross-conditioned Pareto retention, and
  differential execution selection remain negative descendants.
- The last cross-conditioned core was correctness-identical to its matched
  parent on HumanEval-32 (13/32) and MATH-50 (15/50) at roughly 3% extra NFE.
  Its execution-selection child scored 1/32 HumanEval and is rejected.
- Added `symmetric_risk_switch`, a configuration-only descendant with no new
  generation branch or hyperparameter.  Stable attention conflicts are pruned
  to retain parallelism; conflicts involving an adjacent-step conditioned
  top-1 rewrite are not force-filled and receive another ordinary denoising
  step.  This is the minimal uncertainty switch suggested jointly by the MBPP
  instability statistics and the search/accelerate principle in recent DLM
  decoding work.
- Added a focused selector regression test, method note, and a frozen-source
  two-GPU queue.  It regenerates original LLaDA, `symmetric_fast`, the accuracy
  reference, and the new method on HumanEval/MATH matched slices.  Promotion
  requires a strict combined paired gain with no per-task loss greater than one
  and at most 1.35x NFE; MBPP/GSM8K sampled generalization is downstream only.

## 2026-07-17

- Added `revision_margin_fast`, a minimal vertical descendant of the strongest
  `symmetric_fast` parent. For a strong conditioned top-1 change it records the
  current log-probability advantage of the new candidate over the previous one,
  and uses this only inside the unstable tail. It adds no method threshold or
  historical distribution and preserves the parent mature order, horizontal
  tau-0.004 exclusion and fixed commit budget.
- Reworked the pending queue into sampled development gates: HumanEval 32 then
  64; MBPP 100; MATH-500 100; GSM8K 128. New single-trajectory rules must
  strictly beat `symmetric_fast`; ties remain with the parent. A tested,
  auditable profile selector prevents unhealthy or tied arms from receiving
  full runs.
- Corrected future GSM8K auditing to lm-eval `flexible-extract`. The completed
  original-LLaDA zero-shot baseline was independently re-audited at 905/1319
  (68.61%), while the earlier 1/1319 strict-match figure was an extraction
  configuration failure and is not a model result.

## 2026-07-16 — Return to best symmetric parent and broaden benchmarks

- Formally rejected global top-K retention (HumanEval-256 64/164). The
  parent-preserving v2.1 variant recovered 71/164 at 256 steps but reached only
  47/164 at 128 steps, below the 52/164 symmetric-fast parent.
- Cancelled `attention_retention_v21_20260716` while preserving its completed
  HumanEval runs and partial MBPP output.
- Added a cached, zero-shot MATH-500 lm-eval task, stable `unique_id` tracing,
  constant-NFE audit records, general paired task audits, and a resumable long
  queue based only on original LLaDA versus symmetric tau 0.004.
- Added a versioned paper-alignment table for Prism, SOAR and Order-Token Search.
  Paper values remain external references because checkpoints, prompts and
  inference stacks are not fully identical.
- Stopped the initial v1 generalization queue during its first full MATH-500
  stage after a smoke audit exposed a prompt/evaluator mismatch. Version v2 uses
  Prism's official MATH-500 system prompt and answer normalization and writes to
  a new result root; no v1 result was repaired or overwritten.

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

## 2026-07-15

- Cancelled the distribution-response STCC queue after full HumanEval scores
  of 47/164 and 46/164. Restored the previous best attention-stability decoder
  at tau 0.004 (historical 71/164 versus 67/164 baseline) as the fixed anchor.
- Added optional directed-read dependency selection and a speed profile that
  prunes conflicts between cross-step-stable candidates and fills the original
  per-step budget. The default symmetric path is backward compatible; eight
  selector tests pass.
- Started detached queue `attention_recovery_long_20260715_v1`: HumanEval,
  MBPP, Minerva counting/probability, and GSM8K baseline/symmetric/directed and
  128/64-step speed arms, with source freezing, evaluator audits, disk gating,
  resumption, and continue-on-independent-run-failure behavior.

## 2026-07-16

- Designed Response-Credit Draft Exchange on top of the verified symmetric
  tau-0.004 parent. Response credit is earned only after a top-1 survives a
  strong newly committed dependency and resets on a conditioned change; weak
  steps preserve rather than inflate credit. Added a fixed-budget variant.
- Added two deterministic full drafts (top-1 anchor and response-credit
  explorer), agreement-skeleton preservation, and re-denoising of all draft
  disagreements. Code tasks optionally choose among anchor/explorer/repaired
  using prompt-visible doctests and deterministic behavior consensus only;
  hidden tests and reference answers are not passed to the selector.
- Added complete candidate/selection trace fields, 21 focused regression tests,
  new benchmark profiles, and a gated queue covering HumanEval, MBPP,
  MATH-500 and GSM8K. The queue waits for the active best-symmetric queue and
  promotes expensive arms only after paired 32/64/100-example checks.

- Formally audited MBPP directed attention stability at 123/500 (24.6%) versus
  original LLaDA 89/500 (17.8%). Mean NFE was about 150 rather than 128, so the
  gain is accuracy-first. Symmetric was independently audited at 121/500.
- Joined trace and correctness records. Correct directed MBPP samples averaged
  113.7 unstable candidates and 147.8 NFE; incorrect samples averaged 219.7
  unstable candidates and 150.7 NFE. This motivated reducing brittle top-1
  waiting instead of adding another divergence threshold.
- Added a top-K-overlap longitudinal mode on the exact symmetric tau-0.004
  parent. It uses three lexicographic temporal tiers and fixed K=4; horizontal
  selection is unchanged. Added a fixed-budget child for speed comparisons.
- Fixed the variable-NFE validator to compare contiguous records against their
  actual length, not configured steps. Added GSM8K strict-filter auditing so
  lm-eval's two filter records are not mistaken for duplicate tasks. Regression
  tests and real one-record audits pass without overwriting old results.
- Prepared queue `attention_retention_v2_20260716` for HumanEval, MBPP and
  GSM8K original-LLaDA, best-parent, retention and fixed-budget comparisons.
  The old controller is superseded after its active original-LLaDA MBPP
  64-step baseline finishes; weak 64-step method branches are removed.

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
