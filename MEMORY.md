# Memory

## 2026-07-19

- Completed formal HumanEval-164 for confirmed bidirectional v9: 55 correct,
  versus symmetric-fast 49 and original LLaDA 42; untouched 96--163 is 14
  versus 11/11. All identity, source, NFE, residual-mask and evaluator audits
  passed.
- Completed frozen v11 public-example guard replay and independent official
  execution: 58/164, exactly matched by the second execution path; untouched
  96--163 is 15/68. V11 has three recoveries and zero losses versus v9.
- Completed MATH-50 promotion: v9 16, symmetric-fast 15, original LLaDA 13.
  Automatically launched GSM8K-64 and MBPP-50 on separate GPUs.
- Added exact lazy public-guard acceleration and its ordered audit queue. It
  removes an original-LLaDA trajectory only when a strict guard cannot possibly
  select it, so decoded output is provably unchanged.
- Added a preregistered public-tie full-draft verifier screen with mean and
  block-Pareto variants. Both reuse the v9 bidirectional verifier and add no
  threshold, token splice, hidden test, reference answer, or generated probe.
- Added an explicit cross-version mode to the paired evaluator. It still
  requires stable IDs and matching prompt/target hashes, but reports source
  drift rather than falsely claiming source identity; regression tests pass.
- Rejected additional MATH final-answer majority variants on indices 0--24:
  plain majority lost two correct answers and gained none; evidence-gated
  majority changed one neutral example and gained none.
- Added and queued an explicit outcome-set arbiter for MATH/GSM. It removes
  duplicate candidate answers, asks one normal dLLM pass to solve independently,
  and can only select an already generated complete trajectory. MATH 0--15 and
  GSM 0--15 are development; MATH 25--49 and GSM 32--63 were registered as
  holdouts before per-record inspection.

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

## 2026-07-17

- Deleted the old recurring queue monitor at the user's request. The old host
  was already shut down, so no remote process termination was attempted.
- Moved the `seetacloud-codex` alias to the new endpoint using the existing
  dedicated SSH public key. Verified two idle RTX 4080 SUPER GPUs with 32,760
  MiB each and preserved `/root/autodl-tmp` experiment data. No password was
  stored in repository files.
- Completed the versioned GSM8K flexible-extract audit: original LLaDA
  905/1319, symmetric-fast 901/1319 at matched 128-step NFE, and symmetric
  865/1319 at 1.179x NFE. The earlier strict-match 1/1319 baseline is an
  evaluator configuration failure.
- Repaired fast HumanEval pairing without overwriting old files. New
  `paired_vs_matched_baseline_v3` outputs pair 128-step methods to the 42/164
  baseline: symmetric-fast 52/164 (p=0.12145), directed-fast 51/164
  (p=0.18774). At 64 steps the baseline is 28/164, symmetric-fast 14/164, and
  directed-fast 18/164. Prompt and target hashes match in every pair.
- Added versioned MBPP paired audits against original LLaDA 89/500: symmetric
  121/500 (p=0.00244, mean NFE 147.89) and directed 123/500 (p=0.00169, mean
  NFE 149.99). These remain accuracy-first results.
- Added `docs/localleap_full_experiment_report_20260717.md`, separating all
  formal local audits, older pilots, implemented-but-unrun methods, evaluator
  failures, and paper-reported Prism/SOAR/OTS/FiRe/DiffCodeGen values.
- Updated the response-credit queue for the two-GPU machine. Paired arms run as
  independent single-GPU evaluators under `CUDA_VISIBLE_DEVICES=0/1`; manifest
  writes use `flock`. Bash syntax, source diff checks, GPU isolation, selector,
  auditor, profile, response-credit, and synthetic HumanEval evaluator tests
  pass. No generation queue was launched after the requested cancellation.

## Earlier (summary)

- Old “recover gold from past logits / direct response timing” directions failed on math.
- MATH/HumanEval ceiling scripts had metric bugs (see `cursorfeedback/0712grokproblem.md`).
- top-k distance support-index bug was fixed in `src/distribution.py` (tests cover it).
