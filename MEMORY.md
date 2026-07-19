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

## 2026-07-19 continued

- Completed the v9 GSM8K-64 arm: 44/64 versus symmetric-fast 43/64 and
  original LLaDA 40/64. The paired original comparison is 8 method-only / 4
  baseline-only at 18,295 versus 8,192 NFE; all audit alignment checks pass.
- Diagnosed MBPP's generic zero as the wrong evaluation channel, not zero
  executable programs. Added `mbpp_prompt_assertion_execution_v2`, which
  accepts immutable lm-eval samples even when original LLaDA has no attention
  trace, verifies sample/task generation and hashes, and crosschecks two
  independent assertion executors. Formal MBPP-50 is v9 16, fast 14, original
  18; v9 therefore loses two to the true baseline despite beating fast by two.
- Queued frozen post-hoc v11/v12 MBPP audits and stable-ID subset pairing;
  completed artifacts are never rewritten after evaluator changes.
- Added and queued the no-hyperparameter admissible-lazy speed child. It stops
  the slow trajectory only when an optimistic zero-logprob bound proves the
  existing one-nat selector gate unreachable, then applies the unchanged v12
  prompt-visible guard. The quick gate requires exact outputs and lower total
  NFE on HumanEval-32 plus GSM8K-64 before any full evaluation.
- While v11 MBPP-100 occupies GPU1, launched the preregistered v13
  HumanEval-32 mean and Pareto full-draft screens sequentially on the otherwise
  idle GPU0 in a separate versioned prefill root. This does not alter the
  formal queue or reveal its registered holdout.
- Completed the integrated v11 MBPP-100 assertion audit: 40/100. The aligned
  first 50 are 22 for v11, 16 for v9, and 18 for original LLaDA; v11 has no
  paired losses against either comparator. All audit health checks pass.
- Rejected v13 as an accuracy branch. Mean and Pareto each score 18/32,
  identical to v11 in correctness; Pareto is also generation-exact. It saves
  2,012 NFE (15.2%) but contributes no new accuracy beyond the simpler lazy
  guard, so its HumanEval-164 stage is skipped at the next safe boundary.
- Corrected the hidden MBPP challenge audit to exclude 39 records with no
  challenge tests and preserve imports through execution extraction. The 11
  eligible records give 1/11 for v9, fast, and original; these checks remain
  strictly post-hoc and are not tuning data.
- Fixed v15's undefined `block_length` runner contract before model loading,
  added an executable regression test, and committed it as `d07effc`. Its
  optional smoke was interrupted when v12 automatically took both GPUs; the
  formal v12 jobs were preserved to avoid same-GPU contention.

## 2026-07-20 strict offline queue

- Audited the future branch of the strict unified controller and found that
  implementation checked v18 before v19 even though preregistration gave v19
  priority. No generation had started. Preserved v3 with a
  `SUPERSEDED_BY_V4` marker, reversed the checks, and added a regression test.
- Built and detached strict v4 (PID 280312 at launch). All six protocol tests,
  shell contracts, JSON preregistration, socket-blocked four-dataset preflight,
  40-file offline manifest, six-shard model-weight hashes, and static leakage
  audit passed before it entered the v19 wait.
- Strict v4 will freshly regenerate original low-confidence, one globally
  selected candidate, and comparator-only symmetric-fast on all four complete
  benchmarks. It balances candidate GPU assignment, records actual runtime
  chat text/token hashes, requires stable-ID/prompt/target/evaluator/source
  equality, and never performs per-task method routing.
- Committed the priority fix as `2252c13` and the updated audit report as
  `4e33fea`; both are on GitHub main. The remote code, datasets, tokenizer,
  task/evaluator sources, reference papers/repos, and model checkpoint are
  sufficient for the registered chain to continue without Internet access.

## 2026-07-20 direct offline chain v5

- Removed one redundant provisional four-benchmark pass from the future
  scheduler. After full4 recovery, v18 now runs first; v19 is skipped on v18
  acceptance and runs unchanged only on a formal v18 rejection. This is a
  scheduling optimization, not an algorithm or gate change, and was frozen
  before the v18 result.
- Built the self-contained v19 direct slot and detached controller PID 284627.
  Its bootstrap, generic runner, and preregistration hashes pass while it waits
  for the v18 terminal decision.
- Built strict v5 and detached controller PID 284935. Six tests, shell and JSON
  contracts, socket-blocked offline loading of all four complete datasets, the
  40-file/10-Arrow manifest, six-shard model-weight hashes, and static leakage
  audit all passed before it entered the direct-v19 wait.
- Built full4 leakage recovery v3 and detached controller PID 285366. It
  verifies its frozen sources while waiting and resumes only v18.
- Safely stopped five pure-wait controllers (old full4 recovery v2,
  provisional fair recovery, old v19, v19 recovery v2, and strict v4) afte
  their replacements passed. Each old queue has a versioned supersession
  marker; active GSM/MBPP generation was not touched.
- Fresh current-host GSM8K baseline finished at 915/1319 with NFE 168,832,
  zero duplicate IDs and NFE exactly 128 per record. V15 GSM and MBPP remained
  active on separate GPUs at the handoff.
- Committed and pushed the direct scheduler and regression contract as
  `7553e99`; the five-minute monitor now points only at the v3/v18/direct-v19/
  strict-v5 chain.
- Completed the fresh same-host GSM8K-1319 pair. V15 scored 887 versus 915 for
  original low-confidence, with 130 method-only and 158 baseline-only records
  (exact McNemar `p=0.11145`). It used 262,029 versus 168,832 NFE (1.552x) and
  20,934 versus 12,424 seconds (1.684x). Stable IDs, prompt/target hashes and
  source hashes all pass; the method leakage-v2 audit has zero violations.
  This is an algorithm-level cross-task rejection, not an evaluator failure.
  V15 remains a code/speed parent but cannot be the final unified winner;
  v18/v19 must repair the GSM regression to be promoted.
- Added a stable-ID post-generation trace-feature auditor with two regression
  tests. It never reads gold answers or generated text and refuses to overwrite
  outputs. On GSM, the original candidate was generated 0/1319 times; accuracy
  and fast selection strata each have paired balance -14, while zero and
  nonzero disagreement strata are -17 and -11. The failure is candidate-family
  coverage rather than one tunable threshold, so no post-hoc selector sweep is
  justified.
