# LocalLeap / LLaDA experiment audit and method genealogy

**Living report, updated 2026-07-20 00:10 Asia/Shanghai.**

This report separates four kinds of evidence:

1. **Current-server formal results** generated on the present dual RTX 4080
   SUPER host with frozen code and full record-level audits.
2. **Historical locally audited results** that remain useful, but may differ in
   hardware, source revision, prompt, few-shot count, or evaluator.
3. **Development / gate results** used to accept or reject an idea. They are not
   substitutes for full benchmark results.
4. **Paper-reported results**, included only as external references and never
   merged with local paired statistics.

The final deliverable must be one decoding algorithm shared by HumanEval,
MATH-500, GSM8K, and MBPP. Per-task routing, post-hoc ensembles, hidden-test
selection, and choosing a different method for each benchmark are forbidden.

## 1. Frozen local protocol

The current local comparison protocol is:

| Field | Value |
| --- | --- |
| Checkpoint | local LLaDA-8B-Instruct |
| Model path | `/root/autodl-tmp/model/LLaDA/instruct` |
| Remasking baseline | original low-confidence LLaDA |
| Generation length | 256 |
| Global diffusion steps | 128 |
| Block length | 32 |
| Temperature | deterministic / 0 |
| Seeds | `0,1234,1234,1234` |
| Few-shot | 0 for all four tasks in current full4/fair runs |
| MATH evaluator | Prism-aligned prompt and normalization |
| GSM8K evaluator | `exact_match,flexible-extract` |
| HumanEval evaluator | frozen sanitizer plus execution |
| MBPP evaluator | prompt-visible assertions plus independent execution crosscheck |

Every formal pair must match stable IDs, raw prompt hashes, target hashes,
actual chat-rendered input-text hashes, actual input-token-ID hashes, checkpoint
weights, source hashes, generation settings, seed, evaluator version, and
dataset view. Results from the previous server are not formal comparators for
the current dual-GPU host.

## 2. Current best unified method

The trusted accuracy family is the v11 public-guard trajectory method at
`tau=0.004`. v15 is an admissible early-abort implementation that preserves the
same selected output while avoiding conservative-path work that cannot change
the final decision.

Unified v15 profile:

`trajectory_early_lazy_confirmed_public_guard`

Conceptually it contains one coherent decoder, not a task ensemble:

- a fast symmetric attention trajectory;
- a conservative symmetric attention trajectory;
- vertical commit-path evidence accumulated during denoising;
- bidirectional full-draft verification only when the paths disagree;
- a strict prompt-visible public-check guard when such checks exist;
- an admissible upper bound that stops an accuracy trajectory once it cannot
  clear the inherited evidence gate.

Gold answers, normalized targets, correctness labels, and hidden tests are not
inputs to generation or selection.

### 2.1 Exact-output acceleration evidence

| Gate | Accuracy parent | v15 | Parent NFE | v15 NFE | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| HumanEval 164 | 58/164 | 58/164 | 67,882 | 43,147 | 36.44% |
| MATH-500 first 50 | 16/50 | 16/50 | 14,501 | 11,907 | 17.89% |
| GSM8K first 64 | 44/64 | 44/64 | 18,295 | 12,988 | 29.01% |
| MBPP first 100 | 40/100 | 40/100 | 41,475 | 32,731 | 21.08% |

All four gates were output-exact, correctness-exact, identity-aligned, and had
per-record NFE nonincrease.

### 2.2 Current full4 results

| Task | Unified v15 | NFE | Current-server baseline | Status |
| --- | ---: | ---: | ---: | --- |
| HumanEval | 58/164 = 35.37% | 43,147 | fresh fair baseline queued | method complete |
| MATH-500 | 167/500 = 33.40% | 119,799 | generating | method complete |
| GSM8K | in progress | in progress | generating in parallel | active |
| MBPP | queued | queued | queued | pending |

The MATH result has 500/500 records, zero duplicate IDs, zero prompt/generation
mismatches, zero residual masks, and per-example NFE 129--303. It must not be
called a formal gain until the current-server baseline and paired audit finish.

## 3. Historical local baselines and why labels matter

### 3.1 Audited historical full results

| Method / evaluator | HumanEval | MATH-500 | GSM8K | MBPP |
| --- | ---: | ---: | ---: | ---: |
| Original LLaDA, corrected local evaluators | 42/164 (25.61%) | 152/500 (30.40%) | 905/1319 (68.61%) | 104/500 (20.80%) |
| Symmetric-fast, corrected local evaluators | 52/164 (31.71%) | 150/500 (30.00%) | 901/1319 (68.31%) | 115/500 (23.00%) |

These rows do **not** prove a unified symmetric-fast win: it loses two examples
on MATH and four on GSM8K. They also contain source/hardware history and are
therefore descriptive, not the final current-server fair pair.

The original MBPP run was initially reported as 89/500 using the native
lm-eval packaging. Re-evaluation of the identical generations with fenced-code
extraction and prompt-visible assertions gives 104/500. Symmetric-fast changes
from 99/500 to 115/500 under the same corrected evaluator.

### 3.2 Reproduction drift on the present host

The same named low-confidence baseline is not bit-exact across the old and new
servers:

- MATH first 50: 24 decoded outputs differ and five correctness values flip;
  the old 16/50 becomes 13/50 on the current host.
- GSM8K first 64: 46 outputs differ and eight correctness values flip; the old
  42/64 becomes 40/64.

Repeated runs on the current host are internally stable. The likely causes are
the GPU/driver/kernel change, BF16 plus forced Flash Attention, and decoding
source revision. Iterative diffusion argmax decisions amplify small numerical
differences. This is why the fair queue regenerates the baseline on the same
runtime rather than pairing against old JSON by position.

## 4. Evaluator discrepancies already proven

Evaluator choice, not only decoder quality, caused several apparently large
reproduction gaps.

### HumanEval

- Native lm-eval artifacts report only 4--6/164 on identical generations.
- The audited sanitizer/executor reports 42/164.
- 120/164 responses contain fenced code; 36 audit-correct/native-wrong examples
  are recovered from fenced or full-function responses.
- Concatenating the original prompt with a response that already contains a
  complete function can create invalid duplicate definitions.

### MBPP

- 158/500 generations contain fenced code.
- On the same first 50 generations, native lm-eval passes 1/50 while the
  prompt-visible assertion evaluator passes 18/50.
- Prompt assertions are public model input, not hidden tests. The independent
  executor confirms every specialized-evaluator decision.

### GSM8K

- Strict `####` matching gives 1/1319 on the historical generation file.
- Flexible final-number extraction gives 905/1319.
- Current formal reporting uses one frozen flexible extractor for both arms.

### MATH-500

The v2 queue uses Prism's official MATH-500 system prompt and normalization.
The initial v1 queue used a non-aligned prompt/evaluator and was stopped during
its first full run. Its partial outputs are invalid and are never resumed or
repaired.

## 5. Method genealogy and decisions

The work deliberately evolved from simple one-change descendants. Negative
results are retained because they constrain the next design.

| Family / idea | Insight and implementation | Decision |
| --- | --- | --- |
| Original low-confidence | Fixed compute reference, no attention selector | formal baseline |
| Symmetric attention `tau=.004` | Horizontal attention dependency discourages simultaneous commits of strongly coupled tokens | retained accuracy parent |
| Symmetric-fast | Prune stable conflicts and fill the native budget | useful fixed-NFE reference; not unified non-regressive |
| Response credit / revision margin | Use longitudinal invalidation history to rank later commits | no reliable cross-task gain; pruned |
| Risk switch | Spend conservative work only under low confidence | insufficient accuracy evidence; superseded |
| Trajectory likelihood | Choose complete fast/accuracy paths using accumulated commit likelihood | exposed useful vertical evidence, but mean likelihood alone was unreliable |
| Block evidence | Require one-nat-per-block evidence before a path switch | improved structural calibration; retained as ancestor |
| Disagreement evidence | Score only positions where complete drafts differ | avoided shared-token dilution; not a standalone winner |
| Consensus / lazy consensus | Ask original LLaDA to break strong trajectory disagreements, lazily when needed | accuracy useful but expensive; refined further |
| Coverage / convergent coverage | Require revision coverage and convergence before switching | no stable full winner; rejected/superseded |
| Shared skeleton | Preserve common tokens and verify only differing blocks | theoretically sound, but early gates did not justify full promotion |
| Bidirectional block verification | Score each disagreement block under both external full-draft contexts | retained as robust horizontal verifier |
| Confirmed bidirectional | A path-time evidence gate and full-draft verifier must favor the same path | v9 cross-task parent |
| Public guard v11 | Original trajectory may override only by strictly more prompt-visible checks | best code accuracy parent, HE 58/164 and MBPP 40/100 |
| Lazy public guard v12 | Skip original trajectory when the public-check decision is already fixed | MBPP exact to v11, NFE 41,475 to 37,763, wall 1.083x |
| Admissible early abort v15 | Stop the accuracy path when even zero-loss future commits cannot clear the evidence gate | exact-output speed descendant; current full4 method |
| Localized evidence conflict repair v18 | On evidence/verifier disagreement, preserve common tokens, re-denoise one strongest opposing block, then require the same verifier/public guard | preregistered, waiting for full4 |

### Explicitly rejected or invalid branches

- Public-frontier guard v3: HumanEval development 17/32 versus v11 18/32,
  zero gains and one loss; holdout was not opened.
- Dependency threshold `.002` alone: 17/32 versus 18/32 and NFE increased
  11.5%; rejected before holdout.
- Threshold-diversity hybrid: development 20/32, but preregistered holdout tied
  15/68 with zero paired gains/losses. Its exploratory 61/164 full score does
  not override the failed gate.
- Outcome arbiter v14: MATH 3/16 versus parent 6/16 with zero gains/three
  losses and NFE +60.9%; GSM tied 9/16 with NFE +27.9%. Rejected.
- v13: no accuracy gain; stopped rather than extending the queue.
- Public-frontier v2: 0/32 came from evaluator spawn-path precedence and is
  labeled **INVALID**, not an algorithm failure.
- Initial long queue v1: non-Prism MATH prompt/evaluator; labeled invalid and
  never resumed.

## 6. Information-leakage audit

Static audits find no gold answer, normalized target, correctness label,
reference solution, or hidden test passed into `generate` or the trajectory
selector. The Instruct wrapper constructs the model input only from
`req.args[0]` and the tokenizer chat template.

Legacy leakage auditor v1 had a schema error: it rejected `raw_gold`,
`normalized_gold`, and `correct=None` even when the evaluation wrapper appended
them only after the decoder returned. Version 2 now:

- audits the generation/selector AST and the actual `generate` call inputs;
- permits those three fields only at trace top level;
- requires `correct` to remain `None` there;
- recursively forbids answer/correctness keys inside selector diagnostics;
- requires `uses_hidden_tests=false` and `uses_reference_solution=false`;
- separately freezes actual chat-rendered text and token-ID hashes.

The original v1 failure artifact will be preserved. A versioned recovery may
add `RECOVERED_BY_LEAKAGE_V2` and `DONE` only if all seven generation stages are
complete and every failure stage is exclusively `leakage_*`.

## 7. Paper-reported reference values

These are quotations of paper tables, not local reproductions.

### Order-Token Search (arXiv:2601.20339)

LLaDA-8B-Instruct, sequence length 256, steps 128, block 32:

| Paper method | GSM8K | MATH-500 | HumanEval |
| --- | ---: | ---: | ---: |
| Low-confidence baseline | 76.7 | 32.4 | 26.2 |
| Order-Token Search | 79.8 | 36.0 | 34.2 |

OTS jointly expands generation orders and token values, then prunes with an
incremental denoising-action likelihood. Its reported temperature is selected
from a task/length search, so it is not a deterministic local reproduction.

### Prism (arXiv:2602.01842)

Prism uses LLaDA-8B-Instruct, temperature 0.7, 32 steps **per block**, and
generation/NFE 256 for math tasks and 512 for code tasks.

| Paper method | GSM8K | MATH-500 | HumanEval | MBPP |
| --- | ---: | ---: | ---: | ---: |
| LLaDA N=1 | 67.58 | 26.40 | 54.88 | 21.80 |
| Prism K=2 | 74.24 | 30.16 | 71.34 | 29.40 |
| Prism K=4 | 75.30 | 37.70 | 76.19 | 32.40 |
| Prism K=8 | 85.30 | 42.80 | 79.27 | 38.20 |

Prism code N=1 uses length/NFE 512, four times the current local code NFE 128.
K>1 also adds multiple trajectories and verifier calls. Accuracy should be
compared together with NFE, not as an equal-compute row.

### SOAR (arXiv:2602.10953)

SOAR's main comparison uses LLaDA-8B-Base, not the local Instruct checkpoint,
and uses 0-shot HumanEval, 3-shot MBPP, and 4-shot GSM8K.

| Paper method | HumanEval 256/512 | MBPP 256/512 | GSM8K 256/512 | Mean speedup |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 32.3 / 32.9 | 40.8 / 39.2 | 70.4 / 70.9 | 1.00x |
| Adaptive parallel | 32.3 / 32.9 | 40.8 / 39.2 | 70.4 / 71.0 | 2.19x |
| SOAR | 32.9 / 39.0 | 40.8 / 39.4 | 71.3 / 71.5 | 1.62x |

The Base/Instruct checkpoint and few-shot differences prevent direct absolute
accuracy claims.

## 8. Why the local and paper baselines differ

The audit attributes the differences to multiple independent factors:

1. Native versus corrected code extraction/execution.
2. Strict versus flexible numeric answer extraction.
3. Global 128 steps locally versus Prism's 32 steps per block.
4. Code generation length 256 locally versus Prism's 512.
5. Temperature 0 locally versus Prism 0.7 and OTS temperature search.
6. Zero-shot local full4 versus historical 3/4-shot settings.
7. Instruct checkpoint locally versus SOAR's Base checkpoint.
8. Local wrapper chat templating versus repositories that encode raw prompts.
9. GPU/driver/Flash-Attention/BF16 and source-revision drift.
10. Sampling variance in early 32/50/64-example development slices.

The local MATH 30.4% and GSM 68.61% historical baselines are not obviously
abnormal relative to paper protocols. The largest apparent mismatch was in
code, where evaluator packaging and the 256-versus-512 length/NFE difference
dominate.

## 9. Active offline queue chain

1. `best_framework_full4_20260719_v1`: finish v15 and current-server baseline
   generation on all four benchmarks.
2. `full4_leakage_recovery_20260719_v2`: versioned evaluator recovery only if
   the legacy schema is the sole failure.
3. `early_localized_evidence_conflict_repair_20260719_v2`: v18 code gate,
   cross-task development gate, then full promotion only on strict global
   non-regression plus aggregate gain.
4. `fair_three_arm_reproduction_20260719_v1`: current-runtime baseline,
   globally selected accuracy method, and symmetric-fast confirmatory arm.

The fair queue is fully offline. It hashes the six model shards, tokenizer and
configuration files, runtime sources, exact dataset view, raw prompt, rendered
chat input, token IDs, target, stable IDs, and evaluator outputs. The fast arm
must be non-regressive on every task, strictly better in aggregate accuracy,
non-increasing in per-task NFE, and strictly lower in aggregate NFE and wall
time. Otherwise it is rejected, not selectively used on favorable tasks.

## 10. Cached primary sources

The remote server has offline copies of:

- Prism commit `21466a8dcb582ffafc1d218ff74e2c31b664c152`;
- LLaDA commit `9182493720ed723ef8031210d85959364e51cbe0`;
- SOAR commit `ec3eb400e41a43dc05db20a49c0219b9a968d28e`;
- OTS PDF SHA-256
  `3e6ab34eee9f1dec2543a9d7b4765e45f072b0895edb1349178e0eea32a6905e`.

Primary links:

- <https://github.com/ML-GSAI/LLaDA>
- <https://github.com/viiika/Prism>
- <https://github.com/duterscmy/SOAR>
- <https://arxiv.org/abs/2601.20339>

## 11. Pending completion evidence

The project is not complete until all of the following exist and pass:

- four current-server baseline summaries;
- four accuracy-arm summaries;
- four fast-arm summaries;
- stable-ID, prompt, target, rendered-input and token-ID equality for every pair;
- full MBPP independent execution crosschecks;
- per-task accuracy, NFE, wall-time and throughput comparisons;
- zero residual masks and no missing/duplicate identities;
- strict information-leakage reports for every selected run;
- one final globally selected profile with no task routing.

Until then, v15/v11 remains the trusted unified family and later candidates are
reported as pending or rejected rather than promoted by partial accuracy.
