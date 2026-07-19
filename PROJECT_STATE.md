# Project State

Date: 2026-07-17

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

The global top-K retention decoder was formally negative on HumanEval-256
(64/164). Its parent-preserving v2.1 correction recovered 71/164 at 256 steps,
but the fixed-budget 128-step result was only 47/164, below the best
`symmetric_fast` parent result of 52/164. The v2.1 queue was therefore cancelled
without overwriting its completed or partial outputs. Top-K retention is not a
new parent for further development.

The active benchmark direction returns to the strongest verified architecture:
original LLaDA is always the baseline, symmetric attention with tau `0.004` is
the accuracy-first parent, and `symmetric_fast` is its fixed-budget speed child.
The long queue `best_symmetric_long_20260716_v2` adds an offline-cached MATH-500
task, full sample-level and paired audits, and speed/NFE summaries. It prioritizes
MATH-500, MBPP and GSM8K at generation length 256 / 128 steps, then runs
four-shot and length-512 robustness arms. Recent paper rows from Prism, SOAR and
Order-Token Search are stored as paper-reported references in
`docs/training_free_dlm_baseline_alignment_20260716.md`; they are not labeled as
local reproductions.

On 2026-07-17 the old heartbeat monitor was deleted at the user's request and
the new SSH endpoint was converted from one-time password authentication to the
existing dedicated public key. The mounted experiment data survived the host
change. The new machine has two RTX 4080 SUPER GPUs with 32,760 MiB each; no
generation job is currently active.

The completed flexible-extract GSM8K audits are negative for the attention
method: original LLaDA is 905/1319, symmetric-fast is 901/1319 at matched
128-step NFE, and accuracy-first symmetric is 865/1319 at 1.179x NFE. MATH-500
is likewise 152/500 baseline, 150/500 symmetric-fast, and 152/500 symmetric at
1.203x NFE. These results prevent promoting the HumanEval/MBPP improvement as
a universal new baseline.

Several old HumanEval fast paired files had incorrectly joined 128/64-step
methods to the 256-step 67/164 baseline. New immutable
`paired_vs_matched_baseline_v3` outputs correct this: at 128 steps baseline is
42/164, symmetric-fast is 52/164 (p=0.12145), and directed-fast is 51/164
(p=0.18774); at 64 steps baseline is 28/164, symmetric-fast is 14/164, and
directed-fast is 18/164. The old pair summaries are retained but invalid for
matched-budget claims.

The response-credit queue script now supports two independent evaluators in
parallel, one per physical GPU, with locked manifest writes. It deliberately
does not use DDP or unverified tensor parallelism because the 8B model fits on
one card and the custom attention hooks expect a single logical CUDA device.
No queue was launched after this code change. The full audited report is
`docs/localleap_full_experiment_report_20260717.md`.

## 2026-07-19 rapid dual-GPU iteration

The original LLaDA low-confidence decoder at 128 steps remains the formal
baseline. The confirmed bidirectional block selector (`tau=0.004`, v9) is now
formally positive on HumanEval: 55/164 versus 49/164 for symmetric-fast and
42/164 for original LLaDA. On the preregistered untouched indices 96--163 it
is 14/68 versus 11/68 for both controls. Full HumanEval pairing against
symmetric-fast has six method-only cases, zero fast-only cases, and exact
McNemar `p=0.03125`. The cost is 46,890 NFE versus 20,992 for original LLaDA.

The current accuracy best is v11: preserve v9 unless original LLaDA passes
strictly more tests already visible in the prompt. Its frozen replay and an
independent official execution both give 58/164 on HumanEval, with three
recoveries and zero losses versus v9. The untouched 96--163 result is 15/68
versus v9's 14/68 and original LLaDA's 11/68. The same public-assertion guard
was separately formal on MBPP indices 100--199: 29/100 versus 20/100 for its
symmetric parent and 16/100 for original LLaDA, with nine recoveries and zero
losses. Integrated v9+guard MBPP execution remains queued.

MATH-50 promotion passed: v9 is 16/50, symmetric-fast is 15/50 with one
method-only and zero fast-only example, and original LLaDA is 13/50. V9 uses
14,501 NFE versus 6,400 for each control. The frozen parent queue is currently
running GSM8K-64 and MBPP-50 concurrently on the two GPUs:
`/root/autodl-tmp/LocalLeap/llada/results/experiment_queues/confirmed_bidirectional_rapid_20260719_v1`.

Two ordered follow-up queues are submitted. V12 is an exact-output speed child
that skips the third trajectory when the v9 parent already exhausts all
prompt-visible checks; theoretical skip coverage is 50% on HumanEval dev32 and
20% on formal MBPP100. V13 tests one additional accuracy hypothesis only after
v12: on an incomplete public-check tie between two executable drafts, the same
bidirectional block verifier compares both complete drafts. Mean and
block-Pareto variants run in parallel on HumanEval 0--31 and may access the
preregistered 96--163 holdout only after strict development improvement.
Queue roots are:

- `/root/autodl-tmp/LocalLeap/llada_slot_lazy_public_guard/results/experiment_queues/lazy_public_guard_exact_20260719_v1`
- `/root/autodl-tmp/LocalLeap/llada_slot_public_verifier/results/experiment_queues/public_full_draft_rapid_20260719_v1`
- `/root/autodl-tmp/LocalLeap/llada_slot_outcome_arbiter/results/experiment_queues/outcome_arbiter_rapid_20260719_v1`

Relevant commits are `73aebeb`, `bfcad98`, `54e8e01`, `310258b`, and
`f98f080`. The outcome-arbiter implementation and queue are `fc20ff8` and
`d23172f`: only when complete attention trajectories disagree, one additional
normal dLLM pass sees the de-duplicated candidate-answer set and may select only
an existing complete trajectory. It is preregistered on MATH/GSM development
subsets before separate holdouts. Frozen sources must never be edited, and
development/formal records must remain separate.

The v9 cross-task arms are now complete. GSM8K-64 is 44/64 for v9,
43/64 for symmetric-fast, and 40/64 for original LLaDA. Against original
LLaDA, v9 has eight method-only and four baseline-only examples (+6.25 points)
at 18,295 versus 8,192 NFE. MBPP required its task-specific execution channel:
the generic lm-eval aggregate incorrectly reported zero. The immutable
`mbpp_prompt_assertion_execution_v2` audit independently executes the current
prompt assertions by two code paths and gives v9 16/50, symmetric-fast 14/50,
and original LLaDA 18/50. All IDs, prompt/target hashes, generations, NFE and
residual-mask checks pass. Thus v9 is positive on HumanEval, MATH-50 and
GSM8K-64 but not a universal MBPP winner; v11 remains the accuracy candidate
because its guard preserves complete trajectories and had already produced
nine MBPP recoveries without a loss on a separate 100-task slice.

Evaluator commits `5806899` and `88b60af` add versioned baseline-compatible
MBPP sample-log auditing and queued v11/v12 post-hoc execution comparisons.
The v15 speed child in commit `e3b67c1` combines v12's lazy public guard with
an admissible longitudinal early stop: while generating the accuracy path,
uncommitted tokens receive the optimistic maximum log-probability zero; if
even that bound cannot clear the inherited one-nat path gate, the remaining
path and verifier are skipped. It adds no threshold and must reproduce v11
exactly before any speed claim. Its queue waits for v14, then screens
HumanEval-32 and GSM8K-64 on both GPUs before HumanEval-164, MATH-50 and
MBPP-100. The preregistered queue root is
`/root/autodl-tmp/LocalLeap/llada_slot_admissible_lazy_guard/results/experiment_queues/admissible_lazy_guard_20260719_v1`.

## Deprecated (cleaned 2026-07-13)

Trajectory / lateral-response / agreement / ceiling-bug evals were removed from `results/` (local + AutoDL `dlm-seq-flow`).  
Audit notes kept under `cursorfeedback/`. Archived narrative docs under `cursorfeedback/archive/`.

## Kept baselines in this repo

- `results/round0_baseline/gsm8k_lcr.json` — LCR GSM8K 62%
- `results/round_block32_*` — region decoding positive evidence
- `experiments/localleap/` — LocalLeap scripts + protocol (mirror of AutoDL)
