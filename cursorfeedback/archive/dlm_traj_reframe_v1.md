# DLM Trajectory Reframe v1

Date: 2026-07-10

Remote project: `/root/autodl-tmp/dlm-seq-flow`

Arbor session: `.arbor/sessions/dlm-traj-reframe-v1/`

## 1. Why We Need To Reframe

The previous direction treated denoising trajectories as if the correct answer was already present somewhere in the intermediate distributions, waiting for a better sampler to recover it. Current evidence does not support that assumption.

- GSM8K-50: LCR accuracy is 62%, but among the 19 wrong samples the gold answer token was never the answer-region argmax at any denoising step.
- MATH-50 dev: LCR accuracy is 48%, and the same ceiling check is also 0%.
- Round 2 trajectory samplers reduced GSM8K accuracy from 62% to 58%.
- Answer-region flip count can detect some wrong answers, but deterministic regeneration produced the same outputs and gave no gain.

So the issue is not simply that we need a better way to fuse old logits. In these math tasks, many wrong answers are stable wrong answers. The model is not visibly moving through the correct answer and then losing it.

## 2. Keep The Core Idea, Change The Question

The original idea is still valuable:

- Longitudinal view: for one position, the distribution over time records how candidates settle, change, or get rewritten.
- Lateral view: after one token is committed, other still-masked positions may change, revealing dependency between positions.

But we should not jump from this idea directly to a sampler. First we should ask what the trajectory means.

Old question:

> Can we recover a better token by combining or selecting old distributions?

New question:

> What does a distribution change mean: uncertainty at that position, or response to newly visible context?

This keeps the time-axis view, but avoids assuming that the right answer is already present.

## 3. Current Hypothesis State

| Hypothesis | Status | What It Means Now |
|---|---:|---|
| Intermediate distributions contain useful state | Still plausible | Flip and trajectory changes are not random, but their role may be diagnosis or timing rather than direct correction. |
| Correct candidates can be recovered from history | Mostly false on math | Ceiling is 0% on GSM8K and MATH wrong samples. |
| Stability alone means correctness | False | Wrong arithmetic can be stable from early steps. |
| Lateral coupling is negligible | Reopened | The old top-k distance implementation used probability values as support indices, so the near-zero coupling result is not reliable. |
| Timing and grouping may matter | Open | This is the nearest algorithmic target, but only after better observation. |

## 4. Arbor Probe

Q1 First principles: the bottleneck is wrong action space. Evidence: ceiling=0% on GSM8K and MATH; Round 2 methods hurt baseline.

Q2 Hidden assumption: the correct candidate appears in the trajectory. If we drop it, the trajectory should be used to decide timing, dependency, or confidence, not direct recovery.

Q3 Elephant: stable wrong reasoning can look clean. A sampler that rewards stability may make this worse.

Q4 Hamming: no. This is not a one-knob fix because the failure is about what the trajectory is used for.

## 5. Next Observations Before Algorithms

### 5.1 Longitudinal Check

Pick a small set of correct and wrong GSM8K/MATH cases. For answer and reasoning positions, visualize top candidates over denoising steps.

We should separate three cases:

- stable correct: the same candidate stays and is finally correct;
- stable wrong: the same wrong candidate stays;
- rewritten: a candidate changes after surrounding tokens become visible.

Expected figure labels and captions must be in English.

Suggested caption:

> **Longitudinal token trajectories.** Top candidates for a fixed position across denoising steps. Stable wrong cases show that temporal stability alone is not a correctness signal.

### 5.2 Lateral Check

Pick important commit events: numbers, operators, final answer tokens, or line-structure tokens in code. Compare other positions just before and just after the commit.

The question is not whether every pair has high coupling. The question is whether a few meaningful commits rewrite a local group of positions.

Update from the first smoke run: `_topk_l1` and `kl_divergence` used top-k probability values instead of top-k token ids when building the comparison support. This made distribution changes look almost zero. After fixing the support, top response events in a 4-sample GSM8K smoke run moved from about `1e-4` to about `1.4-1.9`. Therefore the old "coupling is nearly absent" conclusion should be treated as an implementation artifact until rerun.

Update from an 8-sample fixed smoke run: mean top response delta is `1.63`, median top response distance is `1` token. This supports a simple lateral reading: strong responses often occur near recently committed tokens. It does not prove correctness gains yet, but it does support the claim that some positions should not be treated as fully independent within the same step.

Suggested caption:

> **Response after token commitment.** Distribution changes at still-masked positions before and after a selected token is committed.

### 5.3 Timing Check

Only after the two checks above, test a simple training-free rule:

- delay positions that are likely to be rewritten by nearby unresolved tokens;
- commit groups of positions whose distributions move together;
- avoid using old logits as extra votes for the final token.

Suggested caption:

> **Commit timing comparison.** Accuracy of standard LCR and trajectory-aware timing on the development split.

## 6. What Not To Do Next

- Do not add more scalar features before we know what the trajectory change means.
- Do not use top-k stability as a correctness signal without a counter to stable wrong cases.
- Do not report only accuracy; show at least one longitudinal plot and one lateral response plot.
- Do not use B_test during iteration.

## 7. Current Arbor Nodes

1. Longitudinal trajectory audit: distinguish internal uncertainty from context rewriting.
2. Lateral response audit: inspect local distribution changes after meaningful commits.
3. Decision timing: later design a training-free timing or grouping rule, not old-logit fusion.

## 8. Smoke Artifacts

- Observation script: `scripts/run_reframe_observation.py`
- Pre-fix smoke: `results/reframe_observation_smoke/`
- Fixed smoke: `results/reframe_observation_smoke_fixed/`
- 8-sample fixed smoke: `results/reframe_observation_8_fixed/`
- Distance fix: `src/distribution.py`

## 9. Method Iteration Update

The first lateral timing rules were useful as tests but not as final methods.

| Method | Evidence | Reading |
|---|---:|---|
| `lcr_spaced` | GSM8K-20: 40% vs LCR 50% | Blanket spacing breaks useful adjacent commits. |
| `lcr_spaced_early` | GSM8K-20: 45% vs LCR 50% | Early-only spacing reduces damage but still underperforms. |
| `lcr_response` w=0.6 | GSM8K-100: 53% vs LCR 60% | Broad response boost changes the path too often. |
| `lcr_response_strong` | GSM8K-100: 58% vs LCR 60% | Strong-response filtering reduces losses but still underperforms as a single path. |
| Selective response branch | GSM8K-100: oracle 64%; simple selector 62% at 1.2x cost | Response is more useful as a second branch than as a replacement for LCR. |
| Selective response branch | GSM8K-200: LCR 60.5%, strong-response branch 55.5%, oracle 64.0%, deployable two-stage selector 62.0% at 1.37x cost | The branch has real headroom, but the current selector is only mildly better than LCR. |

The current best direction is:

1. Run normal LCR.
2. Use trajectory diagnostics to decide whether the sample is risky enough to try a response branch.
3. If triggered, run the response branch and choose it only under a simple diagnostic rule.

This still matches the original idea: unused distributions are not fused into logits; they are used to decide whether another generation path is worth taking.

The next requirement is stronger branch quality or stronger branch diversity. The current response branch changes many predictions but still loses as a single path, so the selector has only 3.5 points of oracle headroom on GSM8K-200.

## 10. 2026-07-12 Update

The latest round tested whether local lateral response can directly decide commit timing. The answer is mostly no, at least for the simple rules tried so far.

| Method | Evidence | Reading |
|---|---:|---|
| `lcr_response_delay` w=0.6 | GSM8K-20: 45% vs LCR 50% | Strong delay over-changes the order. |
| `lcr_response_delay` w=0.3 | GSM8K-100: 58% vs LCR 60% | Weak delay still loses; mean delayed candidates are about 26.9 per sample. |
| `lcr_response_stability_delay` | GSM8K-20: 45% vs LCR 50% | Adding top-1 flip / instability does not fix the timing rule. |
| `lcr_rewrite_branch` | GSM8K-100: 55% | Rewriting a local suffix after a strong response does not generalize. |
| temperature 0.3 `lcr_probe` | GSM8K-100: 62% vs LCR 60%; GSM8K-200: 59.5% vs LCR 60.5% | Low-temperature diversity looked good on 100 samples but did not hold on 200. |
| base + temp03 + strong-response oracle | GSM8K-200: 65.5% | Branch diversity exists, but simple deployable selectors still reach only about 61.5%. |

The important positive result is not the accuracy of these direct methods. It is the diagnostic pattern:

- near-response remains consistently larger than far-response;
- wrong groups tend to show stronger response and larger selected-delta;
- candidate branches add oracle headroom even when their standalone accuracy is worse.

So the trajectory signal is real, but its role should be revised:

1. Lateral response should be treated as a sign that a local region was rewritten by context.
2. Longitudinal flips and confidence changes should be treated as instability evidence.
3. These signals should decide whether to branch, inspect, or reject a path.
4. They should not directly force many individual token delays unless the group structure is better defined.

The next algorithmic step should add evidence at the answer or local-region level. For GSM8K, this likely means saving generated text and using branch agreement / answer consistency together with trajectory risk. For the sampler itself, a better branch should regenerate a dependent local region under a different order, then select using trajectory stability. This stays within the original idea: the unused distributions are model state over time, not extra logits to average.

## 11. Answer-Evidence Selector Check

We tested the narrow claim that trajectory risk can decide *when* to inspect a second path, while decoded answer evidence decides *which* path to accept.

| Split | LCR | Response branch | Two-branch oracle | Fixed risk plus evidence rule |
|---|---:|---:|---:|---:|
| GSM8K 0-99 (discovery) | 60% | 58% | 64% | 63% |
| GSM8K 100-199 (holdout) | 61% | 53% | 64% | 61% |

The fixed rule used the discovery threshold `selected-response delta >= 0.30` and made 17 switches on holdout. It did not improve over LCR. The selector diagnostic figures show substantial overlap between base-correct, branch-only-correct, and both-wrong samples. Thus, the answer text can make a branch look more settled without making it more correct.

Suggested caption:

> **Trajectory risk and answer evidence across branch outcomes.** Branch-only-correct cases tend to have higher response risk, but their feature region overlaps strongly with base-correct and both-wrong cases; simple answer-settling evidence does not generalize as a branch selector.

This is a useful negative result. A stronger method needs a branch that preserves a dependency group differently, not merely a cleaner way to judge two globally different decoded answers.

## 12. Commit-Order Causal Checks

We next tested whether the temporal state can choose a better commit order inside a denoising step.

| Method | GSM8K evidence | Cost | Conclusion |
|---|---:|---:|---|
| Anchor then refresh | 35-40% on 20 samples vs LCR 50% | 81-88 vs 64 NFE | Frequent serial refresh damages the native parallel path. |
| Sparse hypothetical-commit lookahead | 59% on 100 samples vs LCR 60% | 68 vs 64 NFE | Immediate local confidence contraction does not choose a better anchor. |

The lookahead method was deliberately limited to two probes per sample. It temporarily committed each of the two LCR candidates, measured the mean maximum probability at nearby unresolved positions, selected the more contracting probe, and reused its forward pass for the remaining token. On 100 GSM8K samples it lost one LCR-correct example and recovered none.

This separates two claims that should not be conflated:

1. A commit can substantially rewrite nearby distributions.
2. The rewrite direction is a useful rule for changing the commit order.

Only the first is supported. The next method should avoid treating local certainty as a proxy for correctness or for the correct generation order.
