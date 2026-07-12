# Project State

Date: 2026-07-12

Remote worktree: `/root/autodl-tmp/dlm-seq-flow`

Goal: develop a training-free DLM inference method that uses denoising-time distribution trajectories as state, improves over baseline on many samples, and keeps extra cost modest.

## Current Best Understanding

- The old "recover the right token from past logits" direction is mostly wrong for math tasks.
- GSM8K and MATH ceiling checks showed that wrong answers often never have the gold answer token as answer-region argmax.
- The old lateral-coupling result was partly invalid because top-k distance used probability values as support indices instead of token ids.
- After fixing the distance support, commit events do cause strong local distribution changes.
- Directly changing the greedy path with response boosts usually hurts stable correct samples.
- Response-delay variants also hurt: local response is a useful signal, but directly delaying many positions changes the order too aggressively.
- The strongest current interpretation is:
  - lateral response detects when a position was rewritten by nearby commits;
  - longitudinal flips/stability help identify unstable regions;
  - these signals are better at deciding when to branch or reject a path than at choosing a token directly.
- The best current direction is selective branching:
  1. run normal LCR with cheap trajectory diagnostics (`lcr_probe`);
  2. trigger a response-based branch only for risky samples;
  3. choose between LCR and the branch with trajectory diagnostics.

## Current Evidence

- `results/round_reframe_response_delay_w03_100/`
  - LCR: 60.0%
  - weak response-delay: 58.0%
  - net change vs LCR: -2 samples
  - mean delayed candidates: 26.87 per sample
  - near response remained much larger than far response, so the lateral signal exists but the timing rule is wrong.
- `results/round_reframe_response_stability_delay_smoke20/`
  - LCR: 50.0%
  - stability-gated response-delay: 45.0%
  - adding a top-1 flip / instability gate reduced but did not remove the damage.
- `results/round_reframe_rewrite_suffix_100/`
  - local suffix rewrite branch: 55.0%
  - rewriting after a strong local response does not generalize.
- `results/round_reframe_temp03_probe_100/`
  - low-temperature `lcr_probe` at temperature 0.3: 62.0%
  - same-slice LCR: 60.0%
  - no LCR-correct sample was lost on this 100-sample slice, but this did not hold at 200 samples.
- `results/round_reframe_temp03_probe_200/`
  - low-temperature `lcr_probe` at temperature 0.3: 59.5%
  - LCR-probe baseline: 60.5%
  - base + temp03 + strong-response oracle: 65.5%
  - best simple deployable selector found so far: 61.5%
- `results/round_reframe_selective_branch_200/`
  - LCR: 60.5%
  - strong response branch: 55.5%
  - oracle over two branches: 64.0%
  - deployable two-stage selector: 62.0% at about 1.37x cost
- `results/round_reframe_selective_branch_200_fast/`
  - same 200-sample result after vectorized diagnostics:
  - LCR-probe: 60.5%
  - strong response branch: 55.5%
  - best two-stage selector: 62.0%
- `results/round_reframe_probe_speed_10/`
  - LCR and `lcr_probe` both 60% on 10 samples
  - runtime: LCR about 3.61s/sample, `lcr_probe` about 3.74s/sample
- Negative branch checks:
  - response weight 0.3 on GSM8K-200: branch 54.5%, oracle 63.0%, best selector 61.5%
  - response weight 1.0 on GSM8K-200: branch 51.5%, oracle 63.5%, no selector gain
  - temperature 0.3 on GSM8K-50: no oracle headroom
  - temperature 1.0 on GSM8K-50: branch 54%, oracle 64%, no selector gain

## Visual Evidence

- `results/round_reframe_response_delay_w03_100/diagnostics.json`
  - group-level diagnostics for delayed candidates and lateral locality.
- `results/round_reframe_response_delay_w03_100/method_comparison.png`
  - baseline vs delay summary plot.
- `results/round_reframe_temp03_probe_200/oracle_overlap_base_strong.json`
  - shows that extra branches create answer headroom even when individual branches are weaker.

## Running

- No long job is required to interpret the current state.

## Code Changes In Progress

- `src/distribution.py`: fixed top-k support in `_topk_l1` and `kl_divergence`.
- `src/samplers.py`: added `lcr_probe`, response-based branches, diagnostics, and vectorized top-k state recording.
- `src/runner.py`: writes diagnostics into per-sample result records.
- `scripts/analyze_*`: added diagnostics and branch selector analyses.
- `scripts/analyze_oracle_overlap.py`: reports oracle ceilings across many result files.
- `scripts/analyze_multi_branch_selector.py`: tests simple deployable multi-branch selectors.
- `tests/test_distribution.py`: regression test for top-k support.

## Next Gate

The method is not complete yet. Current best deployable result remains only +1.0 to +1.5 points on GSM8K-200, depending on selector and branch set. A stronger result should show a clear gain over LCR on at least 200 samples, preferably with a selector that uses only deployable diagnostics and an average cost near 1.1x-1.4x.

The next useful step is not another direct delay/boost rule. It should either:

1. add task-level evidence to branch selection, such as saving generated text and checking answer consistency, or
2. design a branch whose construction follows the high-level hypothesis more closely: generate dependent local regions in a different order, then select using trajectory stability rather than raw response magnitude.

## Answer-Evidence Selector Check (2026-07-12)

- Added decoded-answer records and minimal answer-settling evidence: final-number recurrence, final-line presence, answer marker, and number diversity in the final text segment.
- Discovery split (GSM8K indices 0-99): LCR 60%, strong response branch 58%, two-branch oracle 64%. A fixed candidate rule, `selected_response_delta >= 0.30` followed by answer-evidence comparison, reached 63%.
- Holdout split (GSM8K indices 100-199): LCR 61%, strong response branch 53%, oracle 64%. The same fixed rule stayed at 61% despite 17 switches.
- Conclusion: this is a negative generalization result. Trajectory response is still a useful risk signal, but answer-format evidence does not reliably identify the correct branch. Do not present the 63% discovery result as a method gain.
- Visual diagnostics, with English labels, are in `results/round_answer_evidence_100/selector_diagnostics.png` and `results/round_answer_evidence_holdout_100/selector_diagnostics.png`.

## Local Refresh and Causal-Probe Checks (2026-07-12)

- `lcr_response_refresh` tests direct serialization: after a high-response step, commit one anchor and refresh the remaining positions before committing the rest of the batch.
  - GSM8K-20: threshold 0.20 gives 35% at 88 NFE; 0.30 gives 40% at 83.65 NFE; 0.40 gives 40% at 81.3 NFE; LCR is 50% at 64 NFE.
  - Direct local serialization is therefore rejected: it perturbs the native parallel denoising path too often and harms accuracy.
- `lcr_response_lookahead` tests a sparse causal probe. At most twice per sample, it temporarily commits either of the two LCR candidates, measures local post-commit confidence, reuses the better probe state, and keeps the same token budget for the step.
  - GSM8K-20: 50% vs LCR 50%, with 68 vs 64 NFE.
  - GSM8K-100: 59% vs LCR 60%, with 68 vs 64 NFE. It lost one LCR-correct sample and recovered none.
  - Conclusion: the direction of immediate local confidence contraction is not a useful commit-order signal under this probe.
- `lcr_terminal_refine` remasks only the final four generated tokens when sample-level response risk is high. GSM8K-20 tied LCR at 50%, triggered on 45% of samples, changed only one token in total, and used 65.8 vs 64 average NFE. It does not provide evidence that a completed rationale makes the terminal answer revisable.
- A 32-token answer-verification prompt was tested on all 20 base outputs. It tied LCR at 50% but changed only one candidate and corrected none; the model mostly repeated the proposed numerical answer. Do not use candidate-conditioned short verification as a selector.

## Risk-Gated Independent Agreement (2026-07-12)

- Fixed policy: if base `response_selected_delta_mean >= 0.35`, generate two 64-token independent derivations with different prompts at temperature 0.3; replace LCR only if the two numerical answers agree. Otherwise keep LCR.
- GSM8K 0-99: 60% to 60%, average NFE 79.36.
- GSM8K 100-199: 61% to 65%, average NFE 74.88. Four accepted replacements were all base-wrong to correct.
- Combined GSM8K-200: 60.5% to 62.5% (+2.0 points), with 6 recovered and 2 newly lost samples; estimated average NFE is 77.12 (1.205x LCR).
- A third independently prompted branch, accepted by three-way numerical majority, did not improve the combined result (still 62.5%) and increased cost to about 1.31x. Keep only the two-branch version.
- Important metric fix: `extract_number` previously retained a terminal sentence period (for example `694.`), causing false negatives for integer answers. It now extracts canonical integers and decimals, with a regression test.

## Cross-Slice Replication and Joint Trajectory Checks (2026-07-12)

- The fixed lateral signal replicated from a 200-sample calibration slice to a new 400-sample slice. `response_selected_delta_mean` had error-ranking AUC 0.650 and 0.676; the fixed high-response group had 37.5% and 37.0% accuracy. The simple longitudinal answer-flip mean did not replicate (AUC 0.590 to 0.537).
- Risk-gated independent agreement failed on the new 400 samples: LCR 58.75%, two-way agreement 58.0%, third-confirmed agreement 58.5%. Across all 600 samples it was 59.33% to 59.50%, exact McNemar `p=1.0`. It is no longer the current method.
- `lcr_response_budget` reduced parallel commits after strong lateral response while keeping 64 NFE. On unseen GSM8K indices 800-839 it reached 47.5% vs LCR 52.5%; mean deferral was 22.7 tokens over 16.85 steps. Lateral response alone does not determine whether to commit or wait.
- `lcr_response_persistence` required a strong response to remain stable for another step before a small priority boost. It tied LCR at 52.5% with 64 NFE and changed only two wrong answers. Persistence is conservative but not yet corrective.
- The active test is `lcr_response_alignment`: at at most two high-risk steps, probe two candidate commits and prefer the one whose downstream distribution changes continue rather than reverse the existing trajectories. Its cost is capped at 68 NFE (1.0625x).
- Concise method framing is maintained in `docs/trajectory_reasoning_method.md`. Replication data and the English-labeled plot are in `results/trajectory_risk_replication/`.

## Region-Scale Dependency Check (2026-07-12)

- The sparse trajectory-alignment probe reached 50.0% vs LCR 52.5% on GSM8K 800-839 at 68 vs 64 NFE. It changed four predictions, recovered none, and lost one correct answer.
- A multi-block implementation bug was found and fixed: selection had not been restricted to the active block. Single-block results are unaffected. A regression test now verifies that all blocks finish without residual masks.
- With the corrected implementation, 32-token blocks reached 72.5% vs the 128-token single-block LCR at 52.5% on the same 40 samples, with identical 64 NFE. Paired changes were 10 candidate-only correct and 2 base-only correct; exact McNemar `p=0.0386`.
- This supports a region-scale interpretation: horizontal dependency should be ordered across reasoning regions, while vertical denoising updates candidates within each region.
- The 400-sample replication on GSM8K indices 400-799 confirmed the result: 71.25% for four 32-token blocks vs 58.75% for single-block LCR, both at 64 NFE. Paired outcomes were 80 recovered and 30 lost, a net gain of 50; exact McNemar `p=2.02e-6`.
- Fixed wavefront and response-decay wavefront are now running on GSM8K 800-839 as strong equal-NFE comparisons.
- Fixed wavefront reached 65.0%; the response-decay wavefront reached only 55.0%, while fixed 32-token blocks remained 72.5% on the same slice. Per-step response control is rejected; block-level adaptation remains open.
