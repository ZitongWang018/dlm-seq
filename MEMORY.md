# Memory

## 2026-07-10

- Reframed the project away from direct old-logit fusion.
- Created Arbor session `dlm-traj-reframe-v1`.
- Added `docs/dlm_traj_reframe_v1.md` to summarize the new high-level hypothesis:
  - longitudinal trajectories show stability, swings, and context rewrites;
  - lateral responses show which positions are affected by committed tokens;
  - trajectory state should decide timing, branching, or selection, not simply vote old logits.
- Found and fixed a bug in `src/distribution.py`:
  - `_topk_l1` and `kl_divergence` used top-k probability values as support indices;
  - corrected both to use token ids.
- Added `tests/test_distribution.py` to catch this bug.
- Added observation script `scripts/run_reframe_observation.py`.
- Smoke observations showed:
  - old lateral response values around `1e-4` were artifacts;
  - fixed response deltas are often around `1.4-1.9`;
  - response positions are often near recently committed positions.
- Tried direct timing rules:
  - `lcr_spaced`: hurt GSM8K-20;
  - `lcr_spaced_early`: still hurt GSM8K-20.
- Tried response-boosted greedy branches:
  - unguarded response improved a small 20-sample slice but failed on 50 and 100 samples;
  - strong-response branch reduced damage but remained below LCR.
- Ran HumanEval ceiling:
  - HumanEval-30 pass@1 46.7%;
  - ceiling ratio 6.2%, still low.
- Added branch selector analyses:
  - on GSM8K-200, two branches had oracle 64.0% vs LCR 60.5%;
  - a simple deployable two-stage selector reached 62.0% at about 1.37x cost.
- Added `lcr_probe`, which records diagnostics while keeping LCR outputs unchanged.
- Vectorized top-k diagnostic collection in `src/samplers.py`.
  - Speed smoke on 10 samples: LCR about 3.61s/sample, `lcr_probe` about 3.74s/sample.
- Started fast 200-sample rerun:
  - `results/round_reframe_selective_branch_200_fast/`.

## 2026-07-12

- Tested response-delay as a timing policy rather than logit fusion.
  - `results/round_reframe_response_delay_smoke20/`: LCR 50%, delay 45%.
  - `results/round_reframe_response_delay_w03_100/`: LCR 60%, weak delay 58%.
  - Delay candidates were frequent, but almost never selected after penalty; the rule over-changed commit order.
- Added and tested `lcr_response_stability_delay`.
  - It delays only near-response positions whose top-1 changed and whose position had previous top-1 flips.
  - `results/round_reframe_response_stability_delay_smoke20/`: LCR 50%, stability-delay 45%.
  - Conclusion: simple single-position delay remains the wrong mechanism.
- Read `results/round_reframe_rewrite_suffix_100/`.
  - local suffix rewrite branch reached 55%, below LCR.
  - This argues against naive local rewrite after high lateral response.
- Added analysis scripts:
  - `scripts/analyze_oracle_overlap.py`
  - `scripts/analyze_multi_branch_selector.py`
- Low-temperature probe branch:
  - `results/round_reframe_temp03_probe_100/`: temperature 0.3 `lcr_probe` reached 62% vs LCR 60%.
  - `results/round_reframe_temp03_probe_200/`: temperature 0.3 fell to 59.5% vs LCR-probe 60.5%.
  - The 100-sample gain was not stable at 200 samples.
- Multi-branch oracle remains meaningful:
  - base + temp03 + strong-response oracle on 200 samples: 65.5%.
  - best simple deployable selector found in this round: 61.5%.
  - Current bottleneck is branch selection, not branch diversity.
- Updated high-level conclusion:
  - lateral response reliably shows local distribution rewriting;
  - wrong groups tend to have stronger near response / selected-delta;
  - these signals work better as instability/risk indicators than as direct token timing commands.

## 2026-07-12 Answer-Evidence Selector Result

- Added `answer_evidence` records, `scripts/analyze_answer_selector.py`, `scripts/repair_answer_records.py`, and `scripts/plot_answer_selector_diagnostics.py`.
- Discovery GSM8K 0-99: LCR 60%, response branch 58%, oracle 64%; risk-gated answer-evidence rule reached 63%.
- Disjoint GSM8K 100-199: LCR 61%, response branch 53%, oracle 64%; the fixed rule reached 61% with 17 switches.
- The negative holdout result is decisive for this iteration: answer-settling style is not enough to select the correct branch. Keep response as a risk/local-rewrite diagnostic, not a final selector.

## 2026-07-12 Local Refresh and Lookahead

- Added `lcr_response_refresh`: anchor commit followed by refreshed selection. On GSM8K-20 it lost sharply at all tested thresholds (35-40% vs LCR 50%) while increasing NFE to 81-88 from 64.
- Added `lcr_response_lookahead`: at most two temporary hypothetical commits per sample, selecting the one that produces stronger local post-commit confidence. GSM8K-20 tied LCR (50%) at 68 vs 64 NFE; GSM8K-100 lost 59% vs LCR 60%, with one base-only correct sample and no recovered samples.
- The current evidence rejects using immediate local response magnitude or local confidence contraction to change the commit order. Preserve these samplers as negative controls.

## 2026-07-12 Terminal Repair and Short Verification

- `lcr_terminal_refine` remasks the final four tokens only for high-risk samples. GSM8K-20 tied LCR (50%), triggered 45% of the time, changed only one token total, and cost 65.8 vs 64 NFE.
- A 32-token candidate-conditioned numerical verifier on the same 20 samples also tied LCR (50%), changed one answer, and corrected none. The model mostly echoes the proposed number instead of independently checking it.

## 2026-07-12 Risk-Gated Independent Agreement

- Fixed a GSM8K metric bug: `extract_number` treated a sentence-final integer such as `694.` as a different string from `694`. Added regression coverage.
- Two independent 64-token derivations, invoked only when base selected-response delta is at least 0.35 and accepted only on numerical agreement, gave GSM8K-200 62.5% vs LCR 60.5% at about 1.205x NFE. Split results were 60 to 60 on indices 0-99 and 61 to 65 on indices 100-199.
- Three-way agreement did not improve the 62.5% combined result and raised cost to about 1.31x. The two-way policy is the current best small-gain result, but it is not evidence of a large or statistically secure improvement.

## 2026-07-12 Cross-Slice Replication and Joint Trajectory Tests

- On independent 200/400-sample slices, lateral selected-response delta replicated as an error-risk signal (AUC 0.650/0.676), while answer-position flip mean did not (0.590/0.537).
- The earlier two-way agreement gain disappeared on a new 400-sample slice. Pooled over 600 samples it gained only one answer, with exact McNemar `p=1.0`; reject it as a reliable method.
- A zero-extra-NFE response-budget rule lost 47.5% vs LCR 52.5% on GSM8K 800-839. A response-persistence rule tied 52.5% and changed only two wrong answers.
- Reframed the joint hypothesis: a commit is a context intervention; lateral change identifies affected positions, and longitudinal evolution says whether that response persists or reverses. The active sparse alignment probe chooses the candidate whose induced response continues existing trajectories, capped at four extra NFE.
- Added `docs/trajectory_reasoning_method.md`, `scripts/analyze_trajectory_risk_replication.py`, and the English-labeled replication plot under `results/trajectory_risk_replication/`.

## 2026-07-12 Region-Scale Dependency Result

- A sparse response-alignment probe that chose hypothetical commits by whether their lateral effects continued prior longitudinal movement lost 50.0% vs LCR 52.5% at 68 NFE.
- Found and fixed a latent multi-block bug: token selection was global even when `block_length < gen_length`. Added an active-block assertion and a fake-model regression test; all earlier single-block evidence remains valid.
- Correct 32-token block decoding reached 72.5% vs 52.5% for 128-token single-block LCR on GSM8K 800-839, with the same 64 NFE. It recovered 10 and lost 2 samples (`p=0.0386`).
- New interpretation: the useful horizontal unit may be a reasoning region, not a token pair. Sequential blocks provide region-level context, while iterative distributions within each block provide the vertical reasoning loop.
- The 400-sample replication confirmed 71.25% for 32-token blocks vs 58.75% single-block LCR at the same 64 NFE. It recovered 80 and lost 30 samples, with exact McNemar `p=2.02e-6`.
- Added fixed wavefront and response-decay wavefront samplers. The latter narrows expansion when near-response rises and expands normally when it decays, using no extra model calls.
- On GSM8K 800-839, fixed wavefront reached 65.0% but response-decay wavefront reached 55.0%; both were below fixed 32-token blocks at 72.5%. Response trends are too noisy for per-step range control, so any adaptive rule should act only at completed region boundaries.
