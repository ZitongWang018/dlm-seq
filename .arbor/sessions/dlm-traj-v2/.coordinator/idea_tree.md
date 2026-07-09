# Idea Tree

**Baseline**: 62.0% | **Trunk**: 62.0%

## ROOT: Research session [RUNNING]

**Insight**: Baseline LCR=62% GSM8K (50). Round1 observation: coupling sparsity 99.98%, path/net ratio 410, all positions classified frozen. Round2: all traj methods 58%, below baseline. Root cause: confidence scalar is wrong observable (loses candidate identity); single-commit coupling is near-zero noise; positions with trajectory data are selection-biased toward hard/uncertain positions. Must fix measurement before fixing sampler.

### 1: Mechanism: replace scalar-confidence path with argmax-identity stability across steps.
Hypothesis: positions whose argmax token remains unchanged for ≥k consecutive steps before commit are more likely to be correct than recently-flipped positions; gating on this stability can outperform LCR.
Observable: accuracy on GSM8K-50 with stability-gated sampler vs LCR 62%.
Conflicts: selection bias — positions with long histories are exactly the uncertain ones LCR deferred; stability may not correlate with correctness if model oscillates around the correct token. [RUNNING]

**Result**: Code written: samplers_v2.py with IdentityState (argmax history + Jaccard top-K), score_ids/score_jac/score_ids+ functions. Eval script run_v2_eval.py ready. Waiting for H4 ceiling (Node 2) to complete before running to confirm headroom exists.

**Branch**: `exp/n1-mechanism-replace-scalar-confide-aa515c34`

### 2: Mechanism: measure the theoretical ceiling before designing any sampler — how often does LLaDA visit the correct token mid-trajectory but commit to a wrong one at the end.
Hypothesis: if ≥15% of wrong answers had the correct token as argmax at some intermediate step, test-time recovery is viable; if <5%, trajectory-based approaches have negligible headroom.
Observable: fraction of incorrect samples where answer-region argmax was correct at any step t < final step.
Conflicts: this is a measurement study, not a method; it only sets the ceiling, not the path to reach it. [RUNNING]

**Branch**: `exp/n2-mechanism-measure-the-theoretica-441dcc62`

### 3: Mechanism: expand evaluation to MATH (harder, longer reasoning chains, 50+ tokens/solution) and increase sample size to reduce variance — current 50-sample GSM8K gives 8% swing per 4 questions.
Hypothesis: on MATH level-1/2, where solutions are longer and uncertainty is higher, trajectory signal should be stronger and sampler improvements more detectable.
Observable: LCR baseline on MATH-50 dev; comparison of sampler gains on GSM8K vs MATH.
Conflicts: MATH solutions are much longer (need gen_length=512+), increasing NFE and wall time per sample significantly. [RUNNING]

**Result**: MATH downloaded: 5000 total, dev=2462 (level 1-3), test=2538 (level 4-5). HumanEval: 164 problems. Loaders datasets_v2.py + run_math_baseline.py written. Need to run LCR baseline on MATH-dev to establish score. Waiting for GPU (Node 2 running).

**Branch**: `exp/n3-mechanism-expand-evaluation-to-m-b3da5c5b`

### 4: Mechanism: exploit top-K candidate-set Jaccard stability (set intersection / union across last k steps) as a richer signal than scalar confidence — captures whether the same tokens compete across steps regardless of their rank order.
Hypothesis: positions with high top-K Jaccard stability (same candidates competing) but low current confidence should be delayed; positions with low Jaccard (candidates changing entirely) indicate context-sensitivity and should also be delayed; only positions with converging Jaccard AND rising confidence should be advanced.
Observable: accuracy vs LCR on GSM8K-50; also correlation between Jaccard stability and final correctness.
Conflicts: computing full top-K Jaccard requires storing token-level history (memory cost); K must be tuned. [PENDING]
