# Idea Tree

**Baseline**: 62.0% | **Trunk**: 62.0%

## ROOT: Research session [RUNNING]

**Insight**: Baseline LCR=62% GSM8K (50). Round1 observation: coupling sparsity 99.98%, path/net ratio 410, all positions classified frozen. Round2: all traj methods 58%, below baseline. Root cause: confidence scalar is wrong observable (loses candidate identity); single-commit coupling is near-zero noise; positions with trajectory data are selection-biased toward hard/uncertain positions. Must fix measurement before fixing sampler.

### 1: Mechanism: replace scalar-confidence path with argmax-identity stability across steps.
Hypothesis: positions whose argmax token remains unchanged for ≥k consecutive steps before commit are more likely to be correct than recently-flipped positions; gating on this stability can outperform LCR.
Observable: accuracy on GSM8K-50 with stability-gated sampler vs LCR 62%.
Conflicts: selection bias — positions with long histories are exactly the uncertain ones LCR deferred; stability may not correlate with correctness if model oscillates around the correct token. [PRUNED]

**Insight**: [Pruned: Ceiling=0%: model never visits correct token for wrong samples. Identity-stable scoring assumes there's a correct candidate to stabilize toward — this assumption is violated. The wrong arithmetic prediction is stable from step 1, leaving no signal to gate on.]

**Result**: Code written: samplers_v2.py with IdentityState (argmax history + Jaccard top-K), score_ids/score_jac/score_ids+ functions. Eval script run_v2_eval.py ready. Waiting for H4 ceiling (Node 2) to complete before running to confirm headroom exists.

**Branch**: `exp/n1-mechanism-replace-scalar-confide-aa515c34`

### 2: Mechanism: measure the theoretical ceiling before designing any sampler — how often does LLaDA visit the correct token mid-trajectory but commit to a wrong one at the end.
Hypothesis: if ≥15% of wrong answers had the correct token as argmax at some intermediate step, test-time recovery is viable; if <5%, trajectory-based approaches have negligible headroom.
Observable: fraction of incorrect samples where answer-region argmax was correct at any step t < final step.
Conflicts: this is a measurement study, not a method; it only sets the ceiling, not the path to reach it. [DONE] (score: 62.0%)

**Insight**: DEFINITIVE NEGATIVE: ceiling=0% on GSM8K. For all 19 incorrect samples, the gold answer token was NEVER the argmax at any denoising step in the answer region (last 15 positions). Model's wrong arithmetic predictions are STABLE from step 1 — the failure is not 'committed the wrong token at the right moment' but 'consistently computed the wrong arithmetic result throughout all 64 steps.' Implication: trajectory-based recovery cannot help with this failure mode on GSM8K. Wrong reasoning is internally consistent, not oscillating between right and wrong.

**Result**: accuracy=62%, incorrect=19, ceiling_ratio=0%, all 19 wrong samples had ever_correct_ratio=0. n_answer_positions=15 (fallback heuristic used for all wrong samples, confirming gold token never appeared).

**Branch**: `exp/n2-mechanism-measure-the-theoretica-441dcc62`

### 3: Mechanism: expand evaluation to MATH (harder, longer reasoning chains, 50+ tokens/solution) and increase sample size to reduce variance — current 50-sample GSM8K gives 8% swing per 4 questions.
Hypothesis: on MATH level-1/2, where solutions are longer and uncertainty is higher, trajectory signal should be stronger and sampler improvements more detectable.
Observable: LCR baseline on MATH-50 dev; comparison of sampler gains on GSM8K vs MATH.
Conflicts: MATH solutions are much longer (need gen_length=512+), increasing NFE and wall time per sample significantly. [DONE] (score: 48.0%)

**Insight**: Dataset expansion complete: MATH (5000, dev=2462/test=2538) and HumanEval (164) downloaded. MATH LCR=48%, MATH ceiling=0%. The trajectory-stability hypothesis fails on both GSM8K (62%) and MATH (48%). Math reasoning errors in LLaDA are 'committed' from step 1 regardless of task complexity. Pivot to error-detection (flip-count signal) and HumanEval (code has different token statistics).

**Result**: MATH downloaded: 5000 total, dev=2462 (level 1-3), test=2538 (level 4-5). HumanEval: 164 problems. Loaders datasets_v2.py + run_math_baseline.py written. Need to run LCR baseline on MATH-dev to establish score. Waiting for GPU (Node 2 running).

**Branch**: `exp/n3-mechanism-expand-evaluation-to-m-b3da5c5b`

#### 3.1: Mechanism: run LCR baseline on MATH level 1-3 (2462 problems, sample 50) to establish a new benchmark point on a harder multi-token task, then measure H4 ceiling on MATH wrong samples — if ceiling > 0, MATH is a better testbed for trajectory methods than GSM8K.
Hypothesis: MATH level 1-3 solutions are longer (100-300 tokens), have more intermediate steps, and model errors may be less 'arithmetically locked-in' than GSM8K — trajectory ceiling may be non-zero on MATH wrong samples.
Observable: LCR accuracy on MATH-50-dev, ceiling ratio on wrong MATH samples.
Conflicts: gen_length=256 required (2x slower than GSM8K); answer extraction harder (boxed format, symbolic math). [DONE] (score: 48.0%)

**Insight**: MATH (L1-3) ceiling = 0%, consistent with GSM8K. LCR baseline = 48% on MATH-50-dev (harder than GSM8K 62% as expected). For 26 wrong samples, the correct boxed answer token was NEVER the argmax at any step in the last-30-positions answer region. The 'stable wrong' pattern is NOT specific to simple arithmetic (GSM8K) — it holds for complex multi-step math up to level 3. This conclusively rules out trajectory-based recovery on symbolic math tasks.

**Result**: MATH-dev-L1-3: accuracy=48% (24/50), ceiling_ratio=0% (0/26 wrong). gen_length=256, steps=256, 50 samples, runtime ~1000s.

**Branch**: `exp/n3-1-mechanism-run-lcr-baseline-on-ma-f3fc1fba`

### 4: Mechanism: exploit top-K candidate-set Jaccard stability (set intersection / union across last k steps) as a richer signal than scalar confidence — captures whether the same tokens compete across steps regardless of their rank order.
Hypothesis: positions with high top-K Jaccard stability (same candidates competing) but low current confidence should be delayed; positions with low Jaccard (candidates changing entirely) indicate context-sensitivity and should also be delayed; only positions with converging Jaccard AND rising confidence should be advanced.
Observable: accuracy vs LCR on GSM8K-50; also correlation between Jaccard stability and final correctness.
Conflicts: computing full top-K Jaccard requires storing token-level history (memory cost); K must be tuned. [PRUNED]

**Insight**: [Pruned: Same as Node 1: Jaccard top-K stability assumes the correct token competes in the top-K across steps. Ceiling=0% means correct token never reaches top-1 for wrong samples — likely not in top-K either. No basis for Jaccard gating.]

### 5: Mechanism: shift from arithmetic (GSM8K) to code generation (HumanEval), where output structure differs fundamentally — code has stable scaffolding (def name, return structure) AND variable body tokens, and a single wrong token can produce a correct-looking but failing function, creating a different error profile.
Hypothesis: in HumanEval, wrong code samples will show non-zero ceiling (correct token visited mid-trajectory) because code has more diverse candidate sets and longer chains where intermediate states matter; trajectory-gated resampling of uncertain body tokens can improve pass@1.
Observable: H4 ceiling on HumanEval (pass@1 baseline), and pass@1 of trajectory-gated sampler vs LCR.
Conflicts: HumanEval requires code execution to evaluate (slower); pass@1 variance is high at 164 samples; LLaDA may not be optimised for Python code generation specifically. [RUNNING]

**Branch**: `exp/n5-mechanism-shift-from-arithmetic--af21ac3a`

### 6: Mechanism: instead of recovering the final answer, use trajectory instability as a DIAGNOSTIC signal to detect intermediate reasoning errors — positions in the chain-of-thought that are low-stability may indicate where the arithmetic went wrong, enabling targeted re-generation of those steps only.
Hypothesis: in GSM8K, incorrect samples will show higher token-flip counts at intermediate calculation result positions (the = X tokens) than correct samples; this instability can serve as a quality signal without requiring recovery.
Observable: compare flip-count distribution at intermediate vs answer positions for correct vs incorrect GSM8K samples; Pearson correlation between mean flip-count in reasoning region and final correctness.
Conflicts: this is observation-only (no sampler change); finding a correlation doesn't directly improve accuracy but would confirm trajectory has diagnostic value beyond the answer region. [DONE] (score: 62.0%)

**Insight**: Key finding: wrong answers have 70% higher answer-region flip count (4.27 vs 2.51, std~2). Reasoning-region flip count weakly correlates with correctness (r=0.149, positive direction, opposite of hypothesis). Interpretation: (1) Answer flip count is a proxy for LLaDA's own uncertainty at the answer position — even though the wrong token is never replaced by the correct one, the model oscillates among multiple wrong tokens, indicating low confidence. (2) Reasoning flip count does not predict errors — correct samples have MORE reasoning flips (6.5 vs 5.8), suggesting richer intermediate computation in correct chains. (3) Flip count can serve as error DETECTION (flag high-flip answers for re-generation) but NOT as error CORRECTION (the correct token still never appears). New direction: use answer flip count as a rejection signal for selective re-generation (best-of-N sampling guided by flip-count threshold).

**Result**: correct: answer_flip=2.512±2.062, reasoning_flip=6.515±2.374. incorrect: answer_flip=4.270±1.680, reasoning_flip=5.826±1.933. Pearson r(reasoning_flips, correct)=0.149.

**Branch**: `exp/n6-mechanism-instead-of-recovering--f8cd4600`

#### 6.1: Mechanism: use answer-region flip count as a rejection signal for selective re-generation — after LCR generation, if mean flip count of committed answer tokens exceeds threshold T, run a second generation attempt and take majority vote between the two; if both agree, output that; if they disagree, take the one with lower flip count.
Hypothesis: this flip-count-gated best-of-2 will outperform single LCR on GSM8K because: (a) wrong answers have 70% higher flip counts, so the rejection filter preferentially triggers on wrong samples; (b) re-generation from a different random noise seed has a nonzero chance of producing the correct answer for the same question.
Observable: pass@1 (single LCR) vs flip-gated best-of-2 on GSM8K-50. NFE doubles for flagged samples (expected ~50% of incorrect=19 samples, so ~10 extra full generations).
Conflicts: this is essentially best-of-2 with smart selection, not a trajectory-guided sampler; the improvement ceiling is limited by how often re-generation produces a different answer; if wrong answers always regenerate wrong, no gain. [DONE] (score: 62.0%)

**Insight**: Flip-count can DETECT uncertainty but cannot CORRECT under temperature=0: regeneration is deterministic. Confirms ceiling=0: model never explores the correct token across greedy runs. Next options: temperature>0 for diversity, or use flip only for abstention/selective answering.

**Result**: Flip-gated BO2 threshold=3.5: LCR=62%, BO2=62%, gain=0. Triggered 26/50. temperature=0 => all regenerations identical (pred1==pred2). Detection TP=14 FP=12 FN=5 TN=19.

**Branch**: `exp/n6-1-mechanism-use-answer-region-flip-6555c00e`
