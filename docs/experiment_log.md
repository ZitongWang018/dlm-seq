# Experiment Log

## Round 0 — Baseline (LCR)

| Dataset | Accuracy | Config |
|---------|----------|--------|
| GSM8K (50) | **62%** | steps=64, gen_length=128 |
| MBPP (50) | **0%** | invalid: steps < gen_length |

Finding: MBPP needs `steps >= gen_length` for full unmask. Fixed to `mbpp_steps=256`.

## Round 1 — Observation (20 GSM8K samples)

| Statistic | Value |
|-----------|-------|
| Coupling sparsity | **99.98%** |
| Mean coupling strength | ~1.6e-5 |
| Mean path/net ratio | ~410 |
| Trajectory class counts | all labeled "frozen" (threshold failure) |

Figures: `results/round1_observation/path_scatter_*.png`, `coupling_*.png`

## Round 2 — Methods on GSM8K (50)

| Method | Accuracy |
|--------|----------|
| LCR | **62%** |
| RCR | 58% |
| Traj | 58% |
| Traj+Lateral | 58% |

Disagreement vs LCR (Traj): only-LCR=6, only-Traj=4.

## Round 3 — Ceiling + Flip (Arbor `dlm-traj-v2`)

### 3a. H4 ceiling — GSM8K (Node 2)

| Metric | Value |
|--------|-------|
| Accuracy | 62% (31/50) |
| Incorrect | 19 |
| Ceiling (gold ever argmax in answer region) | **0%** |

Artifact: `results/round3_ceiling/gsm8k_ceiling.json`

### 3b. MATH LCR + ceiling (Node 3.1)

| Metric | Value |
|--------|-------|
| Dataset | MATH-dev L1–3, 50 |
| LCR accuracy | **48%** (24/50) |
| Ceiling | **0%** (0/26 wrong) |
| Config | steps=256, gen_length=256 |

Artifact: `results/math_baseline/math_ceiling.json`  
Data: `dataset/math/{dev,test}.json`, HumanEval `dataset/humaneval/test.json`

### 3c. Flip diagnostic (Node 6)

| Region | Correct (n=31) | Incorrect (n=19) |
|--------|----------------|------------------|
| Answer flip mean | 2.51 ± 2.06 | **4.27 ± 1.68** |
| Reasoning flip mean | 6.52 ± 2.37 | 5.83 ± 1.93 |
| Pearson r(reasoning flip, correct) | 0.149 | |

Artifact: `results/round3_flip/gsm8k_flip.json`

### 3d. Pruned (premise invalidated by ceiling=0)

- Node 1: argmax identity stability sampler (`src/samplers_v2.py` kept, not fully eval'd)
- Node 4: Jaccard top-K stability sampler

## Round 4 — Flip-gated Best-of-2 (Node 6.1)

| Metric | Value |
|--------|-------|
| Threshold T | 3.5 |
| LCR | 62% |
| BO2 | **62%** (gain **0**) |
| Triggered | 26/50 |
| pred1 ≠ pred2 | **0** (temperature=0, deterministic) |
| Detection | TP=14, FP=12, FN=5, TN=19 |

Artifact: `results/round4_bo2/gsm8k_bo2_t3.5.json`

## Pending

- Node 5: HumanEval ceiling / pass@1 (`scripts/run_humaneval_ceiling.py`)
- BO2 with temperature > 0, or flip-based abstention curves

## Report

Chinese stage report (full narrative): [`docs/阶段性报告.md`](阶段性报告.md)  
Arbor session: `.arbor/sessions/dlm-traj-v2/`  
Report figures: `results/report_assets/`
