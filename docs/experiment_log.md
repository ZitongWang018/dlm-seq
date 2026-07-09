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

## Report

Chinese stage report: [`docs/阶段性报告.md`](阶段性报告.md)  
Report figures: `results/report_assets/`
