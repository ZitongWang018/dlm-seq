# Training-free dLLM baseline alignment (2026-07-16)

This document separates locally audited LocalLeap results from values reported by
recent papers. Paper values are references only; they are not reproductions and
must not be merged into local paired statistics.

## Local comparison contract

- Backbone: `/root/autodl-tmp/model/LLaDA/instruct` (LLaDA-8B-Instruct).
- Baseline: original LLaDA low-confidence decoding, with no attention selector.
- Best accuracy parent: symmetric attention dependency, `tau=0.004`, without
  forced budget filling.
- Fixed-budget speed child: the same symmetric selector with stable-conflict
  pruning and budget filling; its NFE is matched to the original baseline.
- Block length: 32. Seeds: `0,1234,1234,1234`.
- Every local pair must match stable IDs, prompt hashes, target hashes, generation
  length, diffusion steps, few-shot count, and seed.
- Report accuracy, NFE, wall time, token throughput, residual masks, extraction
  failures, and paired gains/losses. Variable-NFE parent results are explicitly
  labeled accuracy-first rather than compute-matched.

## Recent paper-reported reference rows

### Order-Token Search (OTS, arXiv:2601.20339v2)

OTS is the closest paper-level comparison because it uses LLaDA-8B-Instruct,
low-confidence remasking, block size 32, and `generation_length = 2 * steps`.
At sequence length 256 (128 diffusion steps), its paper reports:

| Method | GSM8K | MATH-500 | Countdown | HumanEval |
| --- | ---: | ---: | ---: | ---: |
| LLaDA low-confidence | 76.7 | 32.4 | 19.5 | 26.2 |
| Order-Token Search | 79.8 | 36.0 | 26.2 | 34.2 |

The current local HumanEval-128 row is configuration-near: original LLaDA is
42/164 (25.61%) and symmetric-fast is 52/164 (31.71%). It is legitimate to show
these rows in the same *configuration-alignment* table, but not to claim an OTS
reproduction because repository revision, decoding temperature, prompt template,
and execution stack were not jointly pinned.

### Prism (ICML 2026, arXiv:2602.01842v3)

Prism uses LLaDA-8B-Instruct, zero-shot official test sets, low-confidence
remasking, block length 32, temperature 0.7, and 32 steps per block. It uses
generation length/NFE 256 for GSM8K and MATH-500, and generation length/NFE 512
for HumanEval and MBPP. The paper reports:

| Method | GSM8K (NFE) | MATH-500 (NFE) | HumanEval (NFE) | MBPP (NFE) |
| --- | ---: | ---: | ---: | ---: |
| LLaDA-8B-Instruct N=1 | 67.58 (256) | 26.40 (256) | 54.88 (512) | 21.80 (512) |
| Prism K=2 | 74.24 (283) | 30.16 (334) | 71.34 (549) | 29.40 (561) |
| Prism K=4 | 75.30 (509) | 37.70 (622) | 76.19 (1133) | 32.40 (1196) |
| Prism K=8 | 85.30 (1048) | 42.80 (1304) | 79.27 (2480) | 38.20 (2576) |

Prism is a test-time scaling method with multiple trajectories and self-verifier
calls, so its K>1 rows are not compute-matched to a single-trajectory 128-step
selector. Compare the accuracy-NFE Pareto curve, not accuracy alone.

### SOAR (arXiv:2602.10953v2)

SOAR uses LLaDA-8B-Base rather than Instruct, confidence threshold 0.95, maximum
beam size 2, and reports 0-shot HumanEval, 3-shot MBPP, and 4-shot GSM8K on an
A100-80GB. Its LLaDA-Base rows are:

| Method | HumanEval 256/512 | MBPP 256/512 | GSM8K 256/512 | Mean speedup |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 32.3 / 32.9 | 40.8 / 39.2 | 70.4 / 70.9 | 1.00x reference |
| Adaptive parallel | 32.3 / 32.9 | 40.8 / 39.2 | 70.4 / 71.0 | 2.19x |
| SOAR | 32.9 / 39.0 | 40.8 / 39.4 | 71.3 / 71.5 | 1.62x |

These rows establish the quality-speed comparison protocol, but their absolute
accuracies are not directly comparable to the local Instruct checkpoint.

## New queue matrix

The queue `best_symmetric_long_20260716_v1` prioritizes results that fill the
remaining comparison gaps:

1. Full MATH-500 zero-shot at length 256/128 steps: original LLaDA,
   symmetric-fast, and accuracy-first symmetric.
2. Full MBPP 3-shot symmetric-fast at length 256/128 steps, paired against the
   existing formally audited original-LLaDA baseline.
3. Full GSM8K zero-shot at length 256/128 steps: original LLaDA,
   symmetric-fast, and accuracy-first symmetric.
4. A paired 500-example GSM8K four-shot study for prompt-level SOAR alignment.
5. A 250-example MATH-500 length-512 robustness pair.

The queue stores source hashes, run configs, sample-level audit records, trace
statistics, paired summaries, wall time, throughput, and NFE. Full step-level
diagnostics are restricted to smoke samples to keep disk usage bounded.

## Primary sources

- Prism: https://arxiv.org/abs/2602.01842 and https://github.com/viiika/Prism
- SOAR: https://arxiv.org/abs/2602.10953 and https://github.com/duterscmy/SOAR
- Order-Token Search: https://arxiv.org/abs/2601.20339

