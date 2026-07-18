# Directed Cross-Conditioned Refinement

## Motivation

The established parent is `symmetric_fast` with `tau=0.004`; original LLaDA
low-confidence decoding remains the formal baseline.  Causal-Pareto v3 found
longitudinally invalidated positions and repaired them in directed-attention
order, but its selector masked every changed token at once.  That selector
compared token values under a common context while still withholding the other
changed token values, so it did not validate their joint explicit condition.

This limitation matches the refinement tension reported by Speculative
Refinement: a second stage can damage already-correct structure.  Prism's local
partial remasking motivates keeping the complete high-confidence draft, while
SOAR motivates spending search compute only in uncertain regions.  The method
below borrows those general principles but does not reproduce their search or
verification algorithms.

Primary references:

- Prism: <https://arxiv.org/abs/2602.01842>
- SOAR: <https://arxiv.org/abs/2602.10953>
- Speculative Refinement: <https://arxiv.org/abs/2606.27474>

## Method

1. Generate the full parent draft with the unchanged `symmetric_fast` decoder.
2. Remask only positions that were both longitudinally invalidated after a new
   condition arrived and committed while forced or immature.
3. Repair those positions in source-first order using the non-symmetric
   attention matrix `A[target, source]`.
4. Within every changed block, order changed positions by outgoing minus
   incoming directed attention and alternate them into source and dependent
   views.
5. Mask the source view while keeping the dependent view as explicit draft or
   repaired tokens; score both full-draft candidates in one batch.  Exchange
   the two views and score again.
6. Retain the repaired block only if every changed token has strictly positive
   repaired-minus-original log-probability in its explicit cross-conditioned
   view.  There is no weighted score or additional acceptance threshold.

The vertical signal therefore decides *where* a completed draft needs another
denoising pass.  Directed horizontal attention decides *which explicit token
conditions* must be visible when the repair is validated.  Batched candidate
validation preserves trajectory-level parallelism; traces report both forward
calls and candidate-equivalent evaluations.

For code benchmarks, an optional safety child selects between the retained
parent and repaired candidate using only examples and type information present
in the prompt.  It never uses hidden tests or the reference solution.  This
component is evaluated separately from the generic selector.

## Evaluation policy

Every gate regenerates original LLaDA, `symmetric_fast`, and the new method from
the same frozen source snapshot.  Historical parent outputs are not reused.
HumanEval and MATH-500 start with small paired gates, expand to 64/100 only when
the combined result is non-regressive, and run full evaluations only when both
datasets are individually non-regressive and the combined result strictly
improves.  Pairing uses stable task identity plus prompt and target hashes.
