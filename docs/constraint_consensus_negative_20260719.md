# Constraint-consensus branch: pre-GPU rejection

Date: 2026-07-19 (registered and audited while the v15 full4 queue was still running)

## Hypothesis

Extend the unified v15 decoder with one fallback: when prompt-visible checks are absent and the fast and accuracy-first drafts expose different explicit final claims, generate original LLaDA as a third draft and let an exact two-of-three claim vote override the confirmed parent.

The intended horizontal evidence was trajectory diversity; the existing confirmed bidirectional selector supplied vertical evidence. The implementation never accepted gold answers, reference solutions, hidden tests, or correctness as generation inputs, and the static leakage audit reported zero violations.

## Synthetic and contract checks

- Nested `\\boxed{}` extraction, negative decimals, terminal punctuation, missing claims, majority, tie, and runner-contract cases passed.
- Existing public-example guard tests passed.
- Static generation-leakage audit passed.

## Label-free replay

Replaying only generated drafts (without reading gold or correctness) showed that the fallback would generate a third draft on 31/50 historical MATH records and change six parent decisions. It would not change any of the 64 historical GSM records.

## Historical exploratory outcome

After the method and rule were frozen, the six proposed changes were scored on the already-consumed historical MATH-50 development records:

| Scope | Parent | Candidate | Paired gain | Paired loss |
|---|---:|---:|---:|---:|
| MATH-50 historical exploratory | 16/50 | 14/50 | 0 | 2 |
| GSM-64 historical exploratory | unchanged | unchanged | 0 | 0 |

This is not a formal benchmark result and must not be mixed with the current full4 audit.

## Insight and decision

The majority assumption was invalid: original LLaDA and the fast symmetric path share enough decoding structure that their agreement is correlated, while the v9/v15 parent already requires both longitudinal commit evidence and bidirectional full-draft confirmation. Counting the correlated paths as two independent votes can undo a stronger decision.

The branch is rejected before GPU evaluation. Its waiting controller was not launched, no holdout was opened, and no production/frozen source was changed. Future accuracy work must introduce genuinely orthogonal evidence or a prompt-derived executable constraint; it must not add another unweighted vote from a correlated decoder.
