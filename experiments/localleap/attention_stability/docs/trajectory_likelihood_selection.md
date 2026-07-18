# Two-parent trajectory-likelihood selection

## Motivation

The confidence-filled symmetric parent and the conservative symmetric parent
make different horizontal decisions about strongly dependent positions.  On
the fixed discovery slices their oracle union exceeds either parent by three
examples on both HumanEval (16/32 versus 13/32) and MATH-500 (18/50 versus
15/50), while single-path waiting rules produced no recoveries.  The remaining
problem is therefore path selection, not another commit delay.

## Method

Generate exactly two deterministic candidates with the same model, prompt,
token budget, tau=0.004, and decoding configuration:

- `fast`: prune stable conflicts and fill the native step budget;
- `accuracy`: retain all strong conflicts and allow under-filled steps.

For every token, record its top-1 confidence at the exact denoising step where
it is committed.  The vertical trajectory score is

`mean_j log c_j(commit_step(j))`.

Select the candidate with the larger mean.  A score tie deterministically
selects `fast`.  There is no score weight, margin, task-specific execution,
reference access, candidate splice, or new threshold.

Horizontal thought is represented by the two distinct within-step dependency
schedules.  Vertical thought is represented by evidence accumulated along the
actual sequence of commits, after new token conditions have entered subsequent
forwards.

## Evaluation gate

GPU 0 runs HumanEval and GPU 1 runs MATH-500.  A two-example real-model smoke
precedes paired 32/50 discovery.  Expansion requires at least one method-only
recovery, no more than one loss on either task, positive combined gain over the
fast parent, and at most 2.5x total NFE.  Original LLaDA remains the formal
baseline in every stage.
