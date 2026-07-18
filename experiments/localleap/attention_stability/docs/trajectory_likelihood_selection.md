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

### Block-evidence descendant

The initial mean selector improved HumanEval-32 from 13/32 to 14/32 versus the
fast parent, but its two losses had only 0.007--0.011 nats/token advantage for
the accuracy path, while all three recoveries had 0.057--0.085.  The descendant
therefore keeps `fast` unless the accuracy path accumulates at least one extra
nat per existing generation block:

`(mean_logp_accuracy - mean_logp_fast) * block_length > 1`.

This is an evidence requirement derived from the existing block structure, not
a fitted continuous margin.  A cross-domain counterfactual using completed
MATH-50 trajectories improved 15/50 to an exploratory 16/50.  Formal promotion
uses the unseen HumanEval/32..63 suffix, not the discovery prefix.

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
