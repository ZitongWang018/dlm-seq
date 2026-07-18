# Confidence-Switched Stability Decoding

## Retrospective insight

The verified tau-0.004 decoder and the failed descendants separate two kinds of
dependency events.  A dense attention edge is not automatically useful: when
both candidates keep the same top-1 after the preceding commit, enforcing that
edge only removes parallelism.  When a candidate changes after explicitly
seeing the newly committed token, the same edge represents unfinished
conditioning and should not be force-filled.

This distinction explains the earlier extremes.  Accuracy-first symmetric
decoding waits on every strong edge and can spend too much compute.  The
`symmetric_fast` child prunes stable conflicts but force-fills every remaining
gap, including the informative rewritten cases.  Global Top-K ordering,
response refinement, draft exchange, and differential execution selection all
changed a much larger part of the parent path and either tied it at extra cost
or damaged correct samples.

## Method

The method keeps the original LLaDA confidence order, the symmetric dependency
threshold tau=0.004, and the parent's adjacent-step top-1 test.  At each step:

1. candidates are ordered by parent maturity and confidence;
2. strong same-batch edges are ignored only when both endpoints are unchanged
   under the most recent explicit condition;
3. an edge involving a conditioned top-1 rewrite remains active, and the step
   may underfill rather than commit both endpoints;
4. the next ordinary forward supplies the missing explicit conditioning.

There is no new score weight, candidate count, threshold, verifier, draft, or
historical distribution.  The horizontal signal decides whether two positions
can be committed together; the longitudinal signal decides whether the edge is
still informative.  Easy steps retain fixed-budget parallel decoding, while
risky steps receive extra denoising compute.

## Evaluation gate

Original LLaDA remains the formal baseline.  `symmetric_fast` at tau=0.004 is
the speed parent and accuracy-first symmetric tau=0.004 is an upper reference.
HumanEval and MATH-500 are regenerated from one frozen source snapshot on two
GPUs.  The method expands from 32/50 to 64/100 and then full evaluation only if
its combined paired correctness strictly improves over the speed parent, does
not lose more than one item on either task, and stays within 1.35x NFE.  MBPP
and GSM8K sampled generalization run only after the HE/MATH promotion.
