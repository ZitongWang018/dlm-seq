# Response-Credit Draft Exchange

## Scope

This is a training-free descendant of the verified LocalLeap/LLaDA symmetric
attention decoder at `tau=0.004`. Original LLaDA low-confidence decoding remains
the formal baseline. The method does not alter model parameters, inject KV state,
use a learned verifier, or inspect benchmark reference solutions or hidden tests.

## Motivation and relation to prior work

Prism shows that diffusion inference can benefit from trajectory diversity,
partial remasking, and compute reallocation rather than treating one denoising
path as final. Diffusion in Diffusion motivates refining an already completed
draft under wider bidirectional context. Differential code selection motivates
using agreement in observable program behavior rather than confidence alone.

The implementation here deliberately does not reproduce those algorithms. It
uses exactly two deterministic decoding policies, no learned/self verifier, no
entropy pruning window, no confidence-remask fraction, no mix-scale training,
and no coverage-guided fuzzer. The new object is the interaction between the
existing cross-step conditioning signal and cross-policy full-draft disagreement.

Primary references:

- Prism: <https://arxiv.org/abs/2602.01842>
- Diffusion in Diffusion: <https://arxiv.org/abs/2601.13599>
- DiffCodeGen: <https://arxiv.org/abs/2605.20473>

## Vertical response credit

Let `S_(t-1)` be the positions committed on the preceding step and let
`r_j^t=max_(i in S_(t-1)) d_ji^t`. For an undecoded position, response credit is

```
u_j^t = u_j^(t-1) + 1  if r_j^t > tau and top1_j^t = top1_j^(t-1)
        0               if r_j^t > tau and top1_j^t != top1_j^(t-1)
        u_j^(t-1)       if r_j^t <= tau.
```

Thus a token earns evidence only after surviving a real strong-dependency
conditioning event. Repeated weak steps do not manufacture stability, and a
conditioned top-1 change clears stale evidence. Maturity remains the parent's
binary rule; within the mature tier the order is `(u_j^t, confidence)`.

## Horizontal control and draft exchange

Both trajectories retain the same tau-0.004 symmetric dependency exclusion
within every parallel commit batch:

- anchor: the verified adjacent-top-1 parent order;
- explorer: response-credit order.

After both complete, positions on which they agree form a full-draft skeleton.
Every disagreement is returned to `[MASK]` at once and re-denoised under that
shared skeleton. Repair again uses symmetric exclusion and response credit. It
does not copy the anchor or explorer token at a disputed position. Consequently,
the horizontal component acts both within each batch and across the two complete
policy trajectories, while the repair trajectory supplies new vertical
conditioning events.

The fixed-budget `response_credit_fast` variant uses one trajectory and always
fills the original LLaDA commit budget after the selector runs. It is the direct
parallel-speed comparison; draft exchange is an accuracy-scaling arm and reports
its full NFE and wall-clock cost.

## Code-only differential selection

HumanEval and MBPP retain three candidates: anchor, explorer, and repaired. The
selector receives only the user-visible prompt and generated programs. It:

1. extracts and executes prompt-visible doctest examples when available;
2. compiles/runs each program in an isolated subprocess with resource limits;
3. generates a small deterministic probe set from the public function signature;
4. clusters candidates by their behavior on the common probes;
5. orders candidates lexicographically by visible-example passes, validity,
   behavior-cluster size, successful probes, then repaired-draft tie preference.

`canonical_solution`, benchmark tests, gold answers, and evaluator correctness
are never passed to this selector. MATH-500 and GSM8K disable it entirely.

## Promotion protocol

The new queue starts only after the active best-symmetric queue exits. It runs
2-example real-model/evaluator preflights, paired 32-example HumanEval gates, a
second 64-example gate for the expensive draft branch, and 100-example MBPP or
MATH-500 / 128-example GSM8K checks. Full runs are promoted only when the paired
gate is non-negative within the declared small-sample tolerance. All comparisons
join by stable task id and require identical prompt and target hashes. Results
record candidate texts, selected candidate, response validations/invalidations,
draft disagreement count, per-component NFE, total NFE, residual masks, wall
time, and source hashes.

## Minimal vertical descendant: conditioned revision margin

Binary top-1 change is evidence that a new condition had an effect, but it is
not itself evidence that the new answer is bad. A smaller single-trajectory
descendant therefore preserves the parent's mature/confidence ordering and
changes only the unstable tail. For a strong-dependency position whose top-1
changed after the preceding commit, it records

```
g_j^t = log q_j^t(top1_j^t) - log q_j^t(top1_j^(t-1)).
```

The unstable tail is ordered by `(g_j^t, confidence)`. A large value means the
newly available condition decisively displaced the previous candidate; a small
value means the revision remains ambiguous. This uses the previous top-1 as a
response probe, not as a presumed correct answer. It adds no threshold and no
historical distribution storage. `revision_margin_fast` keeps the original
per-step commit budget and the same tau-0.004 symmetric horizontal exclusion.

Queue promotion now compares `response_credit_fast` and
`revision_margin_fast` directly with the established `symmetric_fast` parent on
the same 32/64 HumanEval samples. Correct-count ties stay with the parent. Only
a new rule with strictly more correct answers is promoted to a fresh full
baseline/parent comparison. Larger MATH-500, GSM8K and MBPP exploration is
sample-gated; full MBPP requires at least +3/100 over the parent.
