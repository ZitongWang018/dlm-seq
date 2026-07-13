# Memory

## 2026-07-13

- Cleaned misleading eval artifacts (failed traj/response/agreement/ceiling) from local + AutoDL `dlm-seq-flow/results`.
- **Did not touch** `/root/autodl-tmp/LocalLeap` (code, configs, baseline/LocalLeap HE results).
- Mirrored LocalLeap scripts into `experiments/localleap/` for future tests.
- HumanEval (postprocess): baseline 40.85%, LocalLeap 40.24%; TPS ~9.7 → ~41 on one GPU.
- Official HE score requires `postprocess_code.py`; raw lm_eval underestimates.

## Earlier (summary)

- Old “recover gold from past logits / direct response timing” directions failed on math.
- MATH/HumanEval ceiling scripts had metric bugs (see `cursorfeedback/0712grokproblem.md`).
- top-k distance support-index bug was fixed in `src/distribution.py` (tests cover it).
