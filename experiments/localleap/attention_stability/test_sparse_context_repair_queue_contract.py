#!/usr/bin/env python3
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    prereg = json.loads(
        (ROOT / "sparse_context_repair_preregistration_20260720_v1.json")
        .read_text(encoding="utf-8")
    )
    assert prereg["single_algorithm"] is True
    assert prereg["method_profile"] == "trajectory_early_sparse_context_repair"
    forbidden = " ".join(prereg["generation_inputs_forbidden"])
    for term in ("raw_gold", "hidden tests", "reference solution", "correctness"):
        assert term in forbidden

    generate = (ROOT / "generate.py").read_text(encoding="utf-8")
    evaluator = (ROOT / "eval_llada.py").read_text(encoding="utf-8")
    runner = (
        ROOT / "scripts" / "run_best_symmetric_benchmark.sh"
    ).read_text(encoding="utf-8")
    generic = (
        ROOT / "scripts" / "run_localized_evidence_conflict_repair_queue.sh"
    ).read_text(encoding="utf-8")
    wrapper = (
        ROOT / "scripts" / "run_sparse_context_repair_after_fair.sh"
    ).read_text(encoding="utf-8")

    assert "select_context_ambiguous_positions" in generate
    assert "early_sparse_context_repair" in generate
    assert "early_sparse_context_repair_lazy_public_guard" in evaluator
    assert "trajectory_early_sparse_context_repair)" in runner
    for variable in ("${profile}", "${run_prefix}", "${preregistration}"):
        assert variable in generic
    assert "audit_generation_leakage_v2.py" in generic
    assert wrapper.index("waiting_for_fair_queue") < wrapper.index(
        "launching_sparse_context_repair_after_fair"
    )
    assert "SKIPPED_V18_ACCEPTED" in wrapper
    assert "fair_three_arm_leakage_recovery_20260720_v2" in wrapper
    assert "HF_DATASETS_OFFLINE=1" in wrapper
    assert "HF_HUB_OFFLINE=1" in wrapper
    assert "TRANSFORMERS_OFFLINE=1" in wrapper
    assert "PROFILE=trajectory_early_sparse_context_repair" in wrapper
    assert "RUN_PREFIX=v19" in wrapper
    print("sparse context repair queue contract passed")


if __name__ == "__main__":
    main()
