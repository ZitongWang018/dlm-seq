import json
from pathlib import Path


ROOT = Path(__file__).parent


def main():
    prereg = json.loads(
        (ROOT / "original_anchor_pareto_preregistration_20260720_v1.json").read_text()
    )
    controller = (
        ROOT / "scripts" / "run_original_anchor_pareto_after_repairs.sh"
    ).read_text()
    runner = (ROOT / "scripts" / "run_best_symmetric_benchmark.sh").read_text()
    generate = (ROOT / "generate.py").read_text()
    evaluator = (ROOT / "eval_llada.py").read_text()

    assert prereg["method_profile"] == "trajectory_original_anchor_pareto"
    assert prereg["single_algorithm"] is True
    assert prereg["task_specific_routing"] is False
    assert prereg["one_rule"]["new_tuned_thresholds"] == "none"
    assert prereg["one_rule"]["token_splicing"] is False
    assert "trajectory_original_anchor_pareto)" in runner
    assert "dependency_likelihood_selection_mode=original_anchor_pareto" in runner
    assert '"original_anchor_pareto"' in evaluator
    assert "def generate_original_anchor_pareto(" in generate
    assert "def select_original_anchor_pareto(" in generate

    for required in (
        "TRANSFORMERS_OFFLINE=1",
        "waiting_for_direct_v19_terminal",
        "he_anchor_baseline_dev32",
        "he_anchor_pareto_dev32",
        "gsm_anchor_pareto_dev64",
        "math_anchor_pareto_dev50",
        "mbpp_anchor_pareto_dev100",
        "audit_generation_leakage_v2.py",
        "prompt_hash_mismatches",
        "target_hash_mismatches",
        "duplicate_or_missing_ids",
        "method_total_nfe",
        "ACCEPTED",
        "REJECTED",
    ):
        assert required in controller, required
    assert "raw_gold" not in generate
    assert "normalized_gold" not in generate
    print("original anchor Pareto queue contract passed")


if __name__ == "__main__":
    main()
