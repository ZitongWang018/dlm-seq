from pathlib import Path


def script_text():
    return (
        Path(__file__).parents[1]
        / "scripts"
        / "run_disagreement_evidence_queue.sh"
    ).read_text()


def test_holdout_is_stable_id_suffix_not_training_prefix():
    script = script_text()
    assert "--start 64 --end 96" in script
    assert "hm > hfast && hm > hbase" in script
    assert "DONE_NO_MATH" in script


def test_two_gpus_focus_humaneval_before_math():
    script = script_text()
    method = script.index("run_gpu he_disagreement_n96 0")
    parents = script.index("run_gpu he_fast_n96 1")
    math = script.index("run_gpu math_disagreement_n50 0")
    assert method < math and parents < math


def test_comparators_are_original_and_fixed_budget_parents():
    script = script_text()
    assert "formal_baseline=original_llada_low_confidence" in script
    assert "he_base_n96 1" in script
    assert "he_fast_n96 1" in script
    assert "he_accuracy_n96" not in script


if __name__ == "__main__":
    test_holdout_is_stable_id_suffix_not_training_prefix()
    test_two_gpus_focus_humaneval_before_math()
    test_comparators_are_original_and_fixed_budget_parents()
    print("3 disagreement-evidence queue tests passed")
