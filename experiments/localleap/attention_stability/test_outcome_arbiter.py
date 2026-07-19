from outcome_arbiter import (
    build_outcome_arbiter_prompt,
    infer_outcome_family,
    normalize_outcome,
    select_arbitrated_candidate,
)


def test_family_and_normalization():
    assert infer_outcome_family({"unique_id": "x", "problem": "p"}) == "math"
    assert infer_outcome_family({"question": "q", "answer": "work #### 1,250"}) == "gsm"
    assert normalize_outcome("final \\boxed{\\frac{3}{2}}", "math") == "3/2"
    assert normalize_outcome("work #### 1,250", "gsm") == "1250"


def test_prompt_deduplicates_outcomes_and_forbids_frequency_voting():
    prompt = build_outcome_arbiter_prompt(
        "What is 1+1?", {"fast": "2", "accuracy": "3", "baseline": "3"}
    )
    assert prompt.count("Candidate answer") == 2
    assert "rather than voting by frequency" in prompt


def test_parent_wins_an_arbiter_tie():
    selected, diagnostics = select_arbitrated_candidate(
        "fast", {"fast": "2", "accuracy": "2", "baseline": "3"}, "2"
    )
    assert selected == "fast"
    assert diagnostics["status"] == "arbiter_confirms_parent"


def test_arbiter_may_select_only_an_existing_complete_trajectory():
    selected, diagnostics = select_arbitrated_candidate(
        "fast", {"fast": "2", "accuracy": "3", "baseline": "3"}, "3"
    )
    assert selected == "accuracy"
    assert diagnostics["creates_novel_selected_answer"] is False
    selected, diagnostics = select_arbitrated_candidate(
        "fast", {"fast": "2", "accuracy": "3", "baseline": "4"}, "5"
    )
    assert selected == "fast"
    assert diagnostics["status"] == "unmatched_or_empty_arbiter"


if __name__ == "__main__":
    test_family_and_normalization()
    test_prompt_deduplicates_outcomes_and_forbids_frequency_voting()
    test_parent_wins_an_arbiter_tie()
    test_arbiter_may_select_only_an_existing_complete_trajectory()
    print("4 outcome-arbiter tests passed")
