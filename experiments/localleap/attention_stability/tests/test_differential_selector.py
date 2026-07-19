from differential_selector import (
    build_type_probes,
    decide_prompt_visible_repair_retention,
    prompt_examples,
    select_differential_candidate,
)


def test_build_type_probes_uses_public_signature_only():
    probes = build_type_probes("def f(x: int, y: str):\n    return y * x\n", "f")
    assert probes[0] == "f(0, '')"
    assert len(probes) == 6


def test_prompt_examples_extracts_only_matching_entry_point():
    prompt = """Examples:\n>>> f(2)\n4\n>>> other(2)\n8\n"""
    assert prompt_examples(prompt, "f") == [("f(2)", "4")]


def test_visible_example_selects_correct_candidate_without_hidden_tests():
    candidates = [
        "def f(x: int):\n    return x + 1\n",
        "def f(x: int):\n    return x * 2\n",
        "def f(x: int):\n    return x - 1\n",
    ]
    selected, diagnostics = select_differential_candidate(
        candidates, ">>> f(3)\n6\n", "f"
    )
    assert selected == 1
    assert diagnostics["uses_hidden_tests"] is False
    assert diagnostics["uses_reference_solution"] is False


def test_behavior_consensus_selects_largest_cluster():
    candidates = [
        "def f(x: int):\n    return x + 1\n",
        "def f(x: int):\n    return 1 + x\n",
        "def f(x: int):\n    return x - 1\n",
    ]
    selected, diagnostics = select_differential_candidate(candidates, "", "f")
    assert selected in {0, 1}
    assert diagnostics["behavior_cluster_sizes"] == [2, 2, 1]


def test_timeout_or_invalid_candidate_loses_to_valid_candidate():
    candidates = [
        "def f(x: int):\n    while True:\n        pass\n",
        "def f(x: int):\n    return x\n",
        "this is not python",
    ]
    selected, diagnostics = select_differential_candidate(candidates, "", "f")
    assert selected == 1
    assert diagnostics["compile_valid"][2] is False


def test_prompt_visible_repair_retention_rejects_fewer_passes():
    repair = {
        "visible_check_count": 2,
        "visible_checks_passed": 1,
        "compile_valid": True,
    }
    parent = {
        "visible_check_count": 2,
        "visible_checks_passed": 2,
        "compile_valid": True,
    }
    selected, diagnostics = decide_prompt_visible_repair_retention(
        repair, parent
    )
    assert selected == "parent"
    assert diagnostics["retained_repair"] is False
    assert diagnostics["uses_hidden_tests"] is False


def test_prompt_visible_repair_retention_uses_compile_on_a_pass_tie():
    repair = {
        "visible_check_count": 2,
        "visible_checks_passed": 0,
        "compile_valid": False,
    }
    parent = {
        "visible_check_count": 2,
        "visible_checks_passed": 0,
        "compile_valid": True,
    }
    selected, _ = decide_prompt_visible_repair_retention(repair, parent)
    assert selected == "parent"


def test_prompt_visible_repair_retention_keeps_an_exact_tie():
    repair = {
        "visible_check_count": 1,
        "visible_checks_passed": 1,
        "compile_valid": True,
    }
    parent = dict(repair)
    selected, diagnostics = decide_prompt_visible_repair_retention(
        repair, parent
    )
    assert selected == "repair"
    assert diagnostics["retained_repair"] is True


if __name__ == "__main__":
    test_build_type_probes_uses_public_signature_only()
    test_prompt_examples_extracts_only_matching_entry_point()
    test_visible_example_selects_correct_candidate_without_hidden_tests()
    test_behavior_consensus_selects_largest_cluster()
    test_timeout_or_invalid_candidate_loses_to_valid_candidate()
    print("5 differential-selector tests passed")
