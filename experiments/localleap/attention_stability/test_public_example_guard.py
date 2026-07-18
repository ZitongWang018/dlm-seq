from differential_selector import select_public_example_guard


PROMPT = '''def square(x: int) -> int:
    """
    >>> square(2)
    4
    >>> square(-3)
    9
    """
'''


def test_strictly_better_baseline_replaces_parent():
    selected, diagnostics = select_public_example_guard(
        "def square(x):\n    return x * x",
        "def square(x):\n    return x + x",
        PROMPT,
        "square",
    )
    assert selected == "baseline"
    assert diagnostics["visible_examples_passed"] == {
        "baseline": 2,
        "parent": 1,
    }


def test_tie_preserves_parent():
    selected, diagnostics = select_public_example_guard(
        "def square(x):\n    return x * x",
        "def square(x):\n    return x ** 2",
        PROMPT,
        "square",
    )
    assert selected == "parent"
    assert diagnostics["visible_examples_passed"] == {
        "baseline": 2,
        "parent": 2,
    }


def test_missing_examples_preserve_parent_without_generated_probes():
    selected, diagnostics = select_public_example_guard(
        "def square(x):\n    return x * x",
        "def square(x):\n    return 0",
        "Return the square of x.",
        "square",
    )
    assert selected == "parent"
    assert diagnostics["visible_example_count"] == 0
    assert diagnostics["uses_generated_probes"] is False
    assert diagnostics["uses_hidden_tests"] is False
    assert diagnostics["uses_reference_solution"] is False


def test_current_task_assertions_are_used_but_fewshot_assertions_are_ignored():
    prompt = '''Earlier task:
assert helper(2) == 100
[DONE]
Current task tests:
assert square(2) == 4
assert square(-3) == 9
[BEGIN]
'''
    selected, diagnostics = select_public_example_guard(
        "def square(x):\n    return x * x",
        "def square(x):\n    return x + 1",
        prompt,
        None,
    )
    assert selected == "baseline"
    assert diagnostics["visible_example_count"] == 0
    assert diagnostics["visible_assertion_count"] == 2
    assert diagnostics["visible_check_count"] == 2
    assert diagnostics["visible_examples_passed"] == {
        "baseline": 2,
        "parent": 0,
    }


if __name__ == "__main__":
    test_strictly_better_baseline_replaces_parent()
    test_tie_preserves_parent()
    test_missing_examples_preserve_parent_without_generated_probes()
    test_current_task_assertions_are_used_but_fewshot_assertions_are_ignored()
    print("4 public-example guard tests passed")
