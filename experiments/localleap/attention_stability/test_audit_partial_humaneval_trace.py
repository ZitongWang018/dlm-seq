from audit_partial_humaneval_trace import build_check_program


def test_program_matches_lm_eval_humaneval_composition():
    doc = {
        "prompt": "def add(a, b):\n",
        "test": "def check(candidate):\n    assert candidate(2, 3) == 5",
        "entry_point": "add",
    }
    program = build_check_program("    return a + b", doc)
    scope = {}
    exec(program, scope)
    assert scope["add"](5, 7) == 12


if __name__ == "__main__":
    test_program_matches_lm_eval_humaneval_composition()
    print("1 partial HumanEval audit test passed")
