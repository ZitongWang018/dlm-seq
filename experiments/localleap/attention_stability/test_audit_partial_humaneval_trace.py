from audit_partial_humaneval_trace import build_check_program


def test_program_matches_lm_eval_humaneval_composition():
    sample = {
        "doc": {
            "prompt": "def add(a, b):\n",
            "entry_point": "add",
        },
        "target": "def check(candidate):\n    assert candidate(2, 3) == 5\ncheck(add)",
    }
    def fake_sanitize(text, entry_point):
        assert entry_point == "add"
        assert "explanation" not in text
        return text.split("\n\n", 1)[1]

    program = build_check_program(
        "```python\ndef add(a, b):\n    return a + b\n```\nexplanation",
        sample,
        fake_sanitize,
    )
    scope = {}
    exec(program, scope)
    assert scope["add"](5, 7) == 12


if __name__ == "__main__":
    test_program_matches_lm_eval_humaneval_composition()
    print("1 partial HumanEval audit test passed")
