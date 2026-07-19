from audit_partial_humaneval_trace import audit, build_check_program


def fake_sanitize(text, entry_point):
    assert entry_point == "add"
    start = text.rfind("def add")
    return text[start:]


def sample(task_id, prompt_hash="p"):
    return {
        "doc": {
            "task_id": task_id,
            "prompt": "def add(a, b):\n",
            "entry_point": "add",
        },
        "target": "def check(candidate):\n    assert candidate(2, 3) == 5\ncheck(add)",
        "prompt_hash": prompt_hash,
        "target_hash": "t-" + task_id,
    }


def trace(index, task_id, generation, *, nfe=128, prompt_hash="p"):
    return {
        "absolute_index": index,
        "task_id": task_id,
        "prompt_hash": prompt_hash,
        "prompt_text": "def add(a, b):\n",
        "decoded_generation": generation,
        "correct": None,
        "nfe": nfe,
    }


def test_program_matches_lm_eval_humaneval_composition():
    row = sample("HumanEval/0")
    program = build_check_program(
        "```python\ndef add(a, b):\n    return a + b\n```\nexplanation",
        row,
        fake_sanitize,
    )
    scope = {}
    exec(program, scope)
    assert scope["add"](5, 7) == 12


def test_prefix_execution_health_and_lineage():
    samples = [sample("HumanEval/0"), sample("HumanEval/1")]
    traces = [
        trace(0, "HumanEval/0", "```python\ndef add(a, b):\n return a + b\n```", nfe=129),
        trace(1, "HumanEval/1", "```python\ndef add(a, b):\n return a - b\n```", nfe=300),
    ]
    records, summary = audit(
        traces,
        samples,
        2.0,
        fake_sanitize,
        expected_total=3,
        trailing_partial=1,
    )
    assert summary["pass"], summary
    assert summary["correct"] == 1
    assert summary["nfe_total"] == 429
    assert summary["missing_prefix_ids"] == []
    assert summary["trailing_partial_lines"] == 1
    assert summary["uses_hidden_tests"]
    assert summary["health_only_dev_prefix"]
    assert summary["post_generation_only"]
    assert not summary["for_candidate_selection"]
    assert [row["target_hash"] for row in records] == ["t-HumanEval/0", "t-HumanEval/1"]


def test_nfe_mask_gap_and_non_null_correct_anomalies():
    samples = [sample("HumanEval/0"), sample("HumanEval/2")]
    traces = [
        trace(0, "HumanEval/0", "def add(a,b):\n return a+b", nfe=128),
        trace(2, "HumanEval/2", "def add(a,b):\n return a+b\n# [MASK]", nfe=float("nan")),
    ]
    traces[-1]["correct"] = True
    _, summary = audit(traces, samples, 2.0, fake_sanitize, expected_total=3)
    assert not summary["pass"]
    assert set(summary["anomalies"]) == {
        "missing_prefix_ids",
        "invalid_or_missing_nfe",
        "residual_masks",
        "generation_trace_contains_non_null_correct",
    }


def test_duplicate_trace_identity_is_rejected():
    samples = [sample("HumanEval/0")]
    traces = [
        trace(0, "HumanEval/0", "def add(a,b): return a+b"),
        trace(1, "HumanEval/0", "def add(a,b): return a+b"),
    ]
    try:
        audit(traces, samples, 2.0, fake_sanitize)
    except ValueError as error:
        assert "duplicate trace id" in str(error)
    else:
        raise AssertionError("duplicate identity was accepted")


if __name__ == "__main__":
    test_program_matches_lm_eval_humaneval_composition()
    test_prefix_execution_health_and_lineage()
    test_nfe_mask_gap_and_non_null_correct_anomalies()
    test_duplicate_trace_identity_is_rejected()
    print("4 partial HumanEval audit tests passed")
