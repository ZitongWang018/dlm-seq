from audit_partial_mbpp_trace import audit_records


PROMPT = '''Write a function add_one(x). Your code should pass:

assert add_one(1) == 2
assert add_one(-1) == 0
[BEGIN]
'''


def row(index, task_id, generation, *, nfe=128, prompt_hash="p"):
    return {
        "absolute_index": index,
        "task_id": task_id,
        "prompt_hash": prompt_hash,
        "prompt_text": PROMPT,
        "entry_point": None,
        "decoded_generation": generation,
        "nfe": nfe,
    }


def test_good_bad_and_compile_invalid_candidates():
    records = [
        row(0, 11, "```python\ndef add_one(x):\n    return x + 1\n```", nfe=129),
        row(1, 12, "```python\ndef add_one(x):\n    return x - 1\n```", nfe=300),
        row(2, 13, "```python\ndef add_one(:\n    pass\n```", nfe=140),
    ]
    summary = audit_records(records, expected_total=4)
    assert summary["pass"], summary
    assert summary["correct"] == 1
    assert summary["compile_valid"] == 2
    assert summary["crosscheck_mismatch_ids"] == []
    assert summary["nfe_total"] == 569
    assert summary["missing_prefix_ids"] == []
    assert not summary["uses_hidden_tests"]
    assert not summary["uses_reference_solution"]


def test_current_prompt_segment_excludes_fewshot_assertions():
    prompt = '''assert solved_demo(1) == 99
[DONE]
Write add_one.
assert add_one(1) == 2
[BEGIN]
'''
    record = row(0, 11, "def add_one(x):\n    return x + 1")
    record["prompt_text"] = prompt
    summary = audit_records([record], expected_total=1)
    assert summary["pass"], summary
    assert summary["correct"] == 1


def test_identity_nfe_mask_and_hash_anomalies():
    records = [
        row(0, 11, "def add_one(x):\n return x + 1", nfe=128),
        row(2, 11, "def add_one(x):\n return x + 1\n# [MASK]", nfe=float("nan"), prompt_hash=""),
    ]
    summary = audit_records(records, expected_total=3)
    assert not summary["pass"]
    assert set(summary["anomalies"]) == {
        "duplicate_task_ids",
        "missing_prefix_ids",
        "invalid_or_missing_nfe",
        "residual_masks",
        "missing_prompt_hash",
    }


def test_missing_visible_assertions_is_anomaly():
    record = row(0, 1, "def add_one(x):\n return x + 1")
    record["prompt_text"] = "Write add_one. [BEGIN]"
    summary = audit_records([record], expected_total=1)
    assert not summary["pass"]
    assert "missing_prompt_visible_checks" in summary["anomalies"]


if __name__ == "__main__":
    test_good_bad_and_compile_invalid_candidates()
    test_current_prompt_segment_excludes_fewshot_assertions()
    test_identity_nfe_mask_and_hash_anomalies()
    test_missing_visible_assertions_is_anomaly()
    print("4 partial MBPP trace audit tests passed")
