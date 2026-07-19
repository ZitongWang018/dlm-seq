from audit_mbpp_assertions import audit_record, trace_from_sample


def make_trace(generation, prompt):
    return {
        "absolute_index": 0,
        "task_id": "mbpp_0",
        "prompt_hash": "p",
        "prompt_text": prompt,
        "entry_point": "square",
        "decoded_generation": generation,
        "raw_gold": "def square(x): return x*x",
        "nfe": 128,
        "seed": {"torch": 1234},
        "generation_settings": {"steps": 128},
    }


TASK = {"stable_task_id": "mbpp_0", "prompt_hash": "p", "target_hash": "t"}
PROMPT = """Old example:\nassert square(2) == 5\n[DONE]\nCurrent:\nassert square(2) == 4\nassert square(-3) == 9\n"""


def test_correct_program_passes_both_execution_paths():
    row = audit_record(make_trace("def square(x):\n    return x*x", PROMPT), TASK)
    assert row["correct"] is True
    assert row["assertion_diagnostics"]["visible_assertion_count"] == 2
    assert row["assertion_diagnostics"]["crosscheck_match"] is True


def test_wrong_program_is_not_repaired_by_the_evaluator():
    row = audit_record(make_trace("def square(x):\n    return x+x", PROMPT), TASK)
    assert row["correct"] is False
    assert row["assertion_diagnostics"]["crosscheck_match"] is True


def test_syntax_error_is_separate_from_a_correctness_pass():
    row = audit_record(make_trace("def square(:\n    pass", PROMPT), TASK)
    assert row["correct"] is False
    assert row["assertion_diagnostics"]["compile_valid"] is False
    assert row["assertion_diagnostics"]["crosscheck_match"] is True


def test_prompt_hash_mismatch_is_rejected():
    bad = dict(TASK, prompt_hash="other")
    try:
        audit_record(make_trace("def square(x): return x*x", PROMPT), bad)
    except ValueError as error:
        assert "prompt hash mismatch" in str(error)
    else:
        raise AssertionError("prompt hash mismatch was accepted")


def test_lm_eval_sample_adapter_preserves_generation_and_identity():
    task = dict(
        TASK,
        absolute_index=7,
        decoded_generation="def square(x): return x*x",
        raw_gold="def square(x): return x*x",
        nfe=128,
    )
    sample = {
        "doc": {"task_id": "mbpp_0"},
        "arguments": {"gen_args_0": {"arg_0": PROMPT}},
        "resps": [[task["decoded_generation"]]],
        "prompt_hash": "p",
        "target_hash": "t",
        "filter": "none",
    }
    trace = trace_from_sample(sample, task)
    assert trace["task_id"] == "mbpp_0"
    assert trace["absolute_index"] == 7
    assert trace["decoded_generation"] == task["decoded_generation"]
    assert audit_record(trace, task)["correct"] is True


def test_lm_eval_sample_adapter_rejects_generation_drift():
    task = dict(
        TASK,
        absolute_index=0,
        decoded_generation="def square(x): return x*x",
        raw_gold="def square(x): return x*x",
        nfe=128,
    )
    sample = {
        "doc": {"task_id": "mbpp_0"},
        "arguments": {"gen_args_0": {"arg_0": PROMPT}},
        "resps": [["def square(x): return 0"]],
        "prompt_hash": "p",
        "target_hash": "t",
        "filter": "none",
    }
    try:
        trace_from_sample(sample, task)
    except ValueError as error:
        assert "generation mismatch" in str(error)
    else:
        raise AssertionError("generation drift was accepted")


if __name__ == "__main__":
    test_correct_program_passes_both_execution_paths()
    test_wrong_program_is_not_repaired_by_the_evaluator()
    test_syntax_error_is_separate_from_a_correctness_pass()
    test_prompt_hash_mismatch_is_rejected()
    test_lm_eval_sample_adapter_preserves_generation_and_identity()
    test_lm_eval_sample_adapter_rejects_generation_drift()
    print("6 MBPP assertion-audit tests passed")
