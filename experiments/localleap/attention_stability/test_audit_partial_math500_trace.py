import importlib.util
from pathlib import Path

from audit_partial_math500_trace import audit_records


UTILS = Path(__file__).parent / "tasks" / "localleap_math500" / "utils.py"
SPEC = importlib.util.spec_from_file_location("test_partial_math_utils", UTILS)
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def row(index, task_id, generation, gold, *, nfe=128, prompt_hash="p"):
    return {
        "absolute_index": index,
        "task_id": task_id,
        "prompt_hash": prompt_hash,
        "decoded_generation": generation,
        "raw_gold": gold,
        "correct": None,
        "nfe": nfe,
    }


def test_nested_fraction_tuple_negative_decimal_and_missing_answer():
    records = [
        row(0, "a", r"Thus $\boxed{\frac{1}{2}}$.", r"\frac{1}{2}", nfe=129),
        row(
            1,
            "b",
            r"<answer>\boxed{(r,\theta)=(3,\frac{\pi}{2})}</answer>",
            r"\left(3,\frac{\pi}{2}\right)",
            nfe=300,
        ),
        row(2, "c", "The final answer is -2.5.", "-2.5", nfe=140),
        row(3, "d", "I cannot determine this.", "4", nfe=150),
    ]
    summary = audit_records(records, EVALUATOR, expected_total=5, evaluator_path=UTILS)
    assert summary["pass"], summary
    assert summary["correct"] == 3
    assert summary["extraction_failures"] == 1
    assert summary["nfe_total"] == 719
    assert summary["missing_prefix_ids"] == []
    assert summary["post_generation_only"]
    assert summary["generation_inputs_used"] == ["decoded_generation"]
    assert summary["evaluation_only_fields_used"] == ["raw_gold"]
    assert not summary["uses_hidden_tests"]
    assert len(summary["task_evaluator_sha256"]) == 64


def test_identity_nfe_mask_hash_gold_and_correct_anomalies():
    records = [
        row(0, "a", r"\boxed{1}", "1", nfe=128),
        row(2, "a", r"\boxed{2} [MASK]", "", nfe=float("inf"), prompt_hash=""),
    ]
    records[-1]["correct"] = True
    summary = audit_records(records, EVALUATOR, expected_total=3)
    assert not summary["pass"]
    assert set(summary["anomalies"]) == {
        "duplicate_task_ids",
        "missing_prefix_ids",
        "invalid_or_missing_nfe",
        "residual_masks",
        "missing_prompt_hash",
        "missing_post_generation_gold",
        "generation_trace_contains_non_null_correct",
    }


def test_out_of_range_and_duplicate_absolute_ids():
    records = [
        row(0, "a", r"\boxed{1}", "1"),
        row(0, "b", r"\boxed{1}", "1"),
        row(4, "c", r"\boxed{1}", "1"),
    ]
    summary = audit_records(records, EVALUATOR, expected_total=3)
    assert not summary["pass"]
    assert "duplicate_absolute_ids" in summary["anomalies"]
    assert "out_of_range_ids" in summary["anomalies"]


if __name__ == "__main__":
    test_nested_fraction_tuple_negative_decimal_and_missing_answer()
    test_identity_nfe_mask_hash_gold_and_correct_anomalies()
    test_out_of_range_and_duplicate_absolute_ids()
    print("3 partial MATH-500 trace audit tests passed")
