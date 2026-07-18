import copy
import json
import tempfile
from pathlib import Path

from compare_exact_output_runs import compare_records, load_records


def record(task_id="HumanEval/0", nfe=100):
    return {
        "task_id": task_id,
        "absolute_index": int(task_id.rsplit("/", 1)[1]),
        "prompt_hash": "prompt-hash",
        "raw_gold": "gold",
        "normalized_gold": "normalized-gold",
        "decoded_generation": "generation",
        "correct": True,
        "nfe": nfe,
    }


def test_exact_faster_candidate_passes():
    parent = {"HumanEval/0": record(nfe=100)}
    candidate = {"HumanEval/0": record(nfe=70)}
    summary = compare_records(parent, candidate)
    assert summary["all_invariants_pass"]
    assert summary["nfe_reduction"] == 30


def test_output_or_prompt_change_fails():
    parent = {"HumanEval/0": record()}
    changed = copy.deepcopy(parent["HumanEval/0"])
    changed["prompt_hash"] = "different"
    changed["decoded_generation"] = "different"
    summary = compare_records(parent, {"HumanEval/0": changed})
    assert not summary["all_invariants_pass"]
    assert not summary["invariants"]["identity_alignment"]
    assert not summary["invariants"]["decoded_generation_exact"]


def test_missing_id_or_nfe_regression_fails():
    parent = {
        "HumanEval/0": record("HumanEval/0", 100),
        "HumanEval/1": record("HumanEval/1", 100),
    }
    candidate = {"HumanEval/0": record("HumanEval/0", 101)}
    summary = compare_records(parent, candidate)
    assert not summary["all_invariants_pass"]
    assert summary["missing_ids"] == ["HumanEval/1"]
    assert not summary["invariants"]["per_record_nfe_nonincrease"]


def test_loader_rejects_duplicate_ids_and_missing_fields():
    with tempfile.TemporaryDirectory() as directory:
        duplicate_path = Path(directory) / "duplicate.jsonl"
        payload = json.dumps(record())
        duplicate_path.write_text(payload + "\n" + payload + "\n", encoding="utf-8")
        try:
            load_records(duplicate_path)
            raise AssertionError("duplicate task id was accepted")
        except ValueError as error:
            assert "duplicate task_id" in str(error)

        missing_path = Path(directory) / "missing.jsonl"
        missing_path.write_text('{"task_id":"HumanEval/0"}\n', encoding="utf-8")
        try:
            load_records(missing_path)
            raise AssertionError("missing audit fields were accepted")
        except ValueError as error:
            assert "missing required fields" in str(error)


if __name__ == "__main__":
    test_exact_faster_candidate_passes()
    test_output_or_prompt_change_fails()
    test_missing_id_or_nfe_regression_fails()
    test_loader_rejects_duplicate_ids_and_missing_fields()
    print("4 exact-output evaluator tests passed")
