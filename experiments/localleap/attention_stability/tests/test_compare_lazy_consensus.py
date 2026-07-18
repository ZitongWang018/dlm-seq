import json
import tempfile
from pathlib import Path

from compare_lazy_consensus import compare_runs


def write_jsonl(path, records):
    Path(path).write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_equivalent_outputs_and_saved_nfe_are_counted():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        full_audit = [{
            "task_id": "HumanEval/0",
            "prompt_hash": "same",
            "decoded_generation": "answer",
            "correct": True,
        }]
        lazy_audit = [dict(full_audit[0])]
        full_trace = [{
            "task_id": "HumanEval/0",
            "nfe": 400,
            "decode_diagnostics": {
                "selected_name": "fast",
                "candidate_nfe": {"fast": 128, "accuracy": 144, "baseline": 128},
            },
        }]
        lazy_trace = [{
            "task_id": "HumanEval/0",
            "nfe": 272,
            "decode_diagnostics": {
                "selected_name": "fast",
                "candidate_nfe": {"fast": 128, "accuracy": 144},
            },
        }]
        paths = [root / name for name in ("fa.jsonl", "la.jsonl", "ft.jsonl", "lt.jsonl")]
        for path, records in zip(paths, (full_audit, lazy_audit, full_trace, lazy_trace)):
            write_jsonl(path, records)
        summary = compare_runs(*paths)
        assert summary["mismatch_count"] == 0
        assert summary["nfe_saved"] == 128
        assert summary["baseline_skips"] == 1


def test_generation_and_selection_mismatches_are_rejected():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        full_audit = [{
            "task_id": "HumanEval/0",
            "prompt_hash": "same",
            "decoded_generation": "answer-a",
            "correct": True,
        }]
        lazy_audit = [{
            "task_id": "HumanEval/0",
            "prompt_hash": "same",
            "decoded_generation": "answer-b",
            "correct": False,
        }]
        full_trace = [{
            "task_id": "HumanEval/0",
            "nfe": 400,
            "decode_diagnostics": {"selected_name": "accuracy", "candidate_nfe": {}},
        }]
        lazy_trace = [{
            "task_id": "HumanEval/0",
            "nfe": 272,
            "decode_diagnostics": {"selected_name": "fast", "candidate_nfe": {}},
        }]
        paths = [root / name for name in ("fa.jsonl", "la.jsonl", "ft.jsonl", "lt.jsonl")]
        for path, records in zip(paths, (full_audit, lazy_audit, full_trace, lazy_trace)):
            write_jsonl(path, records)
        summary = compare_runs(*paths)
        assert summary["mismatch_count"] == 1
        assert set(summary["mismatches"][0]["fields"]) == {
            "decoded_generation",
            "correct",
            "selected_name",
        }


if __name__ == "__main__":
    test_equivalent_outputs_and_saved_nfe_are_counted()
    test_generation_and_selection_mismatches_are_rejected()
    print("2 lazy consensus comparator tests passed")
