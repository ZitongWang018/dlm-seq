import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "slice_audit_records.py"
SPEC = importlib.util.spec_from_file_location("slice_audit_records", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_stable_id_slice_is_complete_and_ordered():
    records = [
        {"stable_task_id": "HumanEval/3"},
        {"stable_task_id": "HumanEval/1"},
        {"stable_task_id": "HumanEval/2"},
        {"stable_task_id": "HumanEval/0"},
    ]
    selected = MODULE.slice_records(records, 1, 4)
    assert [MODULE.humaneval_index(record) for record in selected] == [1, 2, 3]


def test_stable_id_slice_rejects_missing_ids():
    try:
        MODULE.slice_records([{"task_id": "HumanEval/1"}], 1, 3)
    except ValueError as error:
        assert "slice ids mismatch" in str(error)
    else:
        raise AssertionError("missing id was accepted")


if __name__ == "__main__":
    test_stable_id_slice_is_complete_and_ordered()
    test_stable_id_slice_rejects_missing_ids()
    print("2 audit-slice tests passed")
