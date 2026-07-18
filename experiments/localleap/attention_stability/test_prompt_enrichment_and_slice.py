import copy

from enrich_audit_prompts import enrich
from slice_audit_by_index import slice_records


def test_enrichment_checks_hashes_and_preserves_order():
    records = [
        {"task_id": "2", "absolute_index": 1, "prompt_hash": "p2", "target_hash": "t2"},
        {"task_id": "1", "absolute_index": 0, "prompt_hash": "p1", "target_hash": "t1"},
    ]
    samples = [
        {"doc": {"task_id": 1}, "prompt_hash": "p1", "target_hash": "t1", "arguments": {"gen_args_0": {"arg_0": "prompt-1"}}},
        {"doc": {"task_id": 2}, "prompt_hash": "p2", "target_hash": "t2", "arguments": {"gen_args_0": {"arg_0": "prompt-2"}}},
    ]
    output = enrich(records, samples)
    assert [row["task_id"] for row in output] == ["1", "2"]
    assert [row["prompt_text"] for row in output] == ["prompt-1", "prompt-2"]
    broken = copy.deepcopy(samples)
    broken[0]["prompt_hash"] = "wrong"
    try:
        enrich(records, broken)
        raise AssertionError("hash mismatch accepted")
    except ValueError as error:
        assert "prompt_hash mismatch" in str(error)


def test_contiguous_slice_rejects_missing_indices():
    records = [{"absolute_index": index} for index in range(4)]
    assert [row["absolute_index"] for row in slice_records(records, 1, 3)] == [1, 2]
    try:
        slice_records([records[1]], 1, 3)
        raise AssertionError("missing slice index accepted")
    except ValueError as error:
        assert "slice indices mismatch" in str(error)


if __name__ == "__main__":
    test_enrichment_checks_hashes_and_preserves_order()
    test_contiguous_slice_rejects_missing_indices()
    print("2 prompt-enrichment/slice tests passed")
