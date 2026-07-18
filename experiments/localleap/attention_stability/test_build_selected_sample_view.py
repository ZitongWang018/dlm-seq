from build_selected_sample_view import build_view


def test_only_response_is_replaced_and_order_is_stable():
    samples = [
        {"doc": {"task_id": "HumanEval/1"}, "resps": [["old-1"]], "target": "a"},
        {"doc": {"task_id": "HumanEval/0"}, "resps": [["old-0"]], "target": "b"},
    ]
    selected = [
        {"task_id": "HumanEval/0", "decoded_generation": "new-0"},
        {"task_id": "HumanEval/1", "decoded_generation": "new-1"},
    ]
    output = build_view(samples, selected)
    assert [row["doc"]["task_id"] for row in output] == ["HumanEval/0", "HumanEval/1"]
    assert [row["resps"][0][0] for row in output] == ["new-0", "new-1"]
    assert [row["target"] for row in output] == ["b", "a"]


def test_missing_or_duplicate_ids_fail():
    sample = {"doc": {"task_id": "HumanEval/0"}, "resps": [["old"]]}
    selected = {"task_id": "HumanEval/0", "decoded_generation": "new"}
    try:
        build_view([sample, sample], [selected])
        raise AssertionError("duplicate sample id accepted")
    except ValueError as error:
        assert "duplicate" in str(error)
    try:
        build_view([sample], [])
        raise AssertionError("missing selected id accepted")
    except ValueError as error:
        assert "do not align" in str(error)


if __name__ == "__main__":
    test_only_response_is_replaced_and_order_is_stable()
    test_missing_or_duplicate_ids_fail()
    print("2 selected-sample view tests passed")
