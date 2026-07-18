from verify_selected_execution import verify


def test_matching_execution_passes():
    summary = verify(
        [{"task_id": "HumanEval/0", "correct": True}],
        [{"task_id": "HumanEval/0", "pass_at_1": 1.0}],
    )
    assert summary["all_correctness_matches"]
    assert summary["execution_correct"] == 1


def test_mismatch_and_identity_errors_fail():
    summary = verify(
        [{"task_id": "HumanEval/0", "correct": True}],
        [{"task_id": "HumanEval/0", "pass_at_1": 0.0}],
    )
    assert not summary["all_correctness_matches"]
    assert summary["correctness_mismatch_ids"] == ["HumanEval/0"]
    try:
        verify(
            [{"task_id": "HumanEval/0", "correct": True}],
            [{"task_id": "HumanEval/1", "pass_at_1": 1.0}],
        )
        raise AssertionError("identity mismatch accepted")
    except ValueError as error:
        assert "do not align" in str(error)


if __name__ == "__main__":
    test_matching_execution_passes()
    test_mismatch_and_identity_errors_fail()
    print("2 selected-execution verification tests passed")
