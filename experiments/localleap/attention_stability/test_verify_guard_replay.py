from verify_guard_replay import verify


def record(name, correct, generation, baseline_passes=0, parent_passes=0):
    return {
        "task_id": "1",
        "prompt_hash": "p",
        "target_hash": "t",
        "correct": correct,
        "decoded_generation": generation,
        "decode_diagnostics": {
            "public_example_guard": {
                "selected_name": name,
                "visible_examples_passed": {
                    "baseline": baseline_passes,
                    "parent": parent_passes,
                },
            }
        },
    }


def test_strict_baseline_selection_crosschecks():
    method = [record("baseline", True, "b", 2, 1)]
    parent = [record("parent", False, "p")]
    baseline = [record("baseline", True, "b")]
    summary = verify(method, parent, baseline)
    assert summary["all_checks_pass"]
    assert summary["independently_recomputed_correct"] == 1


def test_wrong_tie_selection_is_rejected():
    method = [record("baseline", True, "b", 1, 1)]
    parent = [record("parent", False, "p")]
    baseline = [record("baseline", True, "b")]
    summary = verify(method, parent, baseline)
    assert not summary["all_checks_pass"]
    assert summary["errors"] == ["1:selection_rule"]


if __name__ == "__main__":
    test_strict_baseline_selection_crosschecks()
    test_wrong_tie_selection_is_rejected()
    print("2 guard-replay crosscheck tests passed")
