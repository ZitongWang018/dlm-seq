from audit_mbpp_challenge import audit_record


def make_sample(generation):
    return {
        "doc": {
            "task_id": 11,
            "challenge_test_list": [
                "assert square(2) == 4",
                "assert square(-3) == 9",
            ],
            "test_setup_code": "",
        },
        "prompt_hash": "p",
        "target_hash": "t",
        "resps": [[generation]],
    }


def task(generation):
    return {
        "absolute_index": 0,
        "stable_task_id": "11",
        "prompt_hash": "p",
        "target_hash": "t",
        "decoded_generation": generation,
        "raw_gold": "def square(x): return x*x",
        "nfe": 128,
    }


def test_correct_candidate_passes_both_challenge_paths():
    generation = "def square(x):\n    return x*x"
    row = audit_record(make_sample(generation), task(generation))
    assert row["correct"] is True
    assert row["challenge_diagnostics"]["crosscheck_match"] is True
    assert row["challenge_diagnostics"]["selection_used_challenge_tests"] is False


def test_visible_overfit_candidate_fails_hidden_challenge():
    generation = "def square(x):\n    return 4"
    row = audit_record(make_sample(generation), task(generation))
    assert row["correct"] is False
    assert row["challenge_diagnostics"]["crosscheck_match"] is True


if __name__ == "__main__":
    test_correct_candidate_passes_both_challenge_paths()
    test_visible_overfit_candidate_fails_hidden_challenge()
    print("2 MBPP challenge-audit tests passed")
