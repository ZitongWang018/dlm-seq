import copy

from apply_public_example_guard import apply_guard, summarize


def record(task_id, generation, correct, nfe):
    index = int(task_id.rsplit("/", 1)[1])
    return {
        "absolute_index": index,
        "task_id": task_id,
        "prompt_hash": "same-prompt",
        "prompt_text": "def square(x):\n    >>> square(2)\n    4\n",
        "entry_point": "square",
        "raw_gold": "gold",
        "normalized_gold": "gold",
        "decoded_generation": generation,
        "correct": correct,
        "nfe": nfe,
        "generation_settings": {},
        "decode_diagnostics": {
            "selected_name": "fast",
            "candidate_nfe": {"fast": 128, "accuracy": 140},
        },
    }


def test_replay_selects_strict_public_example_winner_and_sums_nfe():
    parent = {
        "HumanEval/0": record(
            "HumanEval/0", "def square(x):\n    return x + 1", False, 270
        )
    }
    baseline = {
        "HumanEval/0": record(
            "HumanEval/0", "def square(x):\n    return x * x", True, 128
        )
    }
    output = apply_guard(parent, baseline)
    assert output[0]["correct"] is True
    assert output[0]["nfe"] == 398
    assert output[0]["prompt_text"] == parent["HumanEval/0"]["prompt_text"]
    assert output[0]["entry_point"] == "square"
    assert output[0]["decode_diagnostics"]["selected_name"] == "baseline"
    summary = summarize(output, parent, baseline)
    assert summary["method_correct"] == 1
    assert summary["method_only_vs_parent"] == 1


def test_replay_preserves_parent_on_tie():
    parent_record = record(
        "HumanEval/0", "def square(x):\n    return x ** 2", True, 270
    )
    baseline_record = record(
        "HumanEval/0", "def square(x):\n    return x * x", True, 128
    )
    output = apply_guard(
        {"HumanEval/0": parent_record}, {"HumanEval/0": baseline_record}
    )
    assert output[0]["decoded_generation"] == parent_record["decoded_generation"]
    assert output[0]["decode_diagnostics"]["selected_name"] == "fast"


def test_alignment_mismatch_is_rejected():
    parent = {"HumanEval/0": record("HumanEval/0", "", False, 1)}
    baseline = copy.deepcopy(parent)
    baseline["HumanEval/0"]["prompt_hash"] = "different"
    try:
        apply_guard(parent, baseline)
        raise AssertionError("mismatched prompt hash was accepted")
    except ValueError as error:
        assert "prompt_hash mismatch" in str(error)


if __name__ == "__main__":
    test_replay_selects_strict_public_example_winner_and_sums_nfe()
    test_replay_preserves_parent_on_tie()
    test_alignment_mismatch_is_rejected()
    print("3 public-example replay tests passed")
