import copy
import multiprocessing

from apply_public_frontier_guard import configure_runtime_path, select_records


PROMPT = "def square(x):\n    >>> square(2)\n    4\n"


def base_record(name, generation, correct=False):
    return {
        "absolute_index": 0,
        "task_id": "HumanEval/0",
        "prompt_hash": "prompt",
        "target_hash": "target",
        "prompt_text": PROMPT,
        "entry_point": "square",
        "decoded_generation": generation,
        "correct": correct,
        "nfe": 400,
        "generation_settings": {},
        "decode_diagnostics": {"selected_name": name},
    }


def records(fast, accuracy, baseline, selected_name="fast"):
    v9 = base_record(selected_name, fast if selected_name == "fast" else accuracy)
    v9["candidate_generations"] = {"fast": fast, "accuracy": accuracy}
    base = base_record("baseline", baseline)
    selected_generation = {
        "fast": fast,
        "accuracy": accuracy,
        "baseline": baseline,
    }[selected_name]
    v11 = base_record(selected_name, selected_generation)
    return [v9], [base], [v11]


def test_unique_public_winner_reopens_unselected_sibling():
    v9, baseline, v11 = records(
        "def square(x):\n    return x + 1",
        "def square(x):\n    return x * x",
        "def square(x):\n    return 0",
    )
    baseline[0].pop("prompt_text")
    baseline[0].pop("entry_point")
    output = select_records(v9, baseline, v11)[0]
    assert output["decode_diagnostics"]["selected_name"] == "accuracy"
    assert output["decode_diagnostics"]["public_frontier_guard"]["status"] == (
        "unique_public_winner_reopens_frontier"
    )


def test_tied_public_leaders_preserve_v11():
    v9, baseline, v11 = records(
        "def square(x):\n    return x * x",
        "def square(x):\n    return x ** 2",
        "def square(x):\n    return 0",
    )
    output = select_records(v9, baseline, v11)[0]
    assert output["decode_diagnostics"]["selected_name"] == "fast"


def test_no_public_checks_preserves_v11():
    v9, baseline, v11 = records(
        "def square(x):\n    return x + 1",
        "def square(x):\n    return x * x",
        "def square(x):\n    return 0",
    )
    for row in v9 + baseline + v11:
        row["prompt_text"] = "def square(x):\n    pass\n"
    output = select_records(v9, baseline, v11)[0]
    assert output["decode_diagnostics"]["selected_name"] == "fast"


def test_alignment_mismatch_rejected():
    v9, baseline, v11 = records("a", "b", "c")
    baseline = copy.deepcopy(baseline)
    baseline[0]["prompt_hash"] = "other"
    try:
        select_records(v9, baseline, v11)
        raise AssertionError("alignment mismatch was accepted")
    except ValueError as error:
        assert "prompt_hash mismatch" in str(error)


def spawned_selector_probe(queue):
    import differential_selector

    queue.put(
        (
            hasattr(differential_selector, "evaluate_public_candidate"),
            differential_selector.__file__,
        )
    )


def test_runtime_path_does_not_shadow_selector_in_spawned_executor():
    configure_runtime_path("/root/autodl-tmp/LocalLeap/llada")
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=spawned_selector_probe, args=(queue,))
    process.start()
    process.join(10)
    assert process.exitcode == 0
    has_evaluator, path = queue.get(timeout=1)
    assert has_evaluator, path
    assert "attention_stability/differential_selector.py" in path, path


if __name__ == "__main__":
    test_unique_public_winner_reopens_unselected_sibling()
    test_tied_public_leaders_preserve_v11()
    test_no_public_checks_preserves_v11()
    test_alignment_mismatch_rejected()
    test_runtime_path_does_not_shadow_selector_in_spawned_executor()
    print("5 public-frontier guard tests passed")
