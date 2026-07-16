import json
import os
import tempfile

from select_queue_profile import beats, choose_best


def summary(correct, nfe=1.0, wall=1.0, healthy=True):
    mismatch = 0 if healthy else 1
    return {
        "total": 64,
        "method_correct": correct,
        "nfe_ratio_method_over_baseline": nfe,
        "wall_speedup_baseline_over_method": wall,
        "prompt_hash_mismatches": mismatch,
        "target_hash_mismatches": 0,
        "duplicate_or_missing_ids": 0,
    }


def write(root, name, value):
    path = os.path.join(root, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle)
    return path


def test_exact_tie_keeps_established_parent():
    with tempfile.TemporaryDirectory() as root:
        parent = write(root, "parent.json", summary(20, wall=0.9))
        child = write(root, "child.json", summary(20, wall=1.1))
        selected = choose_best(
            [("symmetric_fast", parent), ("revision_margin_fast", child)]
        )
        assert selected["profile"] == "symmetric_fast"


def test_new_rule_wins_only_with_more_correct_answers():
    with tempfile.TemporaryDirectory() as root:
        parent = write(root, "parent.json", summary(20))
        child = write(root, "child.json", summary(21))
        selected = choose_best(
            [("symmetric_fast", parent), ("revision_margin_fast", child)]
        )
        assert selected["profile"] == "revision_margin_fast"
        assert beats(summary(21), summary(20), required_gain=1)
        assert not beats(summary(20), summary(20), required_gain=1)


def test_unhealthy_summary_is_never_selected_or_promoted():
    with tempfile.TemporaryDirectory() as root:
        parent = write(root, "parent.json", summary(20))
        corrupt = write(root, "corrupt.json", summary(64, healthy=False))
        selected = choose_best(
            [("symmetric_fast", parent), ("revision_margin_fast", corrupt)]
        )
        assert selected["profile"] == "symmetric_fast"
        assert not beats(summary(64, healthy=False), summary(20))


if __name__ == "__main__":
    test_exact_tie_keeps_established_parent()
    test_new_rule_wins_only_with_more_correct_answers()
    test_unhealthy_summary_is_never_selected_or_promoted()
    print("3 queue-profile selection tests passed")
