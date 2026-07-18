from pathlib import Path


def queue_script():
    return (
        Path(__file__).parents[1] / "scripts" / "run_trajectory_likelihood_queue.sh"
    ).read_text()


def test_discovery_runs_are_launched_before_required():
    script = queue_script()
    required = (
        "he_base_n32",
        "he_fast_n32",
        "he_likelihood_n32",
        "math_base_n50",
        "math_fast_n50",
        "math_likelihood_n50",
    )
    require_offset = script.index("require_done", script.index("he_likelihood_n32"))
    for stage in required:
        launch_offset = script.index(f"run_gpu {stage} ")
        assert launch_offset < require_offset, stage


def test_queue_uses_two_distinct_gpus_and_strict_gate():
    script = queue_script()
    assert "smoke_he_likelihood 0" in script
    assert "smoke_math_likelihood 1" in script
    assert "hm + mm > hp + mp" in script
    assert "hmo + mmo > 0" in script


if __name__ == "__main__":
    test_discovery_runs_are_launched_before_required()
    test_queue_uses_two_distinct_gpus_and_strict_gate()
    print("2 trajectory-likelihood queue tests passed")
