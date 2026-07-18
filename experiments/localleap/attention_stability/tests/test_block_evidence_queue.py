from pathlib import Path


def script_text():
    return (
        Path(__file__).parents[1] / "scripts" / "run_block_evidence_queue.sh"
    ).read_text()


def test_holdout_is_stable_id_suffix_not_training_prefix():
    script = script_text()
    assert "--start 32 --end 64" in script
    assert "hm > hfast && hm > hbase" in script
    assert "DONE_NO_MATH" in script


def test_two_gpus_focus_humaneval_before_math():
    script = script_text()
    method = script.index("run_gpu he_block_n64 0")
    parents = script.index("run_gpu he_fast_n64 1")
    math = script.index("run_gpu math_block_n50 0")
    assert method < math and parents < math


if __name__ == "__main__":
    test_holdout_is_stable_id_suffix_not_training_prefix()
    test_two_gpus_focus_humaneval_before_math()
    print("2 block-evidence queue tests passed")
