from pathlib import Path


SCRIPT = Path(__file__).parent / "scripts" / "run_best_framework_full4_queue.sh"


def main():
    text = SCRIPT.read_text(encoding="utf-8")
    required = {
        "HumanEval full": "he_method_full164",
        "MATH-500 full": "math_method_full500",
        "GSM8K full": "gsm_method_full1319",
        "MBPP full": "mbpp_method_full500",
        "original baseline": "baseline 0 trace",
        "v15 exact gate": 'row["all_invariants_pass"]',
        "strict speed gate": 'row["nfe_reduction"] > 0',
        "single v11 fallback": "selected_profile=trajectory_confirmed_public_guard",
        "MBPP execution audit": "audit_mbpp_assertions.py",
        "stable paired audit": "compare_paired_task_runs.py",
    }
    missing = [name for name, token in required.items() if token not in text]
    assert not missing, missing
    assert text.count("run_gpu ") == 7
    assert "--limit" not in text
    assert "profile_for" not in text
    assert "trajectory_confirmed_bidirectional_block" not in text
    assert text.count('"${selected_profile}" 0.004 trace') == 4
    assert "audit_generation_leakage.py" in text
    print("full4 queue contract: OK")


if __name__ == "__main__":
    main()
