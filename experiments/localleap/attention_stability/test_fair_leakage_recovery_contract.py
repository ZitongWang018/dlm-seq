#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    script = (
        ROOT / "scripts" / "run_fair_three_arm_leakage_recovery_v2.sh"
    ).read_text(encoding="utf-8")
    assert "audit_generation_leakage_v2.py" in script
    assert "generation_information_leakage_audit_v2" in script
    assert "original_summary_preserved" in script
    assert "source_hash_mismatches" in script
    assert "task_specific_routing" in script
    assert "baseline_generation_basis" in script
    for label in (
        "he_accuracy", "math_accuracy", "gsm_accuracy", "mbpp_accuracy",
        "he_fast", "math_fast", "gsm_fast", "mbpp_fast",
    ):
        assert f"audit_run {label}" in script
    assert "RECOVERED_BY_LEAKAGE_V2" in script
    print("fair leakage recovery contract passed")


if __name__ == "__main__":
    main()
