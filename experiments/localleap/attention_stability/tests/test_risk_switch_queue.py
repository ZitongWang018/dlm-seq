from pathlib import Path


CONTROLLER = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_risk_switch_queue.sh"
)


def main():
    source = CONTROLLER.read_text()
    checks = {
        "he_base_n32": '"${run_base}/humaneval/he_base_n32"',
        "math_base_n50": '"${run_base}/localleap_math500/math_base_n50"',
        "he_base_n64": '"${run_base}/humaneval/he_base_n64"',
        "math_base_n100": '"${run_base}/localleap_math500/math_base_n100"',
        "he_base_full": '"${run_base}/humaneval/he_base_full"',
        "math_base_full": '"${run_base}/localleap_math500/math_base_full"',
    }
    for stage, required_path in checks.items():
        launch = source.index(f"run_gpu {stage}")
        requirement = source.index(required_path)
        assert launch < requirement, f"{stage} is required before it is launched"
    assert source.index("smoke_he_switch") < source.index("FAILED_SMOKE")
    assert source.index("smoke_math_switch") < source.index("FAILED_SMOKE")
    print("risk-switch queue order checks passed")


if __name__ == "__main__":
    main()
