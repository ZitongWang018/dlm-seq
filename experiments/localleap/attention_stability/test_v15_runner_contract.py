import subprocess
from pathlib import Path


def main():
    runner = Path(__file__).with_name("scripts") / "run_best_symmetric_benchmark.sh"
    completed = subprocess.run(
        [
            "bash",
            str(runner),
            "humaneval",
            "0",
            "64",
            "trajectory_early_lazy_confirmed_public_guard",
            "0.004",
            "trace",
            "contract_test_must_not_run",
            "4",
            "256",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2, completed
    assert "requires steps=128 block_length=32" in completed.stderr, completed
    assert "unbound variable" not in completed.stderr, completed
    print("v15 runner profile contract passed")


if __name__ == "__main__":
    main()
