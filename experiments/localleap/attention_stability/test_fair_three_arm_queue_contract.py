#!/usr/bin/env python3
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "scripts" / "run_fair_three_arm_reproduction_queue.sh"
PREREG = ROOT / "fair_three_arm_preregistration_20260719_v1.json"


def main():
    text = SCRIPT.read_text(encoding="utf-8")
    required = (
        "HF_DATASETS_OFFLINE=1",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "run_task_pair humaneval 0",
        "run_task_pair localleap_math500 0",
        "run_task_pair gsm8k 0",
        "run_task_pair mbpp 0",
        "baseline 0 trace",
        "symmetric_fast 0.004 trace",
        "audit_model_input_hashes.py",
        "model_weights.sha256",
        "trajectory_confirmed_public_guard_v11_family",
    )
    for token in required:
        assert token in text, token
    assert "--allow-source-drift" not in text
    assert "num_fewshot" not in text
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    assert prereg["single_algorithm_requirement"] is True
    assert prereg["task_specific_routing_forbidden"] is True
    assert set(prereg["shared_generation"]["fewshot"].values()) == {0}
    print("fair three-arm queue contract tests passed")


if __name__ == "__main__":
    main()
