#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "scripts" / "run_full4_leakage_recovery_and_resume.sh"


def main():
    text = SCRIPT.read_text(encoding="utf-8")
    for token in (
        "generation_information_leakage_audit_v2",
        "validate_recoverable_failure",
        "all(stage.startswith(\"leakage_\")",
        "RECOVERED_BY_LEAKAGE_V2",
        "original_failed_marker_preserved",
        "resume_downstream",
        "run_localized_evidence_conflict_repair_queue.sh",
        "run_fair_three_arm_reproduction_queue.sh",
    ):
        assert token in text, token
    assert "rm " not in text
    assert "rm -" not in text
    assert "--allow-source-drift" in text
    assert "touch \"${full4_queue}/DONE\"" in text
    assert "FAILED\"" not in text.split("RECOVERED_BY_LEAKAGE_V2")[-1]
    print("full4 leakage recovery contract tests passed")


if __name__ == "__main__":
    main()
