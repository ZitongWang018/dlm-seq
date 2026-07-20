#!/usr/bin/env python3
"""Infrastructure contract for the versioned v18/v19 repair queue."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    controller = (
        ROOT / "scripts" / "run_localized_evidence_conflict_repair_queue.sh"
    ).read_text(encoding="utf-8")
    slicer = (ROOT / "slice_audit_by_index.py").read_text(encoding="utf-8")

    assert controller.count("slice_audit_by_index.py") >= 4
    assert "slice_audit_records.py" not in controller
    assert "test_slice_audit_by_index.py" in controller
    assert "HF_DATASETS_OFFLINE=1" in controller
    assert "HF_HUB_OFFLINE=1" in controller
    assert "TRANSFORMERS_OFFLINE=1" in controller
    assert "HF_EVALUATE_OFFLINE=1" in controller
    for helper in ("postprocess_code.py", "humaneval_execution.py", "sanitize.py"):
        assert helper in controller
    assert 'record["absolute_index"]' in slicer
    assert "refusing to overwrite" in slicer
    print("localized repair queue contract passed")


if __name__ == "__main__":
    main()
