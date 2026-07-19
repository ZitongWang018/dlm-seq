#!/usr/bin/env python3
import json
import tempfile
from pathlib import Path

from finalize_fair_three_arm import TASKS, finalize


def pair(baseline, method, baseline_nfe=1280, method_nfe=1200, baseline_wall=10, method_wall=9):
    return {
        "total": 10,
        "baseline_correct": baseline,
        "method_correct": method,
        "baseline_total_nfe": baseline_nfe,
        "method_total_nfe": method_nfe,
        "baseline_wall_seconds": baseline_wall,
        "method_wall_seconds": method_wall,
        "prompt_hash_mismatches": 0,
        "target_hash_mismatches": 0,
        "duplicate_or_missing_ids": 0,
        "source_hash_mismatches": 0,
        "source_hashes_verified": True,
    }


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tasks = {}
        for task in TASKS:
            tasks[task] = {}
            for arm, row in (
                ("accuracy", pair(5, 6, method_nfe=1400, method_wall=12)),
                ("fast", pair(5, 6)),
            ):
                pair_path = root / f"{task}_{arm}.json"
                pair_path.write_text(json.dumps(row), encoding="utf-8")
                input_path = root / f"{task}_{arm}_input.json"
                input_path.write_text(json.dumps({"all_equal": True}), encoding="utf-8")
                tasks[task][arm] = str(pair_path)
                tasks[task][f"{arm}_input_compare"] = str(input_path)
        result = finalize(
            {
                "tasks": tasks,
                "accuracy_profile": "accuracy",
                "fast_profile": "fast",
                "fallback_profile": "v11",
            }
        )
        assert result["selected_profile"] == "accuracy"
        # Force an accuracy regression; the fully faster arm should win.
        first = Path(tasks[TASKS[0]]["accuracy"])
        first.write_text(json.dumps(pair(5, 4)), encoding="utf-8")
        result = finalize(
            {
                "tasks": tasks,
                "accuracy_profile": "accuracy",
                "fast_profile": "fast",
                "fallback_profile": "v11",
            }
        )
        assert result["selected_profile"] == "fast"
        print("fair three-arm finalizer tests passed")


if __name__ == "__main__":
    main()
