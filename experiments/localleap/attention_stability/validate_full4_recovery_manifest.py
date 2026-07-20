#!/usr/bin/env python3
"""Validate that a failed full4 queue is safe for evaluator-only recovery."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_GENERATION_STAGES = {
    "he_method_full164",
    "gsm_method_full1319",
    "math_baseline_full500",
    "mbpp_baseline_full500",
    "math_method_full500",
    "gsm_baseline_full1319",
    "mbpp_method_full500",
}


def validate(manifest: Path, queue_root: Path) -> dict[str, object]:
    rows = list(csv.DictReader(manifest.open(), delimiter="\t"))
    done = {row["stage"] for row in rows if row["status"] == "DONE"}
    missing = sorted(REQUIRED_GENERATION_STAGES - done)
    if missing:
        raise ValueError(f"generation_incomplete: {missing}")

    failed = [row["stage"] for row in rows if row["status"] == "FAILED"]
    generation_failures = sorted(set(failed) & REQUIRED_GENERATION_STAGES)
    if generation_failures:
        raise ValueError(f"generation_failed: {generation_failures}")
    non_leakage_failures = sorted(
        stage for stage in failed if not stage.startswith("leakage_")
    )
    if non_leakage_failures:
        raise ValueError(f"non_leakage_failure: {non_leakage_failures}")

    evidence: list[str] = []
    if failed:
        evidence.extend(f"manifest:{stage}" for stage in failed)

    started = {row["stage"] for row in rows if row["status"] == "STARTED"}
    terminal = {
        row["stage"]
        for row in rows
        if row["status"] in {"DONE", "FAILED"}
    }
    interrupted = sorted(
        stage
        for stage in started - terminal
        if stage.startswith("leakage_") and stage != "leakage_static"
    )
    for stage in interrupted:
        report_name = stage.removeprefix("leakage_") + ".json"
        report = queue_root / "leakage" / report_name
        if not report.is_file():
            continue
        payload = json.loads(report.read_text())
        if payload.get("pass") is False:
            evidence.append(f"interrupted_report:{stage}:{report_name}")

    if not (queue_root / "FAILED").exists():
        raise ValueError("parent_failed_marker_missing")
    if not evidence:
        raise ValueError("no_evaluator_failure_evidence")

    return {
        "schema": "full4_recovery_validation_v2",
        "generation_stages_complete": len(REQUIRED_GENERATION_STAGES),
        "generation_failures": generation_failures,
        "recoverable_evaluator_failure_evidence": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.manifest, args.queue_root)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
