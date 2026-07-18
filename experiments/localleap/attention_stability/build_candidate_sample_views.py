#!/usr/bin/env python3
"""Build versioned lm-eval sample views for trajectory candidates.

The source sample order and task metadata are preserved exactly.  Only the
decoded response is replaced with the already-recorded candidate trajectory,
so the normal HumanEval postprocessor can evaluate every candidate through the
same execution path as the selected output.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def task_id_from_sample(sample: dict) -> str:
    return sample["doc"]["task_id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidates", nargs="+", default=["fast", "accuracy", "baseline"]
    )
    args = parser.parse_args()

    samples = read_jsonl(args.samples)
    traces = read_jsonl(args.trace)
    sample_by_id = {task_id_from_sample(row): row for row in samples}
    trace_by_id = {row["task_id"]: row for row in traces}
    if len(sample_by_id) != len(samples) or len(trace_by_id) != len(traces):
        raise SystemExit("duplicate task IDs")
    if set(sample_by_id) != set(trace_by_id):
        raise SystemExit("sample/trace task IDs do not align")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    for candidate in args.candidates:
        output = args.output_dir / f"{candidate}.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for task_id in sorted(
                sample_by_id, key=lambda value: int(value.split("/")[-1])
            ):
                trace = trace_by_id[task_id]
                generations = trace.get("candidate_generations") or {}
                if candidate not in generations:
                    raise SystemExit(f"missing {candidate} generation for {task_id}")
                row = copy.deepcopy(sample_by_id[task_id])
                row["resps"][0][0] = generations[candidate]
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "sample_count": len(samples),
        "candidates": args.candidates,
        "source_samples": str(args.samples),
        "source_trace": str(args.trace),
        "mutation": "resps[0][0] only",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
