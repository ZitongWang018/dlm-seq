#!/usr/bin/env python3
import argparse
import json
import math
import statistics
from pathlib import Path


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stable_id(row):
    stable = row.get("stable_task_id")
    task = row.get("task_id")
    if stable is not None and task is not None and str(stable) != str(task):
        raise ValueError(f"conflicting stable_task_id/task_id: {stable!r} != {task!r}")
    value = stable if stable is not None else task
    if value is None:
        raise ValueError("record has neither stable_task_id nor task_id")
    return str(value)


def index_records(rows):
    indexed = {}
    for row in rows:
        key = stable_id(row)
        if key in indexed:
            raise ValueError(f"duplicate task id: {key}")
        indexed[key] = row
    return indexed


def metric_row(trace_record):
    item = trace_record["decode_diagnostics"]
    frontier = item["frontier"]
    fill = item["fill"]
    repair = item["repair"]
    source_margins = [
        source["outgoing_reader_count"] - source["incoming_source_count"]
        for source in repair["source_first"]
    ]
    return {
        "selected_forced_commits": frontier["selected_forced_commits"],
        "selected_immature_commits": frontier["selected_immature_commits"],
        "selected_response_invalidations": frontier[
            "selected_response_invalidations"
        ],
        "mean_selected_incident_count": frontier[
            "mean_incident_count_per_remasked_position"
        ],
        "fill_response_invalidations": fill["response_invalidations"],
        "repair_response_invalidations": repair["response_invalidations"],
        "repair_response_validations": repair["response_validations"],
        "repair_revision_margin_candidates": repair[
            "revision_margin_candidates"
        ],
        "revised_token_count": repair["revised_token_count"],
        "mean_source_direction_margin": (
            statistics.fmean(source_margins) if source_margins else 0.0
        ),
    }


def summarize_group(rows, metric_names):
    result = {"count": len(rows)}
    for metric in metric_names:
        values = [row[metric] for row in rows]
        result[metric] = {
            "mean": statistics.fmean(values) if values else 0.0,
            "median": statistics.median(values) if values else 0.0,
        }
    return result


def pearson(values, labels):
    if len(values) < 2 or len(set(values)) < 2 or len(set(labels)) < 2:
        return None
    mean_x = statistics.fmean(values)
    mean_y = statistics.fmean(labels)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(values, labels))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in values))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in labels))
    return numerator / (denom_x * denom_y) if denom_x and denom_y else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("method_records")
    parser.add_argument("--paired-records")
    parser.add_argument("--output")
    args = parser.parse_args()

    traces = index_records(read_jsonl(args.trace))
    method = index_records(read_jsonl(args.method_records))
    if set(traces) != set(method):
        raise SystemExit("trace and method audit ids differ")
    paired = None
    if args.paired_records:
        paired = index_records(read_jsonl(args.paired_records))
        if set(paired) != set(method):
            raise SystemExit("paired and method audit ids differ")

    rows = []
    for stable_id in sorted(method):
        row = {
            "stable_task_id": stable_id,
            "method_correct": bool(method[stable_id]["correct"]),
            **metric_row(traces[stable_id]),
        }
        if paired is not None:
            baseline_correct = bool(paired[stable_id]["baseline_correct"])
            method_correct = bool(paired[stable_id]["method_correct"])
            if baseline_correct and method_correct:
                outcome = "both_correct"
            elif method_correct:
                outcome = "method_only"
            elif baseline_correct:
                outcome = "baseline_only"
            else:
                outcome = "both_wrong"
            row["paired_outcome"] = outcome
        rows.append(row)

    metric_names = [
        key
        for key in rows[0]
        if key not in {"stable_task_id", "method_correct", "paired_outcome"}
    ]
    labels = [int(row["method_correct"]) for row in rows]
    summary = {
        "schema_version": "response_refine_outcome_analysis_v1",
        "records": len(rows),
        "correct": sum(labels),
        "groups_by_method_correct": {
            "correct": summarize_group(
                [row for row in rows if row["method_correct"]], metric_names
            ),
            "incorrect": summarize_group(
                [row for row in rows if not row["method_correct"]], metric_names
            ),
        },
        "point_biserial_correlations_with_correctness": {
            metric: pearson([row[metric] for row in rows], labels)
            for metric in metric_names
        },
        "records_detail": rows,
    }
    if paired is not None:
        summary["groups_by_paired_outcome"] = {
            outcome: summarize_group(
                [row for row in rows if row["paired_outcome"] == outcome],
                metric_names,
            )
            for outcome in ("method_only", "baseline_only", "both_correct", "both_wrong")
        }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
