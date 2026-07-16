import argparse
import datetime as dt
import json
import math
import re
from pathlib import Path


VERSION = "localleap_paired_task_audit_v1"


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def exact_mcnemar_p(baseline_only, method_only):
    discordant = baseline_only + method_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k) for k in range(min(baseline_only, method_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def config_duration_seconds(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    if "start" not in values or "finish" not in values:
        return None
    start = dt.datetime.fromisoformat(values["start"])
    finish = dt.datetime.fromisoformat(values["finish"])
    return (finish - start).total_seconds()


def log_metrics(path):
    if path is None:
        return {}
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    metrics = {}
    patterns = {
        "model_total_seconds": r"Total time:\s*([0-9.]+)\s*seconds",
        "tokens_per_second": r"Tokens per second:\s*([0-9.]+)",
        "reported_total_nfe": r"Total NFE:\s*([0-9]+)",
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            value = matches[-1]
            metrics[key] = float(value) if "." in value else int(value)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_records")
    parser.add_argument("method_records")
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--method-config", required=True)
    parser.add_argument("--baseline-log")
    parser.add_argument("--method-log")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    baseline = {row["stable_task_id"]: row for row in read_jsonl(args.baseline_records)}
    method = {row["stable_task_id"]: row for row in read_jsonl(args.method_records)}
    if set(baseline) != set(method):
        missing_method = sorted(set(baseline) - set(method))
        missing_baseline = sorted(set(method) - set(baseline))
        raise SystemExit(
            f"stable id mismatch: missing_method={missing_method[:5]}, "
            f"missing_baseline={missing_baseline[:5]}"
        )

    paired = []
    for stable_id in sorted(baseline):
        left = baseline[stable_id]
        right = method[stable_id]
        if left["prompt_hash"] != right["prompt_hash"]:
            raise SystemExit(f"prompt hash mismatch: {stable_id}")
        if left["target_hash"] != right["target_hash"]:
            raise SystemExit(f"target hash mismatch: {stable_id}")
        paired.append(
            {
                "stable_task_id": stable_id,
                "prompt_hash": left["prompt_hash"],
                "target_hash": left["target_hash"],
                "baseline_correct": left["correct"],
                "method_correct": right["correct"],
                "baseline_nfe": left["nfe"],
                "method_nfe": right["nfe"],
            }
        )

    baseline_correct = sum(row["baseline_correct"] for row in paired)
    method_correct = sum(row["method_correct"] for row in paired)
    baseline_only = sum(
        row["baseline_correct"] and not row["method_correct"] for row in paired
    )
    method_only = sum(
        row["method_correct"] and not row["baseline_correct"] for row in paired
    )
    baseline_nfe = sum(row["baseline_nfe"] or 0 for row in paired)
    method_nfe = sum(row["method_nfe"] or 0 for row in paired)
    baseline_duration = config_duration_seconds(args.baseline_config)
    method_duration = config_duration_seconds(args.method_config)
    summary = {
        "evaluator_version": VERSION,
        "total": len(paired),
        "baseline_correct": baseline_correct,
        "method_correct": method_correct,
        "baseline_accuracy": baseline_correct / len(paired),
        "method_accuracy": method_correct / len(paired),
        "absolute_gain": (method_correct - baseline_correct) / len(paired),
        "baseline_only": baseline_only,
        "method_only": method_only,
        "exact_mcnemar_p": exact_mcnemar_p(baseline_only, method_only),
        "baseline_total_nfe": baseline_nfe,
        "method_total_nfe": method_nfe,
        "nfe_ratio_method_over_baseline": (
            method_nfe / baseline_nfe if baseline_nfe else None
        ),
        "baseline_wall_seconds": baseline_duration,
        "method_wall_seconds": method_duration,
        "wall_speedup_baseline_over_method": (
            baseline_duration / method_duration
            if baseline_duration is not None and method_duration
            else None
        ),
        "baseline_log_metrics": log_metrics(args.baseline_log),
        "method_log_metrics": log_metrics(args.method_log),
        "prompt_hash_mismatches": 0,
        "target_hash_mismatches": 0,
        "duplicate_or_missing_ids": 0,
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "paired_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in paired:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "paired_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

