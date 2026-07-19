import argparse
import datetime as dt
import json
import math
import re
from pathlib import Path


VERSION = "localleap_paired_task_audit_v4"


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_stable_id(row):
    """Return the evaluator's stable identity across code and non-code tasks."""
    identities = [
        str(row[key])
        for key in ("stable_task_id", "task_id")
        if row.get(key) is not None
    ]
    if not identities:
        raise ValueError("record has neither stable_task_id nor task_id")
    if len(set(identities)) != 1:
        raise ValueError(f"conflicting stable identities: {identities}")
    return identities[0]


def index_records(rows):
    indexed = {}
    for row in rows:
        stable_id = record_stable_id(row)
        if stable_id in indexed:
            raise ValueError(f"duplicate stable identity: {stable_id}")
        indexed[stable_id] = row
    return indexed


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


def config_source_hashes(path):
    """Read frozen source hashes embedded in a benchmark run config."""
    hashes = {}
    pattern = re.compile(r"^([0-9a-f]{64})\s+(.+)$")
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            hashes[Path(match.group(2)).name] = match.group(1)
    required = {"generate.py", "eval_llada.py"}
    missing = sorted(required - set(hashes))
    if missing:
        raise ValueError(f"run config is missing source hashes: {missing}")
    return hashes


def compare_source_hashes(baseline_config, method_config):
    baseline = config_source_hashes(baseline_config)
    method = config_source_hashes(method_config)
    common = sorted(set(baseline) & set(method))
    mismatches = [name for name in common if baseline[name] != method[name]]
    return {
        "common": common,
        "mismatches": mismatches,
        "baseline_only": sorted(set(baseline) - set(method)),
        "method_only": sorted(set(method) - set(baseline)),
    }


def verify_matching_source_hashes(baseline_config, method_config):
    comparison = compare_source_hashes(baseline_config, method_config)
    mismatches = comparison["mismatches"]
    if mismatches:
        raise ValueError(f"source hash mismatch: {mismatches}")
    return comparison["common"]


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
    parser.add_argument(
        "--allow-source-drift",
        action="store_true",
        help=(
            "permit an explicitly reported source mismatch for cross-version "
            "algorithm comparisons; record identity and prompt/target hashes "
            "remain strict"
        ),
    )
    args = parser.parse_args()

    try:
        source_comparison = compare_source_hashes(
            args.baseline_config, args.method_config
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    source_set_drift = (
        source_comparison["baseline_only"] or source_comparison["method_only"]
    )
    if (source_comparison["mismatches"] or source_set_drift) and not args.allow_source_drift:
        raise SystemExit(
            "source manifest mismatch: "
            f"changed={source_comparison['mismatches']} "
            f"baseline_only={source_comparison['baseline_only']} "
            f"method_only={source_comparison['method_only']}"
        )

    baseline = index_records(read_jsonl(args.baseline_records))
    method = index_records(read_jsonl(args.method_records))
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
        if not left.get("evaluator_version") or not right.get("evaluator_version"):
            raise SystemExit(f"missing evaluator version: {stable_id}")
        if left["evaluator_version"] != right["evaluator_version"]:
            raise SystemExit(f"evaluator version mismatch: {stable_id}")
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
        "source_hashes_verified": not source_comparison["mismatches"] and not source_set_drift,
        "source_drift_explicitly_allowed": bool(args.allow_source_drift),
        "source_hash_file_count": len(source_comparison["common"]),
        "source_hash_mismatches": len(source_comparison["mismatches"]),
        "source_hash_mismatch_files": source_comparison["mismatches"],
        "baseline_only_source_files": source_comparison["baseline_only"],
        "method_only_source_files": source_comparison["method_only"],
        "evaluator_version_mismatches": 0,
        "record_evaluator_versions": sorted(
            {row["evaluator_version"] for row in baseline.values()}
        ),
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
