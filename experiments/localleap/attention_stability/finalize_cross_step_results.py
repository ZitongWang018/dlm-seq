import argparse
import json
import re
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def find_samples(root):
    paths = sorted(Path(root).glob("**/samples_humaneval_*.jsonl"))
    paths = [path for path in paths if not str(path).endswith(".cleaned")]
    if len(paths) != 1:
        raise SystemExit(f"expected one samples file under {root}, found {len(paths)}")
    return paths[0]


def last_tps(log_path):
    if log_path is None or not Path(log_path).exists():
        return None
    matches = re.findall(
        r"^Tokens per second:\s*([0-9.]+)\s*$",
        Path(log_path).read_text(encoding="utf-8", errors="replace"),
        flags=re.MULTILINE,
    )
    return float(matches[-1]) if matches else None


def summarize_method(label, root, log_path, baseline_correct):
    root = Path(root)
    if not (root / "DONE").exists():
        raise SystemExit(f"incomplete method root: {root}")
    audit = read_json(root / "audit/audit_summary.json")
    diagnostics = read_json(root / "audit/step_diagnostics_summary.json")
    paired = read_json(root / "paired/paired_summary.json")
    audit_records = read_jsonl(root / "audit/audit_records.jsonl")
    trace_records = read_jsonl(root / "trace/rank_0.jsonl")
    if len(audit_records) != 164 or len(trace_records) != 164:
        raise SystemExit(f"record count mismatch for {label}")
    ids = [record["task_id"] for record in audit_records]
    if len(set(ids)) != 164:
        raise SystemExit(f"duplicate task ids for {label}")
    nfe_values = [int(record["nfe"]) for record in audit_records]
    if min(nfe_values) != 256 or max(nfe_values) != 256:
        raise SystemExit(f"NFE mismatch for {label}")
    core_settings = {
        key: audit_records[0]["generation_settings"].get(key)
        for key in (
            "model_path",
            "steps",
            "gen_length",
            "block_length",
            "temperature",
            "remasking",
        )
    }
    expected = {
        "model_path": "/root/autodl-tmp/model/LLaDA/instruct",
        "steps": 256,
        "gen_length": 256,
        "block_length": 32,
        "temperature": 0,
        "remasking": "low_confidence",
    }
    if core_settings != expected:
        raise SystemExit(f"configuration drift for {label}: {core_settings}")
    if any(
        {key: record["generation_settings"].get(key) for key in expected}
        != expected
        for record in audit_records
    ):
        raise SystemExit(f"within-run configuration drift for {label}")
    if diagnostics.get("schema_version") == "candidate_memory_steps_v2":
        if diagnostics["peak_runtime_full_probability_elements"] != 0:
            raise SystemExit(f"candidate method retained full-vocabulary history: {label}")
        if diagnostics["nfe_per_file_min"] != 256 or diagnostics["nfe_per_file_max"] != 256:
            raise SystemExit(f"candidate diagnostics NFE mismatch: {label}")
    else:
        if diagnostics["steps_per_file_min"] != 256 or diagnostics["steps_per_file_max"] != 256:
            raise SystemExit(f"attention diagnostics NFE mismatch: {label}")
    correct = int(audit["correct"])
    return {
        "label": label,
        "root": str(root),
        "samples": str(find_samples(root)),
        "correct": correct,
        "total": int(audit["total"]),
        "accuracy": float(audit["accuracy"]),
        "delta_correct_vs_baseline": correct - baseline_correct,
        "delta_accuracy_points_vs_baseline": 100.0
        * (float(audit["accuracy"]) - baseline_correct / 164.0),
        "total_nfe": sum(nfe_values),
        "nfe_min": min(nfe_values),
        "nfe_max": max(nfe_values),
        "tokens_per_second_health_only": last_tps(log_path),
        "core_generation_settings": core_settings,
        "paired_vs_baseline": paired,
        "diagnostics": diagnostics,
        "audit_health": {
            key: audit[key]
            for key in (
                "duplicate_ids",
                "missing_ids",
                "prompt_hash_mismatches",
                "generation_mismatches",
                "residual_mask_count",
            )
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llada-root", required=True)
    parser.add_argument("--queue-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    llada = Path(args.llada_root)
    queue_id = args.queue_id
    queue_root = llada / "results/experiment_queues" / queue_id
    specifications = [
        ("tau=0.004", f"results/attention_stability/tau0.004/{queue_id}_tau0p004", f"{queue_id}_tau0p004.log"),
        ("tau=0.0025", f"results/attention_stability/tau0.0025/{queue_id}_tau0p0025", f"{queue_id}_tau0p0025.log"),
        ("tau=0.001", f"results/attention_stability/tau0.001/{queue_id}_tau0p001", f"{queue_id}_tau0p001.log"),
        ("tau=0.0005", f"results/attention_stability/tau0.0005/{queue_id}_tau0p0005", f"{queue_id}_tau0p0005.log"),
        ("candidate-stability", f"results/candidate_memory/confidence/k8_delta0/{queue_id}_candidate_confidence_k8_delta0", f"{queue_id}_candidate_confidence_k8_delta0.log"),
        ("candidate-frontier", f"results/candidate_memory/frontier/k8_delta0/{queue_id}_candidate_frontier_k8_delta0", f"{queue_id}_candidate_frontier_k8_delta0.log"),
    ]
    baseline_correct = 67
    methods = [
        summarize_method(
            label,
            llada / relative_root,
            queue_root / log_name,
            baseline_correct,
        )
        for label, relative_root, log_name in specifications
    ]
    frontier_root = Path(methods[-1]["root"])
    direct_pair = read_json(frontier_root / "paired_vs_candidate_stability/paired_summary.json")
    result = {
        "schema_version": "cross_step_full_queue_summary_v1",
        "queue_id": queue_id,
        "baseline": {
            "correct": baseline_correct,
            "total": 164,
            "accuracy": baseline_correct / 164.0,
            "total_nfe": 41984,
        },
        "speed_note": (
            "TPS is health-only: candidate runs softmax only current masked rows while the old "
            "baseline/attention path softmaxes the full sequence, and diagnostics overhead differs."
        ),
        "methods": methods,
        "frontier_vs_candidate_stability": direct_pair,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "queue_id": queue_id,
        "scores": {item["label"]: item["correct"] for item in methods},
        "frontier_vs_stability": {
            key: direct_pair[key]
            for key in ("method_only", "baseline_only", "exact_mcnemar_p")
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
