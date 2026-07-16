import argparse
import hashlib
import json
from pathlib import Path


VERSION = "stcc_lm_eval_record_audit_v2"


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_filter_records(samples, filter_name):
    if filter_name is None:
        return samples
    selected = [sample for sample in samples if sample.get("filter") == filter_name]
    if not selected:
        available = sorted({sample.get("filter") for sample in samples})
        raise ValueError(f"filter {filter_name!r} not found; available={available}")
    return selected


def prompt_from_sample(sample):
    arguments = sample.get("arguments", {})
    for value in arguments.values():
        if isinstance(value, dict) and "arg_0" in value:
            return value["arg_0"]
    doc = sample.get("doc", {})
    return doc.get("prompt") or doc.get("problem") or doc.get("text") or ""


def metric_value(sample, metric):
    if metric in sample:
        return float(sample[metric])
    normalized = metric.lower().replace("@", "_at_")
    candidates = []
    for key, value in sample.items():
        key_normalized = key.lower().replace("@", "_at_")
        if normalized in key_normalized and isinstance(value, (int, float, bool)):
            candidates.append((key, float(value)))
    if len(candidates) != 1:
        raise KeyError(f"cannot identify metric {metric}; candidates={candidates}")
    return candidates[0][1]


def aggregate_value(results, task, metric):
    task_results = results["results"][task]
    normalized = metric.lower().replace("@", "_at_")
    candidates = []
    for key, value in task_results.items():
        key_normalized = key.lower().replace("@", "_at_")
        if normalized in key_normalized and "stderr" not in key_normalized:
            if isinstance(value, (int, float)):
                candidates.append((key, float(value)))
    if len(candidates) != 1:
        raise KeyError(f"cannot identify aggregate metric {metric}; candidates={candidates}")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("samples")
    parser.add_argument("results")
    parser.add_argument("--task", required=True)
    parser.add_argument("--primary-metric", required=True)
    parser.add_argument("--filter")
    parser.add_argument("--trace")
    parser.add_argument("--constant-nfe", type=int)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.trace and args.constant_nfe is not None:
        raise SystemExit("use either --trace or --constant-nfe, not both")

    samples = select_filter_records(read_jsonl(args.samples), args.filter)
    if len(samples) != args.expected_records:
        raise SystemExit(f"expected {args.expected_records} samples, found {len(samples)}")
    traces = read_jsonl(args.trace) if args.trace else None
    if traces is not None and len(traces) != len(samples):
        raise SystemExit("trace/sample count mismatch")
    trace_by_index = (
        {int(record["absolute_index"]): record for record in traces}
        if traces is not None
        else {}
    )
    if traces is not None and len(trace_by_index) != len(traces):
        raise SystemExit("duplicate trace absolute index")

    records = []
    prompt_hashes = set()
    stable_ids = set()
    for index, sample in enumerate(samples):
        doc = sample.get("doc", {})
        stable_id = str(
            doc.get("task_id")
            or doc.get("id")
            or doc.get("problem_id")
            or doc.get("unique_id")
            or f"index_{sample.get('doc_id', index)}"
        )
        if stable_id in stable_ids:
            raise SystemExit(f"duplicate stable id: {stable_id}")
        stable_ids.add(stable_id)
        prompt = prompt_from_sample(sample)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if prompt_hash in prompt_hashes:
            raise SystemExit(f"duplicate prompt hash: {stable_id}")
        prompt_hashes.add(prompt_hash)
        generation = sample["resps"][0][0]
        target = sample.get("target")
        sample_metric = args.primary_metric.split(",", 1)[0]
        correct_value = metric_value(sample, sample_metric)
        if correct_value not in {0.0, 1.0}:
            raise SystemExit(f"non-binary correctness for {stable_id}: {correct_value}")
        trace = trace_by_index.get(index)
        if trace is not None:
            if trace["prompt_hash"] != prompt_hash:
                raise SystemExit(f"prompt hash mismatch for {stable_id}")
            if str(trace["task_id"]) != stable_id:
                raise SystemExit(f"stable id mismatch for {stable_id}: {trace['task_id']}")
            if trace["decoded_generation"] != generation:
                raise SystemExit(f"generation mismatch for {stable_id}")
        records.append({
            "absolute_index": index,
            "stable_task_id": stable_id,
            "prompt_hash": prompt_hash,
            "raw_gold": doc.get("answer") or doc.get("code") or target,
            "target_hash": hashlib.sha256(str(target).encode("utf-8")).hexdigest(),
            "decoded_generation": generation,
            "correct": bool(correct_value),
            "primary_metric": args.primary_metric,
            "nfe": (
                int(trace["nfe"])
                if trace is not None
                else args.constant_nfe
            ),
            "evaluator_version": VERSION,
        })

    result_files = json.loads(Path(args.results).read_text(encoding="utf-8"))
    metric_key, reported = aggregate_value(result_files, args.task, args.primary_metric)
    recomputed = sum(record["correct"] for record in records) / len(records)
    if abs(recomputed - reported) > 1e-12:
        raise SystemExit(f"aggregate mismatch: records={recomputed}, reported={reported}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "task_audit_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "evaluator_version": VERSION,
        "task": args.task,
        "primary_metric": args.primary_metric,
        "reported_metric_key": metric_key,
        "correct": sum(record["correct"] for record in records),
        "total": len(records),
        "accuracy": recomputed,
        "duplicate_ids": 0,
        "duplicate_prompt_hashes": 0,
        "prompt_hash_mismatches": 0,
        "generation_mismatches": 0,
        "nfe_min": min(
            (record["nfe"] for record in records if record["nfe"] is not None),
            default=None,
        ),
        "nfe_max": max(
            (record["nfe"] for record in records if record["nfe"] is not None),
            default=None,
        ),
        "nfe_total": sum(
            record["nfe"] for record in records if record["nfe"] is not None
        ),
    }
    (output_dir / "task_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
