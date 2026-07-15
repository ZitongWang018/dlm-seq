import argparse
import hashlib
import json
from pathlib import Path

from sanitize import sanitize


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("samples")
    parser.add_argument("trace", nargs="?")
    parser.add_argument("--constant-nfe", type=int)
    parser.add_argument("--postprocess", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    samples = read_jsonl(args.samples)
    cleaned = read_jsonl(args.samples + ".cleaned")
    traces = read_jsonl(args.trace) if args.trace else None
    if len(samples) != len(cleaned) or (traces is not None and len(samples) != len(traces)):
        raise SystemExit(
            f"record count mismatch: samples={len(samples)} cleaned={len(cleaned)} "
            f"traces={None if traces is None else len(traces)}"
        )
    if traces is None and args.constant_nfe is None:
        raise SystemExit("baseline audit requires --constant-nfe when trace is omitted")

    sample_by_id = {item["doc"]["task_id"]: item for item in samples}
    clean_by_id = {item["task_id"]: item for item in cleaned}
    trace_by_id = {item["task_id"]: item for item in traces} if traces is not None else None
    mappings = (sample_by_id, clean_by_id) + ((trace_by_id,) if trace_by_id is not None else ())
    if any(len(mapping) != len(samples) for mapping in mappings):
        raise SystemExit("duplicate or missing stable task ids")
    if set(sample_by_id) != set(clean_by_id) or (
        trace_by_id is not None and set(sample_by_id) != set(trace_by_id)
    ):
        raise SystemExit("task-id sets do not align")

    records = []
    for task_id in sorted(sample_by_id, key=lambda value: int(value.split("/")[-1])):
        sample = sample_by_id[task_id]
        clean = clean_by_id[task_id]
        trace = trace_by_id[task_id] if trace_by_id is not None else {
            "absolute_index": int(task_id.split("/")[-1]),
            "task_id": task_id,
            "prompt_hash": sha256_text(sample["doc"]["prompt"]),
            "decoded_generation": sample["resps"][0][0],
            "nfe": args.constant_nfe,
            "generation_settings": {"steps": args.constant_nfe, "decoder": "baseline"},
        }
        prompt = sample["doc"]["prompt"]
        generation = sample["resps"][0][0]
        if trace["prompt_hash"] != sha256_text(prompt):
            raise SystemExit(f"prompt hash mismatch for {task_id}")
        if trace["decoded_generation"] != generation:
            raise SystemExit(f"decoded generation mismatch for {task_id}")
        normalized_gold = sanitize(prompt + "\n" + sample["doc"]["canonical_solution"], sample["doc"]["entry_point"])
        record = dict(trace)
        record.update({
            "raw_gold": sample["doc"]["canonical_solution"],
            "normalized_gold": normalized_gold,
            "extracted_prediction": clean["completion"],
            "correct": bool(clean["pass_at_1"]),
            "target_hash": sample.get("target_hash") or sha256_text(sample["target"]),
            "evaluator_version": "localleap_postprocess_audit_v1",
        })
        records.append(record)

    nfe_values = [int(record["nfe"]) for record in records]
    if any(value <= 0 for value in nfe_values):
        raise SystemExit("non-positive NFE")
    correct = sum(record["correct"] for record in records)
    accuracy = correct / len(records)
    reported = float(Path(args.postprocess).read_text(encoding="utf-8").strip().splitlines()[-1])
    if abs(accuracy - reported) > 1e-12:
        raise SystemExit(f"aggregate mismatch: records={accuracy} postprocess={reported}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "audit_records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "evaluator_version": "localleap_postprocess_audit_v1",
        "correct": correct,
        "total": len(records),
        "accuracy": accuracy,
        "duplicate_ids": len(records) - len(set(record["task_id"] for record in records)),
        "missing_ids": 0,
        "prompt_hash_mismatches": 0,
        "generation_mismatches": 0,
        "nfe_min": min(nfe_values),
        "nfe_max": max(nfe_values),
        "residual_mask_count": 0,
        "postprocess_accuracy": reported,
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
