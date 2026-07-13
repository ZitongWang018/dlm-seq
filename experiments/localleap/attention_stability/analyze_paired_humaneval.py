import argparse
import json
import math
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def keyed(items, key):
    result = {key(item): item for item in items}
    if len(result) != len(items):
        raise SystemExit("duplicate stable ids")
    return result


def exact_mcnemar_p(method_only, baseline_only):
    discordant = method_only + baseline_only
    if discordant == 0:
        return 1.0
    tail = min(method_only, baseline_only)
    return min(1.0, 2.0 * sum(math.comb(discordant, k) for k in range(tail + 1)) / (2 ** discordant))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_samples")
    parser.add_argument("method_samples")
    parser.add_argument("trace")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    baseline_samples = keyed(read_jsonl(args.baseline_samples), lambda item: item["doc"]["task_id"])
    method_samples = keyed(read_jsonl(args.method_samples), lambda item: item["doc"]["task_id"])
    baseline_clean = keyed(read_jsonl(args.baseline_samples + ".cleaned"), lambda item: item["task_id"])
    method_clean = keyed(read_jsonl(args.method_samples + ".cleaned"), lambda item: item["task_id"])
    traces = keyed(read_jsonl(args.trace), lambda item: item["task_id"])
    id_sets = [set(mapping) for mapping in (baseline_samples, method_samples, baseline_clean, method_clean, traces)]
    if any(ids != id_sets[0] for ids in id_sets[1:]):
        raise SystemExit("paired stable-id sets do not match")

    paired = []
    for task_id in sorted(id_sets[0], key=lambda value: int(value.split("/")[-1])):
        baseline = baseline_samples[task_id]
        method = method_samples[task_id]
        if baseline["prompt_hash"] != method["prompt_hash"]:
            raise SystemExit(f"prompt hash mismatch for {task_id}")
        if baseline["target_hash"] != method["target_hash"]:
            raise SystemExit(f"target hash mismatch for {task_id}")
        baseline_correct = bool(baseline_clean[task_id]["pass_at_1"])
        method_correct = bool(method_clean[task_id]["pass_at_1"])
        paired.append({
            "task_id": task_id,
            "prompt_hash": baseline["prompt_hash"],
            "target_hash": baseline["target_hash"],
            "baseline_correct": baseline_correct,
            "method_correct": method_correct,
            "generation_changed": baseline["resps"][0][0] != method["resps"][0][0],
            "baseline_extracted": baseline_clean[task_id]["completion"],
            "method_extracted": method_clean[task_id]["completion"],
            "decode_diagnostics": traces[task_id]["decode_diagnostics"],
        })

    both_correct = sum(item["baseline_correct"] and item["method_correct"] for item in paired)
    method_only = sum((not item["baseline_correct"]) and item["method_correct"] for item in paired)
    baseline_only = sum(item["baseline_correct"] and (not item["method_correct"]) for item in paired)
    both_wrong = len(paired) - both_correct - method_only - baseline_only
    changed = [item for item in paired if item["generation_changed"]]
    summary = {
        "total": len(paired),
        "baseline_correct": both_correct + baseline_only,
        "method_correct": both_correct + method_only,
        "both_correct": both_correct,
        "method_only": method_only,
        "baseline_only": baseline_only,
        "both_wrong": both_wrong,
        "generation_changed": len(changed),
        "generation_unchanged": len(paired) - len(changed),
        "paired_prompt_hash_mismatches": 0,
        "paired_target_hash_mismatches": 0,
        "exact_mcnemar_p": exact_mcnemar_p(method_only, baseline_only),
        "method_only_task_ids": [item["task_id"] for item in paired if item["method_correct"] and not item["baseline_correct"]],
        "baseline_only_task_ids": [item["task_id"] for item in paired if item["baseline_correct"] and not item["method_correct"]],
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paired_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "paired_records.jsonl").open("w", encoding="utf-8") as handle:
        for item in paired:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
