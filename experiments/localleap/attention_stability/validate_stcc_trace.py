import argparse
import json
import math
from pathlib import Path


VERSION = "stcc_trace_validator_v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--quality-nfe", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.trace).open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if len(records) != args.expected_records:
        raise SystemExit(f"expected {args.expected_records} records, found {len(records)}")
    task_ids = [str(record["task_id"]) for record in records]
    prompt_hashes = [record["prompt_hash"] for record in records]
    if len(set(task_ids)) != len(records):
        raise SystemExit("duplicate task ids")
    if len(set(prompt_hashes)) != len(records):
        raise SystemExit("duplicate prompt hashes")

    nfe_values = []
    residual = 0
    extras = 0
    pruned = 0
    forced = 0
    wall = 0.0
    for record in records:
        nfe = int(record["nfe"])
        summary = record["decode_diagnostics"]
        if nfe <= 0 or nfe != int(summary["actual_nfe"]):
            raise SystemExit(f"invalid NFE for {record['task_id']}")
        if args.quality_nfe is not None and nfe != args.quality_nfe:
            raise SystemExit(
                f"quality NFE mismatch for {record['task_id']}: {nfe} != {args.quality_nfe}"
            )
        if any(
            not math.isfinite(float(summary[key]))
            for key in ("partition_jsd_sum", "partition_jsd_max", "step_wall_time_seconds")
        ):
            raise SystemExit(f"non-finite diagnostic for {record['task_id']}")
        nfe_values.append(nfe)
        residual += int(summary["residual_mask_count"])
        extras += int(summary["extra_commits"])
        pruned += int(summary["horizontal_pruned_low_response_edges"])
        forced += int(summary["forced_conflict_fills"])
        wall += float(summary["step_wall_time_seconds"])
    if residual:
        raise SystemExit(f"residual masks: {residual}")

    summary = {
        "validator_version": VERSION,
        "records": len(records),
        "unique_task_ids": len(set(task_ids)),
        "unique_prompt_hashes": len(set(prompt_hashes)),
        "nfe_min": min(nfe_values),
        "nfe_max": max(nfe_values),
        "total_nfe": sum(nfe_values),
        "total_extra_commits": extras,
        "total_pruned_edges": pruned,
        "total_forced_conflict_fills": forced,
        "summed_step_wall_time_seconds": wall,
        "residual_mask_count": residual,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
