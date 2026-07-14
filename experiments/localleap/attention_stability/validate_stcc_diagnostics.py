import argparse
import json
from pathlib import Path

import torch


SCHEMA = "stcc_distribution_response_steps_v1"
VALIDATOR_VERSION = "stcc_diagnostics_validator_v1"


def _position_indices(masked_positions, selected_positions):
    mapping = {int(position): idx for idx, position in enumerate(masked_positions.tolist())}
    return [mapping[int(position)] for position in selected_positions.tolist()]


def validate_file(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA:
        raise AssertionError(f"{path}: unexpected schema {payload.get('schema_version')}")
    settings = payload["generation_settings"]
    steps = payload["steps"]
    summary = payload["decode_summary"]
    multiplier = int(settings["stcc_extra_multiplier"])
    jsd_threshold = float(settings["stcc_jsd_threshold"])
    extra_threshold = settings.get("stcc_extra_jsd_threshold")
    extra_threshold = jsd_threshold if extra_threshold is None else float(extra_threshold)
    topk = int(settings["stcc_topk"])
    min_overlap = settings.get("stcc_min_topk_overlap")
    min_overlap = max(1, topk - 1) if min_overlap is None else int(min_overlap)
    min_streak = int(settings["stcc_min_stability_streak"])

    previous_after = None
    previous_block = None
    total_extra = 0
    total_pruned = 0
    for expected_nfe, step in enumerate(steps, start=1):
        if int(step["global_nfe"]) != expected_nfe:
            raise AssertionError(f"{path}: non-contiguous NFE")
        before = int(step["mask_count_before"])
        after = int(step["mask_count_after"])
        budget = int(step["base_budget"])
        selected_count = int(step["selected_count"])
        extra_count = int(step["extra_selected_count"])
        block = int(step["block_index"])
        if previous_block is not None and block != previous_block:
            if previous_after != 0:
                raise AssertionError(f"{path}: previous block retained masks")
            previous_after = None
        if previous_after is not None and before > previous_after:
            raise AssertionError(f"{path}: mask count increased")
        if after != before - selected_count:
            raise AssertionError(f"{path}: mask accounting mismatch")
        if selected_count < budget or selected_count > min(before, multiplier * budget):
            raise AssertionError(f"{path}: selected count violates budget")
        if multiplier == 1 and selected_count != budget:
            raise AssertionError(f"{path}: quality arm changed baseline budget")
        if extra_count != selected_count - budget:
            raise AssertionError(f"{path}: extra count mismatch")

        masked = step["masked_positions_global"]
        selected = step["selected_positions_global"]
        base_selected = step["base_selected_positions_global"]
        extra_selected = step["extra_selected_positions_global"]
        if selected.numel() != selected_count or base_selected.numel() != budget:
            raise AssertionError(f"{path}: selected position tensor mismatch")
        if extra_selected.numel() != extra_count:
            raise AssertionError(f"{path}: extra position tensor mismatch")
        if len(set(int(value) for value in selected.tolist())) != selected_count:
            raise AssertionError(f"{path}: duplicate selected position")

        if extra_count:
            extra_indices = _position_indices(masked, extra_selected)
            all_selected_indices = _position_indices(masked, selected)
            has_history = step["has_history"]
            stable = step["top1_stable"]
            jsd = step["partition_jsd_nats"]
            overlap = step["topk_overlap_count"]
            streak = step["stability_streak"]
            conflicts = step["pair_conflict"]
            for idx in extra_indices:
                if not bool(has_history[idx]) or not bool(stable[idx]):
                    raise AssertionError(f"{path}: unverified extra commit")
                if float(jsd[idx]) > extra_threshold + 1e-7:
                    raise AssertionError(f"{path}: high-JSD extra commit")
                if int(overlap[idx]) < min_overlap or int(streak[idx]) < min_streak:
                    raise AssertionError(f"{path}: weak-overlap/streak extra commit")
                peers = [peer for peer in all_selected_indices if peer != idx]
                if peers and bool(conflicts[idx, peers].any()):
                    raise AssertionError(f"{path}: conflicting extra commit")

        raw_edges = step["raw_directed_edges"]
        active_edges = step["active_directed_edges"]
        if bool((active_edges & ~raw_edges).any()):
            raise AssertionError(f"{path}: active edge is not a raw edge")
        total_pruned += int(raw_edges.sum().item() - active_edges.sum().item())
        total_extra += extra_count
        previous_after = after
        previous_block = block

    if not steps or int(steps[-1]["mask_count_after"]) != 0:
        raise AssertionError(f"{path}: residual block masks")
    if int(summary["actual_nfe"]) != len(steps):
        raise AssertionError(f"{path}: summary NFE mismatch")
    if int(summary["extra_commits"]) != total_extra:
        raise AssertionError(f"{path}: summary extra commits mismatch")
    if int(summary["horizontal_pruned_low_response_edges"]) != total_pruned:
        raise AssertionError(f"{path}: summary pruned edges mismatch")
    if int(summary["residual_mask_count"]) != 0:
        raise AssertionError(f"{path}: summary residual masks")
    return {
        "task_id": payload.get("task_id"),
        "absolute_index": int(payload["absolute_index"]),
        "prompt_hash": payload["prompt_hash"],
        "nfe": len(steps),
        "extra_commits": total_extra,
        "pruned_edges": total_pruned,
        "bytes": path.stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostics_dir")
    parser.add_argument("--expected-files", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = sorted(Path(args.diagnostics_dir).glob("*.pt"))
    if args.expected_files is not None and len(paths) != args.expected_files:
        raise SystemExit(f"expected {args.expected_files} files, found {len(paths)}")
    records = [validate_file(path) for path in paths]
    task_ids = [record["task_id"] for record in records]
    indices = [record["absolute_index"] for record in records]
    hashes = [record["prompt_hash"] for record in records]
    if len(task_ids) != len(set(task_ids)):
        raise SystemExit("duplicate task ids")
    if len(indices) != len(set(indices)) or len(hashes) != len(set(hashes)):
        raise SystemExit("duplicate absolute index or prompt hash")
    summary = {
        "validator_version": VALIDATOR_VERSION,
        "schema_version": SCHEMA,
        "files": len(records),
        "unique_task_ids": len(set(task_ids)),
        "nfe_min": min((record["nfe"] for record in records), default=0),
        "nfe_max": max((record["nfe"] for record in records), default=0),
        "total_extra_commits": sum(record["extra_commits"] for record in records),
        "total_pruned_edges": sum(record["pruned_edges"] for record in records),
        "total_bytes": sum(record["bytes"] for record in records),
        "residual_mask_count": 0,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
