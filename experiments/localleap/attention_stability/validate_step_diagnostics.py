import argparse
import json
from pathlib import Path

import torch


def has_contiguous_nfe(steps):
    return [step["global_nfe"] for step in steps] == list(range(1, len(steps) + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostics_dir")
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = sorted(Path(args.diagnostics_dir).glob("*.pt"))
    if len(paths) != args.expected_count:
        raise SystemExit(f"diagnostics count mismatch: {len(paths)} != {args.expected_count}")

    task_ids = []
    step_counts = []
    total_steps = 0
    total_bytes = 0
    unstable_total = 0
    changed_total = 0
    strongly_dependent_total = 0
    rejected_total = 0
    underfilled_total = 0
    fallback_total = 0
    stable_pruned_total = 0
    forced_fill_total = 0
    dependency_min = float("inf")
    dependency_max = float("-inf")
    dependency_mean_sum = 0.0
    asymmetry_mean_sum = 0.0

    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        schema_version = payload["schema_version"]
        if schema_version not in {"attention_stability_steps_v1", "attention_stability_steps_v2"}:
            raise SystemExit(f"unexpected schema in {path}")
        task_ids.append(payload["task_id"])
        steps = payload["steps"]
        expected_steps = int(payload["generation_settings"]["steps"])
        fill_budget = bool(payload["generation_settings"].get("dependency_fill_budget", False))
        if fill_budget and len(steps) != expected_steps:
            raise SystemExit(f"fixed-budget step count mismatch in {path}: {len(steps)} != {expected_steps}")
        if not fill_budget and len(steps) < expected_steps:
            raise SystemExit(f"step count below configured budget in {path}: {len(steps)} < {expected_steps}")
        if not has_contiguous_nfe(steps):
            raise SystemExit(f"non-contiguous NFE in {path}")
        if steps[-1]["mask_count_after"] != 0:
            raise SystemExit(f"residual masks in {path}")

        for step in steps:
            directional = step["directional_attention"]
            symmetric = step["symmetric_dependency"]
            if directional.shape != (32, 32) or symmetric.shape != (32, 32):
                raise SystemExit(f"attention shape mismatch in {path}")
            if not torch.isfinite(directional).all() or not torch.isfinite(symmetric).all():
                raise SystemExit(f"non-finite attention in {path}")
            if not torch.allclose(symmetric, symmetric.transpose(0, 1), atol=1e-3, rtol=0):
                raise SystemExit(f"non-symmetric dependency in {path}")
            selection_dependency = step.get("selection_dependency", symmetric)
            if selection_dependency.shape != (32, 32) or not torch.isfinite(selection_dependency).all():
                raise SystemExit(f"invalid selection dependency in {path}")
            candidate_count = step["mask_count_before"]
            candidate_fields = (
                "masked_positions_global",
                "masked_positions_local",
                "top1_token_ids",
                "top1_confidences",
                "previous_top1_token_ids",
                "candidate_changed",
                "max_dependency_to_previous",
                "maturity",
                "ordered_positions_global",
            )
            if any(len(step[field]) != candidate_count for field in candidate_fields):
                raise SystemExit(f"candidate-state length mismatch in {path}")
            if "temporal_tier" in step:
                temporal_fields = (
                    "current_topk_token_ids",
                    "previous_topk_token_ids",
                    "topk_overlap_count",
                    "temporal_tier",
                )
                if any(len(step[field]) != candidate_count for field in temporal_fields):
                    raise SystemExit(f"temporal candidate-state length mismatch in {path}")
                tiers = step["temporal_tier"]
                if not bool(((tiers >= 0) & (tiers <= 2)).all()):
                    raise SystemExit(f"invalid temporal tier in {path}")
            selected_count = len(step["selected_positions_global"])
            target_count = min(step["budget"], candidate_count)
            if not step["underfilled"] and selected_count != min(step["budget"], candidate_count):
                raise SystemExit(f"selected budget mismatch in {path}")
            if fill_budget and selected_count != target_count:
                raise SystemExit(f"fixed-budget selector underfilled in {path}")
            unstable_total += int(step["unstable_candidates"])
            changed_total += int(step["changed_candidates"])
            strongly_dependent_total += int(step["strongly_dependent_candidates"])
            rejected_total += int(step["rejected_pairs"])
            underfilled_total += int(step["underfilled"])
            fallback_total += int(step["all_immature_fallback"])
            stable_pruned_total += int(step.get("stable_conflicts_pruned", 0))
            forced_fill_total += int(step.get("forced_budget_fills", 0))
            dependency_min = min(dependency_min, float(step["dependency_min"]))
            dependency_max = max(dependency_max, float(step["dependency_max"]))
            dependency_mean_sum += float(step["dependency_mean"])
            asymmetry_mean_sum += float(step["attention_asymmetry_mean"])

        total_steps += len(steps)
        step_counts.append(len(steps))
        total_bytes += path.stat().st_size

    if len(task_ids) != len(set(task_ids)):
        raise SystemExit("duplicate task ids in step diagnostics")
    summary = {
        "schema_version": "attention_stability_steps_v1_or_v2",
        "files": len(paths),
        "unique_task_ids": len(set(task_ids)),
        "total_steps": total_steps,
        "total_bytes": total_bytes,
        "steps_per_file_min": min(step_counts),
        "steps_per_file_max": max(step_counts),
        "unstable_candidates_total": unstable_total,
        "changed_candidates_total": changed_total,
        "strongly_dependent_candidates_total": strongly_dependent_total,
        "rejected_pairs_total": rejected_total,
        "underfilled_steps_total": underfilled_total,
        "all_immature_fallback_steps_total": fallback_total,
        "stable_conflicts_pruned_total": stable_pruned_total,
        "forced_budget_fills_total": forced_fill_total,
        "dependency_min": dependency_min,
        "dependency_max": dependency_max,
        "dependency_mean": dependency_mean_sum / total_steps,
        "attention_asymmetry_mean": asymmetry_mean_sum / total_steps,
        "residual_mask_count": 0,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
