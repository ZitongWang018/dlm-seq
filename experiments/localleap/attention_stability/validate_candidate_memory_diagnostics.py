import argparse
import json
import math
from pathlib import Path

import torch


def _as_float(value):
    return value.to(torch.float32)


def _jsd(first, second):
    midpoint = 0.5 * (first + second)
    first_term = torch.where(
        first > 0,
        first * (torch.log(first) - torch.log(midpoint)),
        torch.zeros_like(first),
    )
    second_term = torch.where(
        second > 0,
        second * (torch.log(second) - torch.log(midpoint)),
        torch.zeros_like(second),
    )
    return 0.5 * (first_term.sum(-1) + second_term.sum(-1))


def _is_valid_topk_membership(scores, membership, k):
    """Validate a Top-K set while allowing arbitrary choices at boundary ties.

    PyTorch's CPU and CUDA ``topk`` kernels may select different indices when
    several values are exactly equal at the K-th boundary.  The recorded set is
    valid if it has the requested size and no excluded candidate has a score
    strictly greater than an included candidate.
    """
    scores = scores.to(torch.float64)
    membership = membership.bool()
    if scores.ndim != 1 or membership.shape != scores.shape:
        return False
    if k < 0 or k > scores.numel() or int(membership.sum().item()) != k:
        return False
    if not torch.isfinite(scores).all():
        return False
    if k == 0 or k == scores.numel():
        return True
    return bool(scores[membership].min() >= scores[~membership].max())


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
    total_steps = 0
    total_bytes = 0
    total_history = 0
    total_stable = 0
    total_changed = 0
    total_eligible = 0
    total_forced = 0
    total_fallback_steps = 0
    full_jsd_sum = 0.0
    full_jsd_max = 0.0
    full_jsd_observations = 0
    sparse_jsd_sum = 0.0
    sparse_jsd_max = 0.0
    approximation_gap_sum = 0.0
    approximation_gap_max = 0.0
    arrival_sum = 0.0
    arrival_max = 0.0
    influence_sum = 0.0
    influence_max = 0.0
    entropy_sum = 0.0
    entropy_count = 0
    topk_overlap_sum = 0
    topk_overlap_count = 0
    peak_runtime_positions = 0
    peak_runtime_full_elements = 0
    peak_cuda_allocated = 0
    peak_cuda_reserved = 0
    wall_time_sum = 0.0
    nfe_values = []
    tie_boundary_frontiers_accepted = 0

    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["schema_version"] != "candidate_memory_steps_v2":
            raise SystemExit(f"unexpected schema in {path}")
        task_ids.append(payload["task_id"])
        settings = payload["generation_settings"]
        topk = int(settings["candidate_memory_topk"])
        delta = float(settings["candidate_memory_confidence_threshold"])
        exact_jsd = bool(settings.get("candidate_memory_exact_jsd", False))
        mask_id = int(settings.get("mask_id", 126336))
        steps = payload["steps"]
        expected_steps = int(settings["steps"])
        if len(steps) != expected_steps:
            raise SystemExit(f"step count mismatch in {path}: {len(steps)} != {expected_steps}")
        if [step["global_nfe"] for step in steps] != list(range(1, expected_steps + 1)):
            raise SystemExit(f"non-contiguous NFE in {path}")
        if steps[-1]["mask_count_after"] != 0:
            raise SystemExit(f"residual masks in {path}")

        previous_step_by_block = {}
        for step in steps:
            directional = step["directional_attention"]
            symmetric = step["symmetric_dependency"]
            if directional.shape != (32, 32) or symmetric.shape != (32, 32):
                raise SystemExit(f"attention shape mismatch in {path}")
            if not torch.isfinite(directional).all() or not torch.isfinite(symmetric).all():
                raise SystemExit(f"non-finite attention in {path}")
            if not torch.allclose(symmetric, symmetric.transpose(0, 1), atol=1e-3, rtol=0):
                raise SystemExit(f"non-symmetric dependency in {path}")

            count = int(step["mask_count_before"])
            selected = step["selected_positions_global"]
            budget = int(step["budget"])
            if int((step["input_block_token_ids"] == mask_id).sum().item()) != count:
                raise SystemExit(f"recorded mask count disagrees with block input in {path}")
            if len(selected) != min(budget, count):
                raise SystemExit(f"fixed-budget violation in {path}")
            if step["mask_count_after"] != count - len(selected):
                raise SystemExit(f"mask accounting mismatch in {path}")
            if int(step["runtime_memory_positions"]) != step["mask_count_after"]:
                raise SystemExit(f"candidate memory deletion mismatch in {path}")
            if int(step["runtime_memory_topk_elements"]) != step["mask_count_after"] * topk:
                raise SystemExit(f"top-k memory size mismatch in {path}")

            one_dimensional_fields = (
                "masked_positions_global",
                "masked_positions_local",
                "current_other_mass",
                "previous_other_mass",
                "current_other_mass_on_previous_partition",
                "current_top1_token_ids",
                "previous_top1_token_ids",
                "top1_confidences",
                "previous_top1_confidences",
                "confidence_change",
                "top1_margin",
                "entropy_nats",
                "has_history",
                "top1_stable",
                "candidate_changed",
                "above_confidence_threshold",
                "eligible",
                "effective_eligible",
                "in_confidence_frontier",
                "baseline_confidence_rank",
                "full_jsd_nats",
                "full_jsd_available",
                "sparse_previous_partition_jsd_nats",
                "decision_jsd_nats",
                "jsd_approximation_gap",
                "previous_top1_current_probability",
                "topk_overlap_count",
                "topk_jaccard",
                "attention_arrival",
                "influence_candidate_jsd",
                "selected_by_fallback",
                "selection_reason",
            )
            if any(len(step[field]) != count for field in one_dimensional_fields):
                raise SystemExit(f"candidate field length mismatch in {path}")
            matrix_fields = (
                "current_topk_token_ids",
                "current_topk_probabilities",
                "previous_topk_token_ids",
                "previous_topk_probabilities",
                "current_probabilities_on_previous_topk",
            )
            if any(tuple(step[field].shape) != (count, topk) for field in matrix_fields):
                raise SystemExit(f"top-k field shape mismatch in {path}")
            history = step["has_history"].bool()
            if not torch.equal(
                step["current_topk_token_ids"][:, 0], step["current_top1_token_ids"]
            ):
                raise SystemExit(f"actual argmax is not first in current Top-K in {path}")
            if history.any() and not torch.equal(
                step["previous_topk_token_ids"][history, 0],
                step["previous_top1_token_ids"][history],
            ):
                raise SystemExit(f"actual historical argmax is not first in Top-K in {path}")

            current_sum = _as_float(step["current_topk_probabilities"]).sum(1) + _as_float(
                step["current_other_mass"]
            )
            if not torch.allclose(current_sum, torch.ones_like(current_sum), atol=2e-5, rtol=0):
                raise SystemExit(f"current probability mass mismatch in {path}")
            if history.any():
                previous_sum = _as_float(step["previous_topk_probabilities"])[history].sum(1) + _as_float(
                    step["previous_other_mass"]
                )[history]
                partition_sum = _as_float(step["current_probabilities_on_previous_topk"])[history].sum(1) + _as_float(
                    step["current_other_mass_on_previous_partition"]
                )[history]
                if not torch.allclose(previous_sum, torch.ones_like(previous_sum), atol=2e-5, rtol=0):
                    raise SystemExit(f"previous probability mass mismatch in {path}")
                if not torch.allclose(partition_sum, torch.ones_like(partition_sum), atol=2e-5, rtol=0):
                    raise SystemExit(f"current partition mass mismatch in {path}")
                previous_partition = torch.cat(
                    (
                        _as_float(step["previous_topk_probabilities"])[history],
                        _as_float(step["previous_other_mass"])[history].unsqueeze(1),
                    ),
                    dim=1,
                )
                current_partition = torch.cat(
                    (
                        _as_float(step["current_probabilities_on_previous_topk"])[history],
                        _as_float(step["current_other_mass_on_previous_partition"])[
                            history
                        ].unsqueeze(1),
                    ),
                    dim=1,
                )
                recomputed_sparse_jsd = _jsd(previous_partition, current_partition)
                recorded_sparse_jsd = _as_float(
                    step["sparse_previous_partition_jsd_nats"]
                )[history]
                if not torch.allclose(
                    recomputed_sparse_jsd, recorded_sparse_jsd, atol=2e-6, rtol=0
                ):
                    raise SystemExit(f"sparse JSD recomputation mismatch in {path}")

            stable = step["top1_stable"].bool()
            changed = step["candidate_changed"].bool()
            if not torch.equal(stable, history & ~changed):
                raise SystemExit(f"stable/change mismatch in {path}")
            if history.any() and not torch.equal(
                stable[history],
                step["current_top1_token_ids"][history] == step["previous_top1_token_ids"][history],
            ):
                raise SystemExit(f"top-1 stability mismatch in {path}")
            above = step["above_confidence_threshold"].bool()
            expected_above = step["top1_confidences"].to(torch.float64) > delta
            if not torch.equal(above, expected_above):
                raise SystemExit(f"confidence gate mismatch in {path}")
            expected_eligible = ~history | (stable & above)
            if not torch.equal(step["eligible"].bool(), expected_eligible):
                raise SystemExit(f"eligibility mismatch in {path}")
            full_confidence = torch.full(
                (int(payload["final_sequence_token_ids"].numel()),),
                -torch.inf,
                dtype=torch.float64,
            )
            full_confidence[step["masked_positions_global"].long()] = step[
                "top1_confidences"
            ].to(torch.float64)
            frontier_size = min(count, budget + 1)
            frontier_scores = step["top1_confidences"].to(torch.float64)
            recorded_frontier = step["in_confidence_frontier"].bool()
            if not _is_valid_topk_membership(
                frontier_scores, recorded_frontier, frontier_size
            ):
                raise SystemExit(f"confidence frontier mismatch in {path}")
            strict_cpu_frontier = torch.zeros(count, dtype=torch.bool)
            strict_cpu_frontier[
                torch.topk(frontier_scores, k=frontier_size).indices
            ] = True
            if not torch.equal(recorded_frontier, strict_cpu_frontier):
                tie_boundary_frontiers_accepted += 1
            expected_effective = expected_eligible
            if settings["candidate_memory_fallback"] == "frontier" and history.any():
                expected_effective = expected_eligible & step["in_confidence_frontier"].bool()
            if not torch.equal(step["effective_eligible"].bool(), expected_effective):
                raise SystemExit(f"effective eligibility mismatch in {path}")

            full_jsd = _as_float(step["full_jsd_nats"])
            full_available = step["full_jsd_available"].bool()
            sparse_jsd = _as_float(step["sparse_previous_partition_jsd_nats"])
            decision_jsd = _as_float(step["decision_jsd_nats"])
            gap = _as_float(step["jsd_approximation_gap"])
            if not torch.isfinite(full_jsd).all() or not torch.isfinite(sparse_jsd).all():
                raise SystemExit(f"non-finite JSD in {path}")
            if exact_jsd and not torch.equal(full_available, history):
                raise SystemExit(f"missing requested exact JSD in {path}")
            if not exact_jsd and full_available.any():
                raise SystemExit(f"unexpected full-vocabulary history in {path}")
            if (full_jsd[full_available] < -1e-6).any() or (
                full_jsd[full_available] > math.log(2) + 2e-5
            ).any():
                raise SystemExit(f"full JSD outside bounds in {path}")
            if (sparse_jsd < -1e-6).any() or (sparse_jsd > math.log(2) + 2e-5).any():
                raise SystemExit(f"sparse JSD outside bounds in {path}")
            if full_available.any() and (
                sparse_jsd[full_available] > full_jsd[full_available] + 2e-5
            ).any():
                raise SystemExit(f"coarsened JSD exceeds full JSD in {path}")
            expected_gap = torch.where(
                full_available, full_jsd - sparse_jsd, torch.zeros_like(full_jsd)
            )
            if not torch.allclose(gap, expected_gap, atol=2e-6, rtol=0):
                raise SystemExit(f"JSD gap mismatch in {path}")
            if not torch.allclose(decision_jsd, sparse_jsd, atol=2e-6, rtol=0):
                raise SystemExit(f"decision JSD is not O(|M|K) sparse JSD in {path}")

            previous_selected = step["previous_selected_positions_global"]
            attention_to_previous = _as_float(step["attention_to_previous_selected"])
            reverse_attention = _as_float(step["reverse_attention_from_previous_selected"])
            if tuple(attention_to_previous.shape) != (count, len(previous_selected)):
                raise SystemExit(f"arrival matrix shape mismatch in {path}")
            if tuple(reverse_attention.shape) != (count, len(previous_selected)):
                raise SystemExit(f"reverse-arrival matrix shape mismatch in {path}")
            if len(previous_selected):
                local_rows = step["masked_positions_local"].long()
                local_columns = previous_selected.long() - int(step["block_start"])
                independently_computed = _as_float(directional).index_select(0, local_rows).index_select(
                    1, local_columns
                )
                if not torch.allclose(attention_to_previous, independently_computed, atol=1e-3, rtol=0):
                    raise SystemExit(f"directional arrival mismatch in {path}")
                independently_computed_reverse = _as_float(directional).index_select(
                    0, local_columns
                ).index_select(1, local_rows).transpose(0, 1)
                if not torch.allclose(reverse_attention, independently_computed_reverse, atol=1e-3, rtol=0):
                    raise SystemExit(f"reverse attention mismatch in {path}")
            if not torch.allclose(
                _as_float(step["directional_attention_asymmetry_to_previous"]),
                (attention_to_previous - reverse_attention).abs(),
                atol=2e-6,
                rtol=0,
            ):
                raise SystemExit(f"directional asymmetry mismatch in {path}")
            arrival = _as_float(step["attention_arrival"])
            if not torch.allclose(arrival, attention_to_previous.sum(1), atol=1e-6, rtol=0):
                raise SystemExit(f"arrival sum mismatch in {path}")
            if not torch.allclose(
                _as_float(step["influence_candidate_jsd"]),
                arrival * sparse_jsd,
                atol=2e-6,
                rtol=0,
            ):
                raise SystemExit(f"influence mismatch in {path}")

            eligible_confidence = torch.full_like(full_confidence, -torch.inf)
            effective_positions = step["masked_positions_global"][expected_effective]
            eligible_confidence[effective_positions.long()] = full_confidence[
                effective_positions.long()
            ]
            eligible_take = min(budget, int(expected_effective.sum().item()))
            expected_selected = (
                torch.topk(eligible_confidence, k=eligible_take).indices.tolist()
                if eligible_take
                else []
            )
            remaining_budget = budget - len(expected_selected)
            if remaining_budget:
                remaining_candidates = [
                    idx
                    for idx, position in enumerate(step["masked_positions_global"].tolist())
                    if int(position) not in expected_selected
                ]
                mode = settings["candidate_memory_fallback"]
                if mode in {"confidence", "frontier"}:
                    fallback_confidence = full_confidence.clone()
                    if expected_selected:
                        fallback_confidence[torch.tensor(expected_selected, dtype=torch.long)] = -torch.inf
                    expected_selected.extend(
                        torch.topk(fallback_confidence, k=remaining_budget).indices.tolist()
                    )
                else:
                    fallback_influence = _as_float(step["influence_candidate_jsd"])
                    fallback_stable = step["top1_stable"].bool()
                    fallback_confidences = step["top1_confidences"].to(torch.float64)
                    remaining_candidates.sort(
                        key=lambda idx: (
                            0 if bool(fallback_stable[idx]) else 1,
                            float(fallback_influence[idx]),
                            -float(fallback_confidences[idx]),
                        )
                    )
                    expected_selected.extend(
                        int(step["masked_positions_global"][idx])
                        for idx in remaining_candidates[:remaining_budget]
                    )
            if set(expected_selected) != set(selected.tolist()):
                raise SystemExit(
                    f"selection rule mismatch in {path}, nfe={step['global_nfe']}, "
                    f"selected={selected.tolist()}, expected={expected_selected}"
                )
            expected_fallback = bool(history.any()) and remaining_budget > 0
            if bool(step["fallback_used"]) != expected_fallback:
                raise SystemExit(f"fallback flag mismatch in {path}")
            expected_forced = remaining_budget if bool(history.any()) else 0
            if int(step["forced_commits"]) != expected_forced:
                raise SystemExit(f"forced-commit count mismatch in {path}")

            candidate_index_by_position = {
                int(position): idx
                for idx, position in enumerate(step["masked_positions_global"].tolist())
            }
            selected_candidate_indices = torch.tensor(
                [candidate_index_by_position[int(position)] for position in selected.tolist()],
                dtype=torch.long,
            )
            expected_selected_tokens = step["current_top1_token_ids"].index_select(
                0, selected_candidate_indices
            )
            if not torch.equal(step["selected_token_ids"], expected_selected_tokens):
                raise SystemExit(f"selected token is not current argmax in {path}")
            final_selected_tokens = payload["final_sequence_token_ids"].index_select(
                0, selected.long()
            ).to(torch.int32)
            if not torch.equal(final_selected_tokens, step["selected_token_ids"]):
                raise SystemExit(f"submitted token changed later in {path}")

            block_index = int(step["block_index"])
            previous = previous_step_by_block.get(block_index)
            if previous is None:
                if history.any() or len(previous_selected):
                    raise SystemExit(f"block memory was not reset in {path}")
                # Bootstrap must exactly match confidence TopB.
                expected = torch.topk(full_confidence, k=budget).indices
                if set(expected.tolist()) != set(selected.tolist()):
                    raise SystemExit(
                        f"bootstrap is not baseline TopB in {path}, block={block_index}, "
                        f"selected={selected.tolist()}, expected={expected.tolist()}"
                    )
            else:
                if not history.all():
                    raise SystemExit(f"missing cross-step memory in {path}")
                if not torch.equal(
                    previous_selected, previous["selected_positions_global"]
                ):
                    raise SystemExit(f"previous selected set mismatch in {path}")
                if not torch.equal(
                    step["previous_selected_token_ids"], previous["selected_token_ids"]
                ):
                    raise SystemExit(f"previous selected token mismatch in {path}")
                expected_input = previous["input_block_token_ids"].clone()
                expected_input[previous["selected_positions_local"].long()] = previous[
                    "selected_token_ids"
                ]
                if not torch.equal(step["input_block_token_ids"], expected_input):
                    raise SystemExit(f"submitted tokens did not form the next input in {path}")
                previous_positions = previous["masked_positions_global"]
                previous_selected_set = set(previous["selected_positions_global"].tolist())
                expected_positions = [
                    int(position) for position in previous_positions.tolist() if position not in previous_selected_set
                ]
                if expected_positions != step["masked_positions_global"].tolist():
                    raise SystemExit(f"cross-step position alignment mismatch in {path}")
                index_by_position = {
                    int(position): idx for idx, position in enumerate(previous_positions.tolist())
                }
                aligned = torch.tensor(
                    [index_by_position[int(position)] for position in step["masked_positions_global"].tolist()],
                    dtype=torch.long,
                )
                if not torch.equal(
                    step["previous_topk_token_ids"], previous["current_topk_token_ids"].index_select(0, aligned)
                ):
                    raise SystemExit(f"top-k token memory mismatch in {path}")
                if not torch.equal(
                    step["previous_top1_token_ids"],
                    previous["current_top1_token_ids"].index_select(0, aligned),
                ):
                    raise SystemExit(f"top-1 argmax memory mismatch in {path}")
                if not torch.allclose(
                    _as_float(step["previous_topk_probabilities"]),
                    _as_float(previous["current_topk_probabilities"]).index_select(0, aligned),
                    atol=2e-6,
                    rtol=0,
                ):
                    raise SystemExit(f"top-k probability memory mismatch in {path}")
            previous_step_by_block[block_index] = step

            full_history_values = full_jsd[full_available]
            sparse_history_values = sparse_jsd[history]
            gap_history_values = gap[full_available]
            arrival_history_values = arrival[history]
            influence_history_values = _as_float(step["influence_candidate_jsd"])[history]
            total_history += int(history.sum().item())
            total_stable += int(stable.sum().item())
            total_changed += int(changed.sum().item())
            total_eligible += int(step["effective_eligible"].sum().item())
            total_forced += int(step["forced_commits"])
            total_fallback_steps += int(step["fallback_used"])
            if sparse_history_values.numel():
                sparse_jsd_sum += float(sparse_history_values.sum().item())
                sparse_jsd_max = max(sparse_jsd_max, float(sparse_history_values.max().item()))
                arrival_sum += float(arrival_history_values.sum().item())
                arrival_max = max(arrival_max, float(arrival_history_values.max().item()))
                influence_sum += float(influence_history_values.sum().item())
                influence_max = max(influence_max, float(influence_history_values.max().item()))
                topk_overlap_sum += int(step["topk_overlap_count"][history].sum().item())
                topk_overlap_count += int(history.sum().item())
            if full_history_values.numel():
                full_jsd_sum += float(full_history_values.sum().item())
                full_jsd_max = max(full_jsd_max, float(full_history_values.max().item()))
                full_jsd_observations += int(full_history_values.numel())
                approximation_gap_sum += float(gap_history_values.sum().item())
                approximation_gap_max = max(
                    approximation_gap_max, float(gap_history_values.max().item())
                )
            entropy_sum += float(_as_float(step["entropy_nats"]).sum().item())
            entropy_count += count
            peak_runtime_positions = max(peak_runtime_positions, int(step["runtime_memory_positions"]))
            peak_runtime_full_elements = max(
                peak_runtime_full_elements, int(step["runtime_full_probability_elements"])
            )
            if not exact_jsd and int(step["runtime_full_probability_elements"]) != 0:
                raise SystemExit(f"O(|M|K) run retained full probabilities in {path}")
            peak_cuda_allocated = max(peak_cuda_allocated, int(step["cuda_memory_allocated_bytes"]))
            peak_cuda_reserved = max(peak_cuda_reserved, int(step["cuda_memory_reserved_bytes"]))
            wall_time_sum += float(step["step_wall_time_seconds"])

        generated_suffix = payload["final_sequence_token_ids"][int(payload["prompt_length"]):]
        if int((generated_suffix == mask_id).sum().item()) != 0:
            raise SystemExit(f"residual mask token in final sequence for {path}")
        total_steps += len(steps)
        total_bytes += path.stat().st_size
        nfe_values.append(len(steps))

    if len(task_ids) != len(set(task_ids)):
        raise SystemExit("duplicate task ids in diagnostics")
    summary = {
        "schema_version": "candidate_memory_steps_v2",
        "validator_version": "candidate_memory_validator_v2_tie_aware",
        "files": len(paths),
        "unique_task_ids": len(set(task_ids)),
        "total_steps": total_steps,
        "total_bytes": total_bytes,
        "nfe_per_file_min": min(nfe_values),
        "nfe_per_file_max": max(nfe_values),
        "history_candidates_total": total_history,
        "stable_candidates_total": total_stable,
        "changed_candidates_total": total_changed,
        "eligible_candidates_total": total_eligible,
        "forced_commits_total": total_forced,
        "fallback_steps_total": total_fallback_steps,
        "tie_boundary_frontiers_accepted": tie_boundary_frontiers_accepted,
        "full_jsd_observations": full_jsd_observations,
        "full_jsd_mean": (
            full_jsd_sum / full_jsd_observations if full_jsd_observations else None
        ),
        "full_jsd_max": full_jsd_max,
        "sparse_jsd_mean": sparse_jsd_sum / total_history if total_history else 0.0,
        "sparse_jsd_max": sparse_jsd_max,
        "jsd_approximation_gap_mean": (
            approximation_gap_sum / full_jsd_observations
            if full_jsd_observations
            else None
        ),
        "jsd_approximation_gap_max": approximation_gap_max,
        "arrival_mean": arrival_sum / total_history if total_history else 0.0,
        "arrival_max": arrival_max,
        "influence_mean": influence_sum / total_history if total_history else 0.0,
        "influence_max": influence_max,
        "entropy_mean_nats": entropy_sum / entropy_count if entropy_count else 0.0,
        "topk_overlap_mean": topk_overlap_sum / topk_overlap_count if topk_overlap_count else 0.0,
        "peak_runtime_memory_positions": peak_runtime_positions,
        "peak_runtime_full_probability_elements": peak_runtime_full_elements,
        "peak_cuda_allocated_bytes": peak_cuda_allocated,
        "peak_cuda_reserved_bytes": peak_cuda_reserved,
        "step_wall_time_seconds_total": wall_time_sum,
        "residual_mask_count": 0,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
