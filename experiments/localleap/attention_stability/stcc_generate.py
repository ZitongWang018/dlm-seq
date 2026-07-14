"""Distribution-response spatiotemporal decoding for LLaDA.

The persistent state is Top-K plus an OTHER bucket for every still-masked
position.  It is used only to schedule commits; it is never injected into the
Transformer.  Horizontal constraints are optional and preserve the baseline
budget unless an explicit acceleration multiplier is greater than one.
"""

import time

import torch
import torch.nn.functional as F

from generate import (
    _forward_with_block_attention,
    _jsd_from_probabilities,
    add_gumbel_noise,
    get_num_transfer_tokens,
)


def _confidence_order(masked, confidence, sequence_length):
    full_confidence = torch.full(
        (sequence_length,), -torch.inf, dtype=confidence.dtype, device=confidence.device
    )
    full_confidence[masked] = confidence
    positions = torch.topk(full_confidence, k=masked.numel()).indices.tolist()
    position_to_index = {int(position): idx for idx, position in enumerate(masked.tolist())}
    return [position_to_index[int(position)] for position in positions]


def _put_argmax_first(topk_ids, topk_probs, top1, probabilities):
    """Make Top-K memory agree with torch.argmax on BF16 probability ties."""
    for row_idx in range(topk_ids.shape[0]):
        matches = torch.where(topk_ids[row_idx] == top1[row_idx])[0]
        if matches.numel():
            slot = int(matches[0].item())
            if slot:
                first_id = topk_ids[row_idx, 0].clone()
                first_prob = topk_probs[row_idx, 0].clone()
                topk_ids[row_idx, 0] = topk_ids[row_idx, slot]
                topk_probs[row_idx, 0] = topk_probs[row_idx, slot]
                topk_ids[row_idx, slot] = first_id
                topk_probs[row_idx, slot] = first_prob
        else:
            topk_ids[row_idx, 1:] = topk_ids[row_idx, :-1].clone()
            topk_probs[row_idx, 1:] = topk_probs[row_idx, :-1].clone()
            topk_ids[row_idx, 0] = top1[row_idx]
            topk_probs[row_idx, 0] = probabilities[row_idx, top1[row_idx]]


def select_stcc_tokens(
    logits,
    mask_index,
    x,
    base_budget,
    block_start,
    candidate_topk=8,
    jsd_threshold=0.01,
    horizontal_mode="none",
    directional_attention=None,
    attention_threshold=0.004,
    extra_multiplier=1,
    extra_jsd_threshold=None,
    min_topk_overlap=None,
    min_stability_streak=1,
    previous_memory=None,
):
    """Select baseline commits and optional safe extra commits.

    Response classes, from most to least mature, are: stable top-1 with low
    partition JSD; changed top-1 with low JSD (near-boundary flip); stable top-1
    with high JSD; changed top-1 with high JSD.  This prevents top-1 identity
    from acting as the sole notion of maturity.
    """
    if horizontal_mode not in {"none", "symmetric", "directed"}:
        raise ValueError(f"unsupported horizontal mode: {horizontal_mode}")
    if candidate_topk < 2 or candidate_topk > logits.shape[-1]:
        raise ValueError("candidate_topk must be in [2, vocabulary_size]")
    if extra_multiplier < 1:
        raise ValueError("extra_multiplier must be at least one")
    if extra_jsd_threshold is None:
        extra_jsd_threshold = jsd_threshold
    if min_topk_overlap is None:
        min_topk_overlap = max(1, candidate_topk - 1)
    if previous_memory is None:
        previous_memory = [{} for _ in range(x.shape[0])]
    if horizontal_mode != "none" and directional_attention is None:
        raise ValueError("horizontal decoding requires directional attention")

    x0 = torch.argmax(add_gumbel_noise(logits, temperature=0.0), dim=-1)
    x0 = torch.where(mask_index, x0, x)
    transfer_index = torch.zeros_like(x, dtype=torch.bool)
    next_memory = [{} for _ in range(x.shape[0])]
    diagnostics = {
        "history_candidates": 0,
        "stable_candidates": 0,
        "changed_candidates": 0,
        "low_response_candidates": 0,
        "base_commits": 0,
        "extra_commits": 0,
        "horizontal_raw_directed_edges": 0,
        "horizontal_active_directed_edges": 0,
        "horizontal_pruned_low_response_edges": 0,
        "horizontal_rejections": 0,
        "forced_conflict_fills": 0,
        "partition_jsd_sum": 0.0,
        "partition_jsd_max": 0.0,
        "topk_overlap_sum": 0,
        "history_observations": 0,
        "selected_changed_candidates": 0,
        "selected_high_response_candidates": 0,
        "candidate_state": [],
    }

    for batch_idx in range(x.shape[0]):
        masked = torch.where(mask_index[batch_idx])[0]
        if not masked.numel():
            continue
        take_base = min(int(base_budget), int(masked.numel()))
        candidate_logits = logits[batch_idx, masked].to(torch.float64)
        probabilities = F.softmax(candidate_logits, dim=-1)
        current_top1 = x0[batch_idx, masked]
        confidence = probabilities.gather(1, current_top1.unsqueeze(-1)).squeeze(-1)
        topk_probs, topk_ids = torch.topk(probabilities, k=candidate_topk, dim=-1)
        _put_argmax_first(topk_ids, topk_probs, current_top1, probabilities)
        other_mass = (1.0 - topk_probs.sum(dim=-1)).clamp(0.0, 1.0)
        margin = topk_probs[:, 0] - topk_probs[:, 1]

        count = masked.numel()
        has_history = torch.zeros(count, dtype=torch.bool, device=x.device)
        previous_top1 = torch.full((count,), -1, dtype=torch.long, device=x.device)
        previous_confidence = torch.zeros(count, dtype=torch.float32, device=x.device)
        previous_streak = torch.zeros(count, dtype=torch.int16, device=x.device)
        previous_topk_ids = torch.full(
            (count, candidate_topk), -1, dtype=torch.long, device=x.device
        )
        previous_topk_probs = torch.zeros(
            (count, candidate_topk), dtype=torch.float32, device=x.device
        )
        previous_other = torch.ones(count, dtype=torch.float32, device=x.device)
        partition_jsd = torch.zeros(count, dtype=torch.float32, device=x.device)
        overlap = torch.zeros(count, dtype=torch.int16, device=x.device)

        for idx, position in enumerate(masked.tolist()):
            item = previous_memory[batch_idx].get(position)
            if item is None:
                continue
            has_history[idx] = True
            previous_top1[idx] = item["top1_id"]
            previous_confidence[idx] = item["top1_confidence"]
            previous_streak[idx] = item.get("stability_streak", 0)
            previous_topk_ids[idx] = item["topk_ids"]
            previous_topk_probs[idx] = item["topk_probs"]
            previous_other[idx] = item["other_mass"]
            current_on_previous = probabilities[idx].to(torch.float32).gather(
                0, item["topk_ids"]
            )
            current_other = (1.0 - current_on_previous.sum()).clamp(0.0, 1.0)
            old_partition = torch.cat((item["topk_probs"], item["other_mass"].view(1)))
            new_partition = torch.cat((current_on_previous, current_other.view(1)))
            partition_jsd[idx] = _jsd_from_probabilities(
                old_partition.unsqueeze(0), new_partition.unsqueeze(0)
            )[0]
            overlap[idx] = int(
                (topk_ids[idx].unsqueeze(1) == item["topk_ids"].unsqueeze(0))
                .any(dim=1)
                .sum()
                .item()
            )

        stable = has_history & (current_top1 == previous_top1)
        changed = has_history & ~stable
        low_response = has_history & (partition_jsd <= float(jsd_threshold))
        stability_streak = torch.where(
            stable & low_response,
            previous_streak.to(torch.int32) + 1,
            torch.zeros(count, dtype=torch.int32, device=x.device),
        ).to(torch.int16)

        confidence_order = _confidence_order(masked, confidence, x.shape[1])
        if not bool(has_history.any()):
            response_order = confidence_order
            response_class = torch.full(
                (count,), -1, dtype=torch.int8, device=x.device
            )
        else:
            response_class = torch.full((count,), 4, dtype=torch.int8, device=x.device)
            response_class[stable & low_response] = 0
            response_class[changed & low_response] = 1
            response_class[stable & ~low_response] = 2
            response_class[changed & ~low_response] = 3
            response_class[~has_history] = -1
            response_order = sorted(
                range(count),
                key=lambda idx: (
                    int(response_class[idx].item()),
                    float(partition_jsd[idx].item()),
                    -int(overlap[idx].item()),
                    -int(stability_streak[idx].item()),
                    -float(confidence[idx].item()),
                    confidence_order.index(idx),
                ),
            )

        if horizontal_mode == "none":
            attention_between = torch.zeros(
                (count, count), dtype=torch.float32, device=x.device
            )
            raw_directed = torch.zeros_like(attention_between, dtype=torch.bool)
            active_directed = raw_directed.clone()
            pair_conflict = raw_directed.clone()
        else:
            local = masked - block_start
            attention_between = directional_attention[batch_idx].index_select(
                0, local
            ).index_select(1, local).to(torch.float32)
            raw_directed = attention_between > float(attention_threshold)
            raw_directed.fill_diagonal_(False)
            # A target with no history is conservatively informative.  A target
            # with low cross-step response prunes dense but ineffective reads.
            informative_target = (~has_history) | (~low_response)
            active_directed = raw_directed & informative_target.unsqueeze(1)
            if horizontal_mode == "directed":
                pair_conflict = active_directed | active_directed.transpose(0, 1)
            else:
                symmetric = 0.5 * (attention_between + attention_between.transpose(0, 1))
                informative_pair = informative_target.unsqueeze(1) | informative_target.unsqueeze(0)
                pair_conflict = (symmetric > float(attention_threshold)) & informative_pair
                pair_conflict.fill_diagonal_(False)

        selected = []
        rejected = []
        for idx in response_order:
            if len(selected) >= take_base:
                break
            if selected and bool(pair_conflict[idx, selected].any()):
                rejected.append(idx)
                continue
            selected.append(idx)

        if len(selected) < take_base:
            for idx in response_order:
                if idx in selected:
                    continue
                selected.append(idx)
                diagnostics["forced_conflict_fills"] += 1
                if len(selected) >= take_base:
                    break

        base_selected = list(selected)
        extra_limit = min(int(masked.numel()), int(extra_multiplier) * take_base)
        if extra_limit > len(selected) and bool(has_history.any()):
            safe_extra = (
                has_history
                & stable
                & (partition_jsd <= float(extra_jsd_threshold))
                & (overlap >= int(min_topk_overlap))
                & (stability_streak >= int(min_stability_streak))
            )
            for idx in response_order:
                if len(selected) >= extra_limit:
                    break
                if idx in selected or not bool(safe_extra[idx]):
                    continue
                if selected and bool(pair_conflict[idx, selected].any()):
                    continue
                selected.append(idx)

        selected_positions = masked[
            torch.tensor(selected, dtype=torch.long, device=x.device)
        ]
        transfer_index[batch_idx, selected_positions] = True
        selected_set = set(selected)

        for idx, position in enumerate(masked.tolist()):
            if idx in selected_set:
                continue
            next_memory[batch_idx][position] = {
                "top1_id": current_top1[idx].to(torch.long).detach().clone(),
                "top1_confidence": confidence[idx].to(torch.float32).detach().clone(),
                "topk_ids": topk_ids[idx].to(torch.long).detach().clone(),
                "topk_probs": topk_probs[idx].to(torch.float32).detach().clone(),
                "other_mass": other_mass[idx].to(torch.float32).detach().clone(),
                "stability_streak": stability_streak[idx].detach().clone(),
            }

        raw_edges = int(raw_directed.sum().item())
        active_edges = int(active_directed.sum().item())
        diagnostics["history_candidates"] += int(has_history.sum().item())
        diagnostics["stable_candidates"] += int(stable.sum().item())
        diagnostics["changed_candidates"] += int(changed.sum().item())
        diagnostics["low_response_candidates"] += int(low_response.sum().item())
        diagnostics["base_commits"] += len(base_selected)
        diagnostics["extra_commits"] += len(selected) - len(base_selected)
        diagnostics["horizontal_raw_directed_edges"] += raw_edges
        diagnostics["horizontal_active_directed_edges"] += active_edges
        diagnostics["horizontal_pruned_low_response_edges"] += raw_edges - active_edges
        diagnostics["horizontal_rejections"] += len(rejected)
        history_values = partition_jsd[has_history]
        diagnostics["history_observations"] += int(has_history.sum().item())
        if history_values.numel():
            diagnostics["partition_jsd_sum"] += float(history_values.sum().item())
            diagnostics["partition_jsd_max"] = max(
                diagnostics["partition_jsd_max"], float(history_values.max().item())
            )
            diagnostics["topk_overlap_sum"] += int(overlap[has_history].sum().item())
        selected_tensor = torch.tensor(selected, dtype=torch.long, device=x.device)
        diagnostics["selected_changed_candidates"] += int(
            changed[selected_tensor].sum().item()
        )
        diagnostics["selected_high_response_candidates"] += int(
            (has_history & ~low_response)[selected_tensor].sum().item()
        )
        diagnostics["candidate_state"].append({
            "masked_positions_global": masked.to(torch.int32).cpu(),
            "masked_positions_local": (masked - block_start).to(torch.int16).cpu(),
            "current_top1_token_ids": current_top1.to(torch.int32).cpu(),
            "previous_top1_token_ids": previous_top1.to(torch.int32).cpu(),
            "top1_confidences": confidence.to(torch.float64).cpu(),
            "previous_top1_confidences": previous_confidence.cpu(),
            "top1_margin": margin.to(torch.float32).cpu(),
            "current_topk_token_ids": topk_ids.to(torch.int32).cpu(),
            "current_topk_probabilities": topk_probs.to(torch.float32).cpu(),
            "current_other_mass": other_mass.to(torch.float32).cpu(),
            "previous_topk_token_ids": previous_topk_ids.to(torch.int32).cpu(),
            "previous_topk_probabilities": previous_topk_probs.cpu(),
            "previous_other_mass": previous_other.cpu(),
            "has_history": has_history.cpu(),
            "top1_stable": stable.cpu(),
            "candidate_changed": changed.cpu(),
            "partition_jsd_nats": partition_jsd.cpu(),
            "low_distribution_response": low_response.cpu(),
            "topk_overlap_count": overlap.cpu(),
            "stability_streak": stability_streak.cpu(),
            "response_class": response_class.cpu(),
            "response_order_positions_global": masked[
                torch.tensor(response_order, dtype=torch.long, device=x.device)
            ].to(torch.int32).cpu(),
            "attention_between_masked": attention_between.to(torch.float16).cpu(),
            "raw_directed_edges": raw_directed.cpu(),
            "active_directed_edges": active_directed.cpu(),
            "pair_conflict": pair_conflict.cpu(),
            "base_selected_positions_global": masked[
                torch.tensor(base_selected, dtype=torch.long, device=x.device)
            ].to(torch.int32).cpu(),
            "extra_selected_positions_global": masked[
                torch.tensor(selected[len(base_selected):], dtype=torch.long, device=x.device)
            ].to(torch.int32).cpu(),
            "selected_positions_global": selected_positions.to(torch.int32).cpu(),
            "rejected_positions_global": masked[
                torch.tensor(rejected, dtype=torch.long, device=x.device)
            ].to(torch.int32).cpu(),
        })

    diagnostics["runtime_memory_positions"] = sum(len(items) for items in next_memory)
    diagnostics["runtime_memory_topk_elements"] = (
        diagnostics["runtime_memory_positions"] * candidate_topk
    )
    return x0, transfer_index, diagnostics, next_memory


@torch.no_grad()
def generate_stcc(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    mask_id=126336,
    eos_id=126081,
    early_stop=False,
    candidate_topk=8,
    jsd_threshold=0.01,
    horizontal_mode="none",
    attention_threshold=0.004,
    extra_multiplier=1,
    extra_jsd_threshold=None,
    min_topk_overlap=None,
    min_stability_streak=1,
    collect_step_diagnostics=False,
):
    """Generate with distribution response, optional horizontal control, and extras."""
    x = torch.full(
        (1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=model.device
    )
    x[:, :prompt.shape[1]] = prompt.clone()
    if gen_length % block_length:
        raise ValueError("gen_length must be divisible by block_length")
    num_blocks = gen_length // block_length
    if steps % num_blocks:
        raise ValueError("steps must be divisible by number of blocks")
    block_steps = steps // num_blocks
    nfe = 0
    step_records = []
    summary = {
        "decoder": "stcc_distribution_response_v1",
        "candidate_topk": int(candidate_topk),
        "jsd_threshold": float(jsd_threshold),
        "horizontal_mode": horizontal_mode,
        "attention_threshold": float(attention_threshold),
        "extra_multiplier": int(extra_multiplier),
        "extra_jsd_threshold": float(
            jsd_threshold if extra_jsd_threshold is None else extra_jsd_threshold
        ),
        "min_topk_overlap": int(
            max(1, candidate_topk - 1) if min_topk_overlap is None else min_topk_overlap
        ),
        "min_stability_streak": int(min_stability_streak),
        "configured_steps": int(steps),
        "history_candidates": 0,
        "stable_candidates": 0,
        "changed_candidates": 0,
        "low_response_candidates": 0,
        "base_commits": 0,
        "extra_commits": 0,
        "horizontal_raw_directed_edges": 0,
        "horizontal_active_directed_edges": 0,
        "horizontal_pruned_low_response_edges": 0,
        "horizontal_rejections": 0,
        "forced_conflict_fills": 0,
        "partition_jsd_sum": 0.0,
        "partition_jsd_max": 0.0,
        "topk_overlap_sum": 0,
        "history_observations": 0,
        "selected_changed_candidates": 0,
        "selected_high_response_candidates": 0,
        "step_wall_time_seconds": 0.0,
        "peak_runtime_memory_positions": 0,
        "residual_mask_count": 0,
    }

    for block_idx in range(num_blocks):
        block_start = prompt.shape[1] + block_idx * block_length
        block_end = block_start + block_length
        initial_mask = x[:, block_start:block_end] == mask_id
        schedule = get_num_transfer_tokens(initial_mask, block_steps)
        previous_memory = None
        step_idx = 0
        while bool((x[:, block_start:block_end] == mask_id).any()):
            if step_idx >= block_steps:
                raise RuntimeError("STCC exceeded the baseline block-step budget")
            if x.is_cuda:
                torch.cuda.synchronize(x.device)
            step_start = time.perf_counter()
            mask_index = x == mask_id
            mask_index[:, block_end:] = False
            remaining = int(mask_index[0].sum().item())
            base_budget = min(int(schedule[0, step_idx].item()), remaining)
            nfe += 1
            if horizontal_mode == "none":
                logits = model(x).logits
                directional_attention = None
            else:
                logits, directional_attention, _ = _forward_with_block_attention(
                    model, x, block_start, block_end
                )
            x0, transfer, step_diag, next_memory = select_stcc_tokens(
                logits=logits,
                mask_index=mask_index,
                x=x,
                base_budget=base_budget,
                block_start=block_start,
                candidate_topk=candidate_topk,
                jsd_threshold=jsd_threshold,
                horizontal_mode=horizontal_mode,
                directional_attention=directional_attention,
                attention_threshold=attention_threshold,
                extra_multiplier=extra_multiplier,
                extra_jsd_threshold=extra_jsd_threshold,
                min_topk_overlap=min_topk_overlap,
                min_stability_streak=min_stability_streak,
                previous_memory=previous_memory,
            )
            selected = torch.where(transfer[0])[0]
            if selected.numel() < base_budget:
                raise RuntimeError("STCC failed to preserve the baseline commit budget")
            if int(extra_multiplier) == 1 and selected.numel() != base_budget:
                raise RuntimeError("quality STCC changed the baseline commit budget")
            if x.is_cuda:
                torch.cuda.synchronize(x.device)
            elapsed = time.perf_counter() - step_start

            for key in (
                "history_candidates",
                "stable_candidates",
                "changed_candidates",
                "low_response_candidates",
                "base_commits",
                "extra_commits",
                "horizontal_raw_directed_edges",
                "horizontal_active_directed_edges",
                "horizontal_pruned_low_response_edges",
                "horizontal_rejections",
                "forced_conflict_fills",
                "topk_overlap_sum",
                "history_observations",
                "selected_changed_candidates",
                "selected_high_response_candidates",
            ):
                summary[key] += int(step_diag[key])
            summary["partition_jsd_sum"] += float(step_diag["partition_jsd_sum"])
            summary["partition_jsd_max"] = max(
                summary["partition_jsd_max"], float(step_diag["partition_jsd_max"])
            )
            summary["step_wall_time_seconds"] += elapsed
            summary["peak_runtime_memory_positions"] = max(
                summary["peak_runtime_memory_positions"],
                int(step_diag["runtime_memory_positions"]),
            )
            if collect_step_diagnostics:
                state = step_diag["candidate_state"][0]
                step_records.append({
                    "block_index": block_idx,
                    "step_index_in_block": step_idx,
                    "global_nfe": nfe,
                    "block_start": block_start,
                    "block_end": block_end,
                    "base_budget": base_budget,
                    "selected_count": int(selected.numel()),
                    "extra_selected_count": int(selected.numel()) - base_budget,
                    "mask_count_before": remaining,
                    "mask_count_after": remaining - int(selected.numel()),
                    "input_block_token_ids": x[0, block_start:block_end].to(torch.int32).cpu(),
                    "selected_positions_global": selected.to(torch.int32).cpu(),
                    "selected_token_ids": x0[0, selected].to(torch.int32).cpu(),
                    "step_wall_time_seconds": elapsed,
                    "horizontal_raw_directed_edges": step_diag["horizontal_raw_directed_edges"],
                    "horizontal_active_directed_edges": step_diag["horizontal_active_directed_edges"],
                    "horizontal_pruned_low_response_edges": step_diag[
                        "horizontal_pruned_low_response_edges"
                    ],
                    "horizontal_rejections": step_diag["horizontal_rejections"],
                    "forced_conflict_fills": step_diag["forced_conflict_fills"],
                    "runtime_memory_positions": step_diag["runtime_memory_positions"],
                    "runtime_memory_topk_elements": step_diag["runtime_memory_topk_elements"],
                    **state,
                })
            previous_memory = next_memory
            x[transfer] = x0[transfer]
            step_idx += 1

        if early_stop and bool((x[:, block_start:block_end] == eos_id).any()):
            x[:, block_end:] = eos_id
            break

    summary["actual_nfe"] = int(nfe)
    summary["nfe_reduction"] = int(steps) - int(nfe)
    summary["mean_commits_per_nfe"] = float(gen_length) / max(1, int(nfe))
    if summary["history_observations"]:
        summary["partition_jsd_mean"] = (
            summary["partition_jsd_sum"] / summary["history_observations"]
        )
        summary["topk_overlap_mean"] = (
            summary["topk_overlap_sum"] / summary["history_observations"]
        )
    summary["residual_mask_count"] = int((x == mask_id).sum().item())
    if collect_step_diagnostics:
        summary["_step_records"] = step_records
    return x, nfe, summary
