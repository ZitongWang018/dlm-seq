# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA and Fast-dLLM repos: https://github.com/ML-GSAI/LLaDA
# and https://github.com/NVlabs/Fast-dLLM

import torch
import numpy as np
import torch.nn.functional as F
import os
import time
from typing import Tuple, Union, Any, List, Dict
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def _dependency_blocks(model):
    """Return decoder blocks without changing or wrapping the model."""
    core = getattr(model, "module", model)
    return core.model.transformer.blocks


def _forward_with_block_attention(model, x, block_start, block_end):
    """Run the ordinary forward and average requested attention over all layers/heads."""
    positions = torch.arange(block_start, block_end, device=x.device)
    tracker = {"positions": positions, "sum": None, "count": 0}
    blocks = _dependency_blocks(model)
    for block in blocks:
        block._dependency_tracker = tracker
    try:
        output = model(x)
    finally:
        for block in blocks:
            block._dependency_tracker = None
    if tracker["count"] != len(blocks):
        raise RuntimeError(f"dependency probe saw {tracker['count']} layers, expected {len(blocks)}")
    directional = tracker["sum"] / tracker["count"]
    symmetric = 0.5 * (directional + directional.transpose(-2, -1))
    return output.logits, directional, symmetric


def select_attention_stability_tokens(
    logits,
    temperature,
    remasking,
    mask_index,
    x,
    budget,
    dependency,
    dependency_threshold,
    block_start,
    previous_top1=None,
    previous_selected=None,
):
    """Lexicographic maturity/confidence ordering plus greedy dependency exclusion."""
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    if remasking == 'low_confidence':
        probabilities = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(probabilities, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == 'random':
        x0_p = torch.rand(x0.shape, device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    diagnostics = {
        "unstable_candidates": 0,
        "changed_candidates": 0,
        "strongly_dependent_candidates": 0,
        "rejected_pairs": 0,
        "underfilled": False,
        "all_immature_fallback": False,
        "candidate_state": [],
    }

    for batch_idx in range(x.shape[0]):
        masked = torch.where(mask_index[batch_idx])[0]
        if masked.numel() == 0:
            continue
        maturity = torch.ones(masked.numel(), dtype=torch.bool, device=x.device)
        changed = torch.zeros(masked.numel(), dtype=torch.bool, device=x.device)
        max_dependency = torch.zeros(masked.numel(), dtype=torch.float32, device=x.device)
        if previous_top1 is not None and previous_selected is not None and previous_selected.numel() > 0:
            masked_local = masked - block_start
            selected_local = previous_selected - block_start
            max_dependency = dependency[batch_idx].index_select(0, masked_local).index_select(1, selected_local).max(dim=1).values
            changed = x0[batch_idx, masked] != previous_top1[batch_idx, masked]
            maturity = ~((max_dependency > dependency_threshold) & changed)
            diagnostics["unstable_candidates"] += int((~maturity).sum().item())
            diagnostics["changed_candidates"] += int(changed.sum().item())
            diagnostics["strongly_dependent_candidates"] += int((max_dependency > dependency_threshold).sum().item())

        mature_positions = masked[maturity]
        immature_positions = masked[~maturity]
        if mature_positions.numel() == 0:
            # Explicit all-immature fallback from the algorithm definition.
            ordered = masked[torch.topk(confidence[batch_idx, masked], k=masked.numel()).indices]
            diagnostics["all_immature_fallback"] = True
        else:
            mature_order = mature_positions[torch.topk(confidence[batch_idx, mature_positions], k=mature_positions.numel()).indices]
            if immature_positions.numel() > 0:
                immature_order = immature_positions[torch.topk(confidence[batch_idx, immature_positions], k=immature_positions.numel()).indices]
                ordered = torch.cat((mature_order, immature_order))
            else:
                ordered = mature_order

        selected = []
        rejected = []
        for position in ordered.tolist():
            if len(selected) >= int(budget):
                break
            if selected:
                local_position = position - block_start
                selected_local = torch.tensor([item - block_start for item in selected], device=x.device)
                if dependency[batch_idx, local_position, selected_local].max() > dependency_threshold:
                    diagnostics["rejected_pairs"] += 1
                    rejected.append(position)
                    continue
            selected.append(position)
        if len(selected) < min(int(budget), masked.numel()):
            diagnostics["underfilled"] = True
        if selected:
            transfer_index[batch_idx, torch.tensor(selected, device=x.device)] = True

        previous_values = (
            previous_top1[batch_idx, masked]
            if previous_top1 is not None
            else torch.full_like(masked, -1)
        )
        diagnostics["candidate_state"].append({
            "masked_positions_global": masked.to(torch.int32).cpu(),
            "masked_positions_local": (masked - block_start).to(torch.int16).cpu(),
            "top1_token_ids": x0[batch_idx, masked].to(torch.int32).cpu(),
            "top1_confidences": confidence[batch_idx, masked].to(torch.float32).cpu(),
            "previous_top1_token_ids": previous_values.to(torch.int32).cpu(),
            "candidate_changed": changed.cpu(),
            "max_dependency_to_previous": max_dependency.to(torch.float32).cpu(),
            "maturity": maturity.cpu(),
            "ordered_positions_global": ordered.to(torch.int32).cpu(),
            "selected_positions_global": torch.tensor(selected, dtype=torch.int32),
            "rejected_positions_global": torch.tensor(rejected, dtype=torch.int32),
        })

    return x0, transfer_index, diagnostics


@torch.no_grad()
def generate_attention_stability(
    model,
    prompt,
    dependency_threshold,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.,
    remasking='low_confidence',
    mask_id=126336,
    eos_id=126081,
    early_stop=False,
    collect_step_diagnostics=False,
):
    """Baseline LLaDA decoding with the proposed attention/stability selection layered on top."""
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    block_steps = steps // num_blocks
    nfe = 0
    summary = {
        "decoder": "attention_stability_v1",
        "dependency_threshold": float(dependency_threshold),
        "configured_steps": int(steps),
        "gen_length": int(gen_length),
        "block_length": int(block_length),
        "unstable_candidates": 0,
        "changed_candidates": 0,
        "strongly_dependent_candidates": 0,
        "rejected_pairs": 0,
        "underfilled_steps": 0,
        "all_immature_fallback_steps": 0,
        "dependency_max": 0.0,
        "dependency_mean_sum": 0.0,
        "dependency_observations": 0,
        "attention_asymmetry_max": 0.0,
        "attention_asymmetry_mean_sum": 0.0,
    }
    step_records = []

    for block_idx in range(num_blocks):
        block_start = prompt.shape[1] + block_idx * block_length
        block_end = block_start + block_length
        initial_mask = x[:, block_start:block_end] == mask_id
        transfer_schedule = get_num_transfer_tokens(initial_mask, block_steps)
        step_idx = 0
        previous_top1 = None
        previous_selected = None
        while (x[:, block_start:block_end] == mask_id).any():
            if step_idx >= block_steps:
                # Conservative under-filling may require extra forwards.  Continue
                # with the last baseline budget rather than silently relaxing tau.
                schedule_idx = block_steps - 1
            else:
                schedule_idx = step_idx
            budget = int(transfer_schedule[0, schedule_idx].item())
            nfe += 1
            mask_index = x == mask_id
            mask_index[:, block_end:] = False
            logits, directional_attention, dependency = _forward_with_block_attention(model, x, block_start, block_end)
            summary["dependency_max"] = max(summary["dependency_max"], float(dependency.max().item()))
            summary["dependency_mean_sum"] += float(dependency.mean().item())
            summary["dependency_observations"] += 1
            asymmetry = (directional_attention - directional_attention.transpose(-2, -1)).abs()
            summary["attention_asymmetry_max"] = max(
                summary["attention_asymmetry_max"], float(asymmetry.max().item())
            )
            summary["attention_asymmetry_mean_sum"] += float(asymmetry.mean().item())
            x0, transfer_index, step_diagnostics = select_attention_stability_tokens(
                logits=logits,
                temperature=temperature,
                remasking=remasking,
                mask_index=mask_index,
                x=x,
                budget=budget,
                dependency=dependency,
                dependency_threshold=dependency_threshold,
                block_start=block_start,
                previous_top1=previous_top1,
                previous_selected=previous_selected,
            )
            if not transfer_index.any():
                raise RuntimeError("attention-stability selector made no progress")
            summary["unstable_candidates"] += step_diagnostics["unstable_candidates"]
            summary["changed_candidates"] += step_diagnostics["changed_candidates"]
            summary["strongly_dependent_candidates"] += step_diagnostics["strongly_dependent_candidates"]
            summary["rejected_pairs"] += step_diagnostics["rejected_pairs"]
            summary["underfilled_steps"] += int(step_diagnostics["underfilled"])
            summary["all_immature_fallback_steps"] += int(step_diagnostics["all_immature_fallback"])
            if collect_step_diagnostics:
                candidate_state = step_diagnostics["candidate_state"][0]
                selected_positions = torch.where(transfer_index[0])[0]
                previous_selected_cpu = (
                    previous_selected.to(torch.int32).cpu()
                    if previous_selected is not None
                    else torch.empty(0, dtype=torch.int32)
                )
                step_records.append({
                    "block_index": block_idx,
                    "step_index_in_block": step_idx,
                    "global_nfe": nfe,
                    "block_start": block_start,
                    "block_end": block_end,
                    "schedule_index": schedule_idx,
                    "budget": budget,
                    "mask_count_before": int(mask_index[0].sum().item()),
                    "mask_count_after": int(mask_index[0].sum().item() - transfer_index[0].sum().item()),
                    "input_block_token_ids": x[0, block_start:block_end].to(torch.int32).cpu(),
                    "previous_selected_positions_global": previous_selected_cpu,
                    "previous_selected_token_ids": (
                        x[0, previous_selected].to(torch.int32).cpu()
                        if previous_selected is not None
                        else torch.empty(0, dtype=torch.int32)
                    ),
                    "selected_positions_global": selected_positions.to(torch.int32).cpu(),
                    "selected_positions_local": (selected_positions - block_start).to(torch.int16).cpu(),
                    "selected_token_ids": x0[0, selected_positions].to(torch.int32).cpu(),
                    "directional_attention": directional_attention[0].to(torch.float16).cpu(),
                    "symmetric_dependency": dependency[0].to(torch.float16).cpu(),
                    "dependency_min": float(dependency.min().item()),
                    "dependency_mean": float(dependency.mean().item()),
                    "dependency_max": float(dependency.max().item()),
                    "attention_asymmetry_mean": float(asymmetry.mean().item()),
                    "attention_asymmetry_max": float(asymmetry.max().item()),
                    "unstable_candidates": step_diagnostics["unstable_candidates"],
                    "changed_candidates": step_diagnostics["changed_candidates"],
                    "strongly_dependent_candidates": step_diagnostics["strongly_dependent_candidates"],
                    "rejected_pairs": step_diagnostics["rejected_pairs"],
                    "underfilled": step_diagnostics["underfilled"],
                    "all_immature_fallback": step_diagnostics["all_immature_fallback"],
                    **candidate_state,
                })
            previous_top1 = x0.detach().clone()
            previous_selected = torch.where(transfer_index[0])[0]
            x[transfer_index] = x0[transfer_index]
            step_idx += 1

        if early_stop and (x[:, block_start:block_end] == eos_id).any():
            x[:, block_end:] = eos_id
            break

    if summary["dependency_observations"]:
        summary["dependency_mean"] = summary.pop("dependency_mean_sum") / summary["dependency_observations"]
        summary["attention_asymmetry_mean"] = (
            summary.pop("attention_asymmetry_mean_sum") / summary["dependency_observations"]
        )
    if collect_step_diagnostics:
        summary["_step_records"] = step_records
    return x, nfe, summary


def _jsd_from_probabilities(first, second):
    """Jensen-Shannon divergence in nats for normalized rows."""
    first = first.to(torch.float32)
    second = second.to(torch.float32)
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
    return 0.5 * (first_term.sum(dim=-1) + second_term.sum(dim=-1))


def select_candidate_memory_tokens(
    logits,
    temperature,
    remasking,
    mask_index,
    x,
    budget,
    directional_attention,
    block_start,
    candidate_topk,
    confidence_threshold,
    fallback_mode,
    collect_exact_jsd=False,
    previous_memory=None,
    previous_selected=None,
):
    """Select a fixed baseline budget using cross-step candidate stability.

    ``previous_memory`` contains only positions that remained masked after the
    preceding step.  Top-K plus OTHER is the default persistent state, giving
    O(|M|K) history.  An exact full-vocabulary buffer can be enabled explicitly
    for a diagnostic ablation; it never changes selection and is never written
    to the diagnostics files.
    """
    if candidate_topk < 1 or candidate_topk > logits.shape[-1]:
        raise ValueError("candidate_topk must be between 1 and vocabulary size")
    if fallback_mode not in {"confidence", "impact", "frontier"}:
        raise ValueError(f"unsupported candidate-memory fallback: {fallback_mode}")

    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    x0 = torch.where(mask_index, x0, x)
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if previous_memory is None:
        previous_memory = [{} for _ in range(x.shape[0])]
    next_memory = [{} for _ in range(x.shape[0])]
    diagnostics = {
        "history_candidates": 0,
        "stable_candidates": 0,
        "changed_candidates": 0,
        "high_confidence_candidates": 0,
        "eligible_candidates": 0,
        "forced_commits": 0,
        "fallback_used": False,
        "fallback_reason": "none",
        "candidate_state": [],
    }

    for batch_idx in range(x.shape[0]):
        masked = torch.where(mask_index[batch_idx])[0]
        if masked.numel() == 0:
            continue

        # Restrict the expensive vocabulary softmax to currently masked rows.
        # Float64 matches the original low-confidence selector's probabilities.
        candidate_logits = logits[batch_idx, masked].to(torch.float64)
        if remasking == "low_confidence":
            current_probabilities = F.softmax(candidate_logits, dim=-1)
            current_top1 = x0[batch_idx, masked]
            confidence = current_probabilities.gather(
                1, current_top1.unsqueeze(-1)
            ).squeeze(-1)
        elif remasking == "random":
            current_probabilities = F.softmax(candidate_logits, dim=-1)
            current_top1 = x0[batch_idx, masked]
            confidence = torch.rand(masked.shape, device=x.device, dtype=torch.float64)
        else:
            raise NotImplementedError(remasking)

        current_topk_probs, current_topk_ids = torch.topk(
            current_probabilities, k=candidate_topk, dim=-1
        )
        # torch.argmax and torch.topk use different tie breaks on BF16 plateaus.
        # Make the actual argmax candidate the first memory entry explicitly.
        for candidate_idx in range(masked.numel()):
            top1_id = current_top1[candidate_idx]
            matches = torch.where(current_topk_ids[candidate_idx] == top1_id)[0]
            if matches.numel():
                top1_slot = int(matches[0].item())
                if top1_slot != 0:
                    saved_id = current_topk_ids[candidate_idx, 0].clone()
                    saved_probability = current_topk_probs[candidate_idx, 0].clone()
                    current_topk_ids[candidate_idx, 0] = current_topk_ids[
                        candidate_idx, top1_slot
                    ]
                    current_topk_probs[candidate_idx, 0] = current_topk_probs[
                        candidate_idx, top1_slot
                    ]
                    current_topk_ids[candidate_idx, top1_slot] = saved_id
                    current_topk_probs[candidate_idx, top1_slot] = saved_probability
            else:
                current_topk_ids[candidate_idx, 1:] = current_topk_ids[
                    candidate_idx, :-1
                ].clone()
                current_topk_probs[candidate_idx, 1:] = current_topk_probs[
                    candidate_idx, :-1
                ].clone()
                current_topk_ids[candidate_idx, 0] = top1_id
                current_topk_probs[candidate_idx, 0] = current_probabilities[
                    candidate_idx, top1_id
                ]
        current_other_mass = (1.0 - current_topk_probs.sum(dim=-1)).clamp(min=0.0, max=1.0)
        top1_margin = (
            current_topk_probs[:, 0] - current_topk_probs[:, 1]
            if candidate_topk > 1
            else current_topk_probs[:, 0]
        )
        entropy = -torch.where(
            current_probabilities > 0,
            current_probabilities * torch.log(current_probabilities),
            torch.zeros_like(current_probabilities),
        ).sum(dim=-1)

        count = masked.numel()
        has_history = torch.zeros(count, dtype=torch.bool, device=x.device)
        previous_topk_ids = torch.full(
            (count, candidate_topk), -1, dtype=torch.long, device=x.device
        )
        previous_topk_probs = torch.zeros(
            (count, candidate_topk), dtype=torch.float32, device=x.device
        )
        previous_other_mass = torch.ones(count, dtype=torch.float32, device=x.device)
        previous_confidence = torch.zeros(count, dtype=torch.float32, device=x.device)
        previous_top1 = torch.full((count,), -1, dtype=torch.long, device=x.device)
        full_jsd = torch.zeros(count, dtype=torch.float32, device=x.device)
        full_jsd_available = torch.zeros(count, dtype=torch.bool, device=x.device)
        sparse_jsd = torch.zeros(count, dtype=torch.float32, device=x.device)
        current_probs_on_previous_topk = torch.zeros(
            (count, candidate_topk), dtype=torch.float32, device=x.device
        )
        current_other_on_previous_partition = torch.ones(
            count, dtype=torch.float32, device=x.device
        )
        previous_top1_current_probability = torch.zeros(
            count, dtype=torch.float32, device=x.device
        )
        overlap_count = torch.zeros(count, dtype=torch.int16, device=x.device)

        for candidate_idx, global_position in enumerate(masked.tolist()):
            memory_item = previous_memory[batch_idx].get(global_position)
            if memory_item is None:
                continue
            has_history[candidate_idx] = True
            previous_topk_ids[candidate_idx] = memory_item["topk_ids"]
            previous_topk_probs[candidate_idx] = memory_item["topk_probs"]
            previous_other_mass[candidate_idx] = memory_item["other_mass"]
            previous_top1[candidate_idx] = memory_item["top1_id"]
            previous_confidence[candidate_idx] = memory_item["top1_confidence"]

            current_row = current_probabilities[candidate_idx].to(torch.float32)
            if "full_probabilities" in memory_item:
                previous_row = memory_item["full_probabilities"]
                full_jsd[candidate_idx] = _jsd_from_probabilities(
                    previous_row.unsqueeze(0), current_row.unsqueeze(0)
                )[0]
                full_jsd_available[candidate_idx] = True
            current_on_previous = current_row.gather(0, memory_item["topk_ids"])
            current_other = (1.0 - current_on_previous.sum()).clamp(min=0.0, max=1.0)
            current_probs_on_previous_topk[candidate_idx] = current_on_previous
            current_other_on_previous_partition[candidate_idx] = current_other
            sparse_previous = torch.cat(
                (memory_item["topk_probs"], memory_item["other_mass"].view(1))
            )
            sparse_current = torch.cat((current_on_previous, current_other.view(1)))
            sparse_jsd[candidate_idx] = _jsd_from_probabilities(
                sparse_previous.unsqueeze(0), sparse_current.unsqueeze(0)
            )[0]
            previous_top1_current_probability[candidate_idx] = current_row[
                memory_item["top1_id"]
            ]
            overlap_count[candidate_idx] = int(
                (current_topk_ids[candidate_idx].unsqueeze(1) == memory_item["topk_ids"].unsqueeze(0))
                .any(dim=1)
                .sum()
                .item()
            )

        stable = has_history & (current_top1 == previous_top1)
        changed = has_history & ~stable
        high_confidence = confidence > float(confidence_threshold)
        # The first step of every block is the baseline TopB bootstrap.
        eligible = (~has_history) | (stable & high_confidence)

        attention_to_previous = torch.empty(
            (count, 0), dtype=torch.float32, device=x.device
        )
        reverse_attention_from_previous = torch.empty(
            (count, 0), dtype=torch.float32, device=x.device
        )
        arrival = torch.zeros(count, dtype=torch.float32, device=x.device)
        if previous_selected is not None and previous_selected.numel() > 0:
            masked_local = masked - block_start
            previous_selected_local = previous_selected - block_start
            attention_to_previous = directional_attention[batch_idx].index_select(
                0, masked_local
            ).index_select(1, previous_selected_local).to(torch.float32)
            reverse_attention_from_previous = directional_attention[batch_idx].index_select(
                0, previous_selected_local
            ).index_select(1, masked_local).transpose(0, 1).to(torch.float32)
            arrival = attention_to_previous.sum(dim=1)
        # The formal O(|M|K) method uses the previous-Top-K partition JSD.
        # Optional exact JSD is diagnostic only and never changes the trajectory.
        decision_jsd = sparse_jsd
        influence = arrival * decision_jsd

        # Reconstruct the baseline's full-sequence confidence vector.  Calling
        # torch.topk on this vector with the actual transfer budget preserves
        # its tie behavior exactly; sorting only the masked subvector does not.
        full_confidence = torch.full(
            (x.shape[1],), -torch.inf, dtype=confidence.dtype, device=x.device
        )
        full_confidence[masked] = confidence
        position_to_candidate = {
            int(position): idx for idx, position in enumerate(masked.tolist())
        }
        confidence_positions = torch.topk(full_confidence, k=count).indices.tolist()
        confidence_order = [position_to_candidate[int(position)] for position in confidence_positions]
        confidence_rank = torch.empty(count, dtype=torch.int16, device=x.device)
        for rank, idx in enumerate(confidence_order, start=1):
            confidence_rank[idx] = rank
        frontier_size = min(count, int(budget) + 1)
        in_frontier = torch.zeros(count, dtype=torch.bool, device=x.device)
        frontier_positions = torch.topk(full_confidence, k=frontier_size).indices.tolist()
        frontier_indices = [position_to_candidate[int(position)] for position in frontier_positions]
        in_frontier[torch.tensor(frontier_indices, dtype=torch.long, device=x.device)] = True
        effective_eligible = eligible
        if fallback_mode == "frontier" and bool(has_history.any()):
            effective_eligible = eligible & in_frontier

        eligible_indices = [idx for idx in confidence_order if bool(effective_eligible[idx])]
        eligible_confidence = torch.full_like(full_confidence, -torch.inf)
        eligible_positions = masked[effective_eligible]
        eligible_confidence[eligible_positions] = full_confidence[eligible_positions]
        eligible_take = min(int(budget), int(effective_eligible.sum().item()))
        if eligible_take:
            selected_positions_from_eligible = torch.topk(
                eligible_confidence, k=eligible_take
            ).indices.tolist()
            selected_indices = [
                position_to_candidate[int(position)]
                for position in selected_positions_from_eligible
            ]
        else:
            selected_indices = []
        selection_reason = ["not_selected"] * count
        for idx in selected_indices:
            selection_reason[idx] = "bootstrap" if not has_history[idx] else "eligible"

        remaining_budget = int(budget) - len(selected_indices)
        if remaining_budget > 0:
            remaining = [idx for idx in range(count) if idx not in selected_indices]
            if fallback_mode in {"confidence", "frontier"}:
                fallback_confidence = full_confidence.clone()
                if selected_indices:
                    fallback_confidence[
                        masked[torch.tensor(selected_indices, dtype=torch.long, device=x.device)]
                    ] = -torch.inf
                fallback_positions = torch.topk(
                    fallback_confidence, k=remaining_budget
                ).indices.tolist()
                remaining = [
                    position_to_candidate[int(position)] for position in fallback_positions
                ]
            else:
                # Data-driven ablation: preserve stable-but-below-delta candidates,
                # otherwise prefer the candidate least affected by the new condition.
                remaining.sort(
                    key=lambda idx: (
                        0 if bool(stable[idx]) else 1,
                        float(influence[idx].item()),
                        -float(confidence[idx].item()),
                    )
                )
            fallback_indices = remaining[:remaining_budget]
            selected_indices.extend(fallback_indices)
            diagnostics["fallback_used"] = bool(has_history.any())
            if has_history.any():
                diagnostics["fallback_reason"] = (
                    "no_eligible" if not bool(effective_eligible.any()) else "budget_fill"
                )
                diagnostics["forced_commits"] += len(fallback_indices)
            for idx in fallback_indices:
                selection_reason[idx] = f"fallback_{fallback_mode}"

        if not selected_indices:
            raise RuntimeError("candidate-memory selector made no progress")
        selected_positions = masked[
            torch.tensor(selected_indices, dtype=torch.long, device=x.device)
        ]
        transfer_index[batch_idx, selected_positions] = True
        selected_set = set(selected_indices)

        # Persist only positions that remain masked.  Full probabilities are a
        # transient one-step diagnostic buffer and are deliberately never saved.
        for candidate_idx, global_position in enumerate(masked.tolist()):
            if candidate_idx in selected_set:
                continue
            memory_item = {
                "top1_id": current_top1[candidate_idx].to(torch.long).detach().clone(),
                "top1_confidence": confidence[candidate_idx]
                .to(torch.float32)
                .detach()
                .clone(),
                "topk_ids": current_topk_ids[candidate_idx].to(torch.long).detach().clone(),
                "topk_probs": current_topk_probs[candidate_idx].to(torch.float32).detach().clone(),
                "other_mass": current_other_mass[candidate_idx].to(torch.float32).detach().clone(),
            }
            if collect_exact_jsd:
                memory_item["full_probabilities"] = current_probabilities[candidate_idx].to(
                    torch.float32
                ).detach().clone()
            next_memory[batch_idx][global_position] = memory_item

        diagnostics["history_candidates"] += int(has_history.sum().item())
        diagnostics["stable_candidates"] += int(stable.sum().item())
        diagnostics["changed_candidates"] += int(changed.sum().item())
        diagnostics["high_confidence_candidates"] += int(high_confidence.sum().item())
        diagnostics["eligible_candidates"] += int(effective_eligible.sum().item())
        eligible_order = selected_indices + [
            idx for idx in eligible_indices if idx not in selected_indices
        ]
        fallback_order = [idx for idx in confidence_order if idx not in eligible_order]
        if fallback_mode == "impact":
            fallback_order.sort(
                key=lambda idx: (
                    0 if bool(stable[idx]) else 1,
                    float(influence[idx].item()),
                    -float(confidence[idx].item()),
                )
            )
        ordered_indices = eligible_order + fallback_order
        selected_confidence = confidence[
            torch.tensor(selected_indices, dtype=torch.long, device=x.device)
        ]
        diagnostics["candidate_state"].append({
            "masked_positions_global": masked.to(torch.int32).cpu(),
            "masked_positions_local": (masked - block_start).to(torch.int16).cpu(),
            "current_topk_token_ids": current_topk_ids.to(torch.int32).cpu(),
            "current_topk_probabilities": current_topk_probs.to(torch.float32).cpu(),
            "current_other_mass": current_other_mass.to(torch.float32).cpu(),
            "previous_topk_token_ids": previous_topk_ids.to(torch.int32).cpu(),
            "previous_topk_probabilities": previous_topk_probs.cpu(),
            "previous_other_mass": previous_other_mass.cpu(),
            "current_probabilities_on_previous_topk": current_probs_on_previous_topk.cpu(),
            "current_other_mass_on_previous_partition": current_other_on_previous_partition.cpu(),
            "current_top1_token_ids": current_top1.to(torch.int32).cpu(),
            "previous_top1_token_ids": previous_top1.to(torch.int32).cpu(),
            # Keep FP64 so independent validation can reproduce torch.topk's
            # tie/near-tie ordering exactly.
            "top1_confidences": confidence.to(torch.float64).cpu(),
            "previous_top1_confidences": previous_confidence.cpu(),
            "confidence_change": (confidence.to(torch.float32) - previous_confidence).cpu(),
            "top1_margin": top1_margin.to(torch.float32).cpu(),
            "entropy_nats": entropy.to(torch.float32).cpu(),
            "has_history": has_history.cpu(),
            "top1_stable": stable.cpu(),
            "candidate_changed": changed.cpu(),
            "above_confidence_threshold": high_confidence.cpu(),
            "eligible": eligible.cpu(),
            "effective_eligible": effective_eligible.cpu(),
            "in_confidence_frontier": in_frontier.cpu(),
            "baseline_confidence_rank": confidence_rank.cpu(),
            "full_jsd_nats": full_jsd.cpu(),
            "full_jsd_available": full_jsd_available.cpu(),
            "sparse_previous_partition_jsd_nats": sparse_jsd.cpu(),
            "decision_jsd_nats": decision_jsd.cpu(),
            "jsd_approximation_gap": torch.where(
                full_jsd_available,
                full_jsd - sparse_jsd,
                torch.zeros_like(full_jsd),
            ).cpu(),
            "previous_top1_current_probability": previous_top1_current_probability.cpu(),
            "topk_overlap_count": overlap_count.cpu(),
            "topk_jaccard": (
                overlap_count.to(torch.float32)
                / (2 * candidate_topk - overlap_count.to(torch.float32)).clamp(min=1)
            ).cpu(),
            "attention_to_previous_selected": attention_to_previous.cpu(),
            "reverse_attention_from_previous_selected": reverse_attention_from_previous.cpu(),
            "directional_attention_asymmetry_to_previous": (
                attention_to_previous - reverse_attention_from_previous
            ).abs().cpu(),
            "attention_arrival": arrival.cpu(),
            "influence_candidate_jsd": influence.cpu(),
            "ordered_positions_global": masked[
                torch.tensor(ordered_indices, dtype=torch.long, device=x.device)
            ].to(torch.int32).cpu(),
            "selected_positions_global": selected_positions.to(torch.int32).cpu(),
            "selected_by_fallback": torch.tensor(
                [selection_reason[idx].startswith("fallback") for idx in range(count)],
                dtype=torch.bool,
            ),
            "selected_confidence_loss_from_baseline": (
                confidence.max() - selected_confidence
            ).to(torch.float32).cpu(),
            "selection_reason": selection_reason,
        })

    diagnostics["runtime_memory_positions"] = sum(len(items) for items in next_memory)
    diagnostics["runtime_memory_topk_elements"] = (
        diagnostics["runtime_memory_positions"] * candidate_topk
    )
    diagnostics["runtime_full_probability_elements"] = sum(
        int(item.get("full_probabilities", torch.empty(0)).numel())
        for items in next_memory
        for item in items.values()
    )
    return x0, transfer_index, diagnostics, next_memory


@torch.no_grad()
def generate_candidate_memory(
    model,
    prompt,
    candidate_topk=8,
    confidence_threshold=0.0,
    fallback_mode="confidence",
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    eos_id=126081,
    early_stop=False,
    collect_step_diagnostics=False,
    collect_exact_jsd=False,
):
    """Cross-step candidate validation with a fixed baseline decode budget."""
    x = torch.full(
        (1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=model.device
    )
    x[:, :prompt.shape[1]] = prompt.clone()
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    block_steps = steps // num_blocks
    nfe = 0
    if x.is_cuda:
        torch.cuda.reset_peak_memory_stats(x.device)
    summary = {
        "decoder": "candidate_memory_stability_v2",
        "candidate_topk": int(candidate_topk),
        "confidence_threshold": float(confidence_threshold),
        "fallback_mode": fallback_mode,
        "configured_steps": int(steps),
        "gen_length": int(gen_length),
        "block_length": int(block_length),
        "jsd_definition": "previous_topk_fixed_partition_plus_other",
        "exact_jsd_diagnostics": bool(collect_exact_jsd),
        "jsd_log_base": "natural",
        "history_candidates": 0,
        "stable_candidates": 0,
        "changed_candidates": 0,
        "high_confidence_candidates": 0,
        "eligible_candidates": 0,
        "forced_commits": 0,
        "fallback_steps": 0,
        "full_jsd_sum": 0.0,
        "full_jsd_max": 0.0,
        "full_jsd_observations": 0,
        "sparse_jsd_sum": 0.0,
        "sparse_jsd_max": 0.0,
        "arrival_sum": 0.0,
        "arrival_max": 0.0,
        "influence_sum": 0.0,
        "influence_max": 0.0,
        "history_observations": 0,
        "peak_runtime_memory_positions": 0,
        "peak_runtime_full_probability_elements": 0,
        "peak_cuda_allocated_bytes": 0,
        "peak_cuda_reserved_bytes": 0,
    }
    step_records = []

    for block_idx in range(num_blocks):
        block_start = prompt.shape[1] + block_idx * block_length
        block_end = block_start + block_length
        initial_mask = x[:, block_start:block_end] == mask_id
        transfer_schedule = get_num_transfer_tokens(initial_mask, block_steps)
        previous_memory = None
        previous_selected = None
        step_idx = 0
        while (x[:, block_start:block_end] == mask_id).any():
            if step_idx >= block_steps:
                raise RuntimeError("candidate-memory decoder exceeded the baseline block budget")
            budget = int(transfer_schedule[0, step_idx].item())
            step_start_time = time.perf_counter()
            nfe += 1
            mask_index = x == mask_id
            mask_index[:, block_end:] = False
            logits, directional_attention, dependency = _forward_with_block_attention(
                model, x, block_start, block_end
            )
            x0, transfer_index, step_diagnostics, next_memory = select_candidate_memory_tokens(
                logits=logits,
                temperature=temperature,
                remasking=remasking,
                mask_index=mask_index,
                x=x,
                budget=budget,
                directional_attention=directional_attention,
                block_start=block_start,
                candidate_topk=candidate_topk,
                confidence_threshold=confidence_threshold,
                fallback_mode=fallback_mode,
                collect_exact_jsd=collect_exact_jsd,
                previous_memory=previous_memory,
                previous_selected=previous_selected,
            )
            selected_positions = torch.where(transfer_index[0])[0]
            if int(selected_positions.numel()) != budget:
                raise RuntimeError(
                    f"candidate-memory selector changed baseline budget: {selected_positions.numel()} != {budget}"
                )

            candidate_state = step_diagnostics["candidate_state"][0]
            history_mask = candidate_state["has_history"]
            history_count = int(history_mask.sum().item())
            full_available = candidate_state["full_jsd_available"]
            full_jsd_values = candidate_state["full_jsd_nats"][full_available]
            sparse_jsd_values = candidate_state[
                "sparse_previous_partition_jsd_nats"
            ][history_mask]
            arrival_values = candidate_state["attention_arrival"][history_mask]
            influence_values = candidate_state["influence_candidate_jsd"][history_mask]
            summary["history_candidates"] += step_diagnostics["history_candidates"]
            summary["stable_candidates"] += step_diagnostics["stable_candidates"]
            summary["changed_candidates"] += step_diagnostics["changed_candidates"]
            summary["high_confidence_candidates"] += step_diagnostics[
                "high_confidence_candidates"
            ]
            summary["eligible_candidates"] += step_diagnostics["eligible_candidates"]
            summary["forced_commits"] += step_diagnostics["forced_commits"]
            summary["fallback_steps"] += int(step_diagnostics["fallback_used"])
            summary["history_observations"] += history_count
            if history_count:
                summary["sparse_jsd_sum"] += float(sparse_jsd_values.sum().item())
                summary["sparse_jsd_max"] = max(
                    summary["sparse_jsd_max"], float(sparse_jsd_values.max().item())
                )
                summary["arrival_sum"] += float(arrival_values.sum().item())
                summary["arrival_max"] = max(
                    summary["arrival_max"], float(arrival_values.max().item())
                )
                summary["influence_sum"] += float(influence_values.sum().item())
                summary["influence_max"] = max(
                    summary["influence_max"], float(influence_values.max().item())
                )
            if full_jsd_values.numel():
                summary["full_jsd_sum"] += float(full_jsd_values.sum().item())
                summary["full_jsd_max"] = max(
                    summary["full_jsd_max"], float(full_jsd_values.max().item())
                )
                summary["full_jsd_observations"] += int(full_jsd_values.numel())
            summary["peak_runtime_memory_positions"] = max(
                summary["peak_runtime_memory_positions"],
                step_diagnostics["runtime_memory_positions"],
            )
            summary["peak_runtime_full_probability_elements"] = max(
                summary["peak_runtime_full_probability_elements"],
                step_diagnostics["runtime_full_probability_elements"],
            )
            if x.is_cuda:
                cuda_allocated = int(torch.cuda.max_memory_allocated(x.device))
                cuda_reserved = int(torch.cuda.max_memory_reserved(x.device))
            else:
                cuda_allocated = 0
                cuda_reserved = 0
            summary["peak_cuda_allocated_bytes"] = max(
                summary["peak_cuda_allocated_bytes"], cuda_allocated
            )
            summary["peak_cuda_reserved_bytes"] = max(
                summary["peak_cuda_reserved_bytes"], cuda_reserved
            )

            if collect_step_diagnostics:
                asymmetry = (
                    directional_attention - directional_attention.transpose(-2, -1)
                ).abs()
                previous_selected_cpu = (
                    previous_selected.to(torch.int32).cpu()
                    if previous_selected is not None
                    else torch.empty(0, dtype=torch.int32)
                )
                step_records.append({
                    "block_index": block_idx,
                    "step_index_in_block": step_idx,
                    "global_nfe": nfe,
                    "block_start": block_start,
                    "block_end": block_end,
                    "budget": budget,
                    "mask_count_before": int(mask_index[0].sum().item()),
                    "mask_count_after": int(
                        mask_index[0].sum().item() - transfer_index[0].sum().item()
                    ),
                    "input_block_token_ids": x[0, block_start:block_end]
                    .to(torch.int32)
                    .cpu(),
                    "previous_selected_positions_global": previous_selected_cpu,
                    "previous_selected_token_ids": (
                        x[0, previous_selected].to(torch.int32).cpu()
                        if previous_selected is not None
                        else torch.empty(0, dtype=torch.int32)
                    ),
                    "selected_positions_global": selected_positions.to(torch.int32).cpu(),
                    "selected_positions_local": (selected_positions - block_start)
                    .to(torch.int16)
                    .cpu(),
                    "selected_token_ids": x0[0, selected_positions].to(torch.int32).cpu(),
                    "directional_attention": directional_attention[0].to(torch.float16).cpu(),
                    "symmetric_dependency": dependency[0].to(torch.float16).cpu(),
                    "attention_asymmetry_mean": float(asymmetry.mean().item()),
                    "attention_asymmetry_max": float(asymmetry.max().item()),
                    "history_candidates": step_diagnostics["history_candidates"],
                    "stable_candidates": step_diagnostics["stable_candidates"],
                    "changed_candidates": step_diagnostics["changed_candidates"],
                    "high_confidence_candidates": step_diagnostics[
                        "high_confidence_candidates"
                    ],
                    "eligible_candidates": step_diagnostics["eligible_candidates"],
                    "forced_commits": step_diagnostics["forced_commits"],
                    "fallback_used": step_diagnostics["fallback_used"],
                    "fallback_reason": step_diagnostics["fallback_reason"],
                    "runtime_memory_positions": step_diagnostics[
                        "runtime_memory_positions"
                    ],
                    "runtime_memory_topk_elements": step_diagnostics[
                        "runtime_memory_topk_elements"
                    ],
                    "runtime_full_probability_elements": step_diagnostics[
                        "runtime_full_probability_elements"
                    ],
                    "cuda_memory_allocated_bytes": cuda_allocated,
                    "cuda_memory_reserved_bytes": cuda_reserved,
                    "step_wall_time_seconds": time.perf_counter() - step_start_time,
                    **candidate_state,
                })

            previous_memory = next_memory
            previous_selected = selected_positions
            x[transfer_index] = x0[transfer_index]
            step_idx += 1

        if early_stop and (x[:, block_start:block_end] == eos_id).any():
            x[:, block_end:] = eos_id
            break

    observations = summary["history_observations"]
    if observations:
        summary["sparse_jsd_mean"] = summary["sparse_jsd_sum"] / observations
        summary["arrival_mean"] = summary["arrival_sum"] / observations
        summary["influence_mean"] = summary["influence_sum"] / observations
    if summary["full_jsd_observations"]:
        summary["full_jsd_mean"] = (
            summary["full_jsd_sum"] / summary["full_jsd_observations"]
        )
    if collect_step_diagnostics:
        summary["_step_records"] = step_records
    return x, nfe, summary


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0., remasking='low_confidence',
    mask_id=126336, eos_id=126081, early_stop=False, threshold=None, relaxed_threshold=None, radius=4):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        eos_id: The toke id of <|endoftext|> is 126081.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = block_start + block_length

        block_mask_index = (x[:, block_start: block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            if (x[:, block_start: block_end] == mask_id).sum() == 0:
                if early_stop and (x[:, block_start:block_end] == eos_id).any():
                    x[:, block_end: ] = eos_id
                    return x, nfe
                break
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, block_end:] = 0
            x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None,
                                                    threshold, relaxed_threshold, radius)
            x[transfer_index] = x0[transfer_index]
            i += 1

    return x, nfe


def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens,
    threshold=None, relaxed_threshold=None, radius=4):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            mask_positions = torch.where(mask_index[j])[0]

            neighbor_positions = set()
            use_localleap = False

            if relaxed_threshold is not None:
                anchor_mask = confidence[j][mask_positions] >= threshold
                anchor_count = anchor_mask.sum().item()

                if anchor_count >= 1:
                    use_localleap = True
                    anchor_positions = mask_positions[anchor_mask]
                    for pos in anchor_positions:
                        pos_val = pos.item()
                        # Add all positions of the anchor's neighbors.
                        for neignbor_pos in range(max(0, pos_val - radius), min(confidence.shape[1], pos_val + radius + 1)):
                            neighbor_positions.add(neignbor_pos)

            for k in range(1, num_transfer_tokens[j]):
                pos = select_index[k].item()
                if use_localleap:
                    effective_threshold = relaxed_threshold if pos in neighbor_positions else threshold
                else:
                    effective_threshold = threshold

                if confidence[j, select_index[k]] < effective_threshold:
                    transfer_index[j, select_index[k]] = False

    return x0, transfer_index


def main():
    device = 'cuda'

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    length = 256
    block_length = 32
    steps = int(length / block_length)

    import prompts as _prompts
    prompt_gms8k_5shot_1 = _prompts.prompt_gms8k_5shot_1
    prompt = prompt_gms8k_5shot_1

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, steps=steps, gen_length=length, block_length=block_length, temperature=0., remasking='low_confidence',
                    threshold=0.9, relaxed_threshold=0.75, radius=4)
    answer = tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    print('nfe:', out[1], 'answer length:', (torch.tensor(tokenizer(answer)["input_ids"]) != 126081).sum())
    print('answer:', tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()

# Alias used by eval_llada.py (missing in upstream)
generate_localleap = generate
