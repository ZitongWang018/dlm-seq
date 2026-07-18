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
    previous_topk_ids=None,
    temporal_mode="top1",
    temporal_topk=4,
    prune_stable_conflicts=False,
    fill_budget=False,
    previous_response_credit=None,
):
    """Lexicographic maturity/confidence ordering plus greedy dependency exclusion."""
    if temporal_mode not in {
        "top1",
        "topk_overlap",
        "response_credit",
        "revision_margin",
    }:
        raise ValueError(f"unsupported temporal mode: {temporal_mode}")
    if temporal_topk < 1 or temporal_topk > logits.shape[-1]:
        raise ValueError("temporal_topk must be between 1 and vocabulary size")
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    probabilities = F.softmax(logits.to(torch.float64), dim=-1)
    if remasking == 'low_confidence':
        x0_p = torch.gather(probabilities, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == 'random':
        x0_p = torch.rand(x0.shape, device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    current_topk_full = torch.full(
        (*x.shape, temporal_topk), -1, dtype=torch.long, device=x.device
    )
    current_response_credit_full = torch.zeros_like(x, dtype=torch.int16)
    diagnostics = {
        "unstable_candidates": 0,
        "changed_candidates": 0,
        "candidate_continuity_candidates": 0,
        "intermediate_candidates": 0,
        "response_validations": 0,
        "response_invalidations": 0,
        "response_credit_max": 0,
        "revision_margin_max": 0.0,
        "revision_margin_sum": 0.0,
        "revision_margin_candidates": 0,
        "strongly_dependent_candidates": 0,
        "rejected_pairs": 0,
        "stable_conflicts_pruned": 0,
        "forced_budget_fills": 0,
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
        overlap_count = torch.zeros(masked.numel(), dtype=torch.int16, device=x.device)
        temporal_tier = torch.full(
            (masked.numel(),), 2, dtype=torch.int8, device=x.device
        )
        current_topk_ids = torch.topk(
            probabilities[batch_idx, masked], k=temporal_topk, dim=-1
        ).indices
        current_topk_full[batch_idx, masked] = current_topk_ids
        max_dependency = torch.zeros(masked.numel(), dtype=torch.float32, device=x.device)
        response_credit = torch.zeros(masked.numel(), dtype=torch.int16, device=x.device)
        revision_margin = torch.zeros(
            masked.numel(), dtype=torch.float32, device=x.device
        )
        history_available = previous_top1 is not None and previous_selected is not None and previous_selected.numel() > 0
        if history_available:
            masked_local = masked - block_start
            selected_local = previous_selected - block_start
            max_dependency = dependency[batch_idx].index_select(0, masked_local).index_select(1, selected_local).max(dim=1).values
            changed = x0[batch_idx, masked] != previous_top1[batch_idx, masked]
            strong_dependency = max_dependency > dependency_threshold
            if previous_response_credit is not None:
                previous_credit = previous_response_credit[batch_idx, masked]
            else:
                previous_credit = torch.zeros_like(response_credit)
            response_validated = strong_dependency & ~changed
            response_invalidated = strong_dependency & changed
            previous_ids = previous_top1[batch_idx, masked]
            valid_previous_ids = (previous_ids >= 0) & (
                previous_ids < probabilities.shape[-1]
            )
            safe_previous_ids = previous_ids.clamp(
                min=0, max=probabilities.shape[-1] - 1
            )
            previous_candidate_probability = torch.gather(
                probabilities[batch_idx, masked],
                dim=-1,
                index=safe_previous_ids.unsqueeze(-1),
            ).squeeze(-1)
            current_candidate_probability = confidence[batch_idx, masked]
            raw_revision_margin = (
                torch.log(current_candidate_probability.clamp_min(1e-30))
                - torch.log(previous_candidate_probability.clamp_min(1e-30))
            ).to(torch.float32)
            revision_candidates = strong_dependency & changed & valid_previous_ids
            revision_margin = torch.where(
                revision_candidates,
                raw_revision_margin,
                torch.zeros_like(raw_revision_margin),
            )
            diagnostics["revision_margin_candidates"] += int(
                revision_candidates.sum().item()
            )
            if revision_candidates.any():
                selected_revision_margins = revision_margin[revision_candidates]
                diagnostics["revision_margin_max"] = max(
                    diagnostics["revision_margin_max"],
                    float(selected_revision_margins.max().item()),
                )
                diagnostics["revision_margin_sum"] += float(
                    selected_revision_margins.sum().item()
                )
            incremented_credit = torch.clamp(
                previous_credit.to(torch.int32) + 1,
                max=torch.iinfo(torch.int16).max,
            ).to(torch.int16)
            response_credit = torch.where(
                response_validated,
                incremented_credit,
                torch.where(response_invalidated, torch.zeros_like(previous_credit), previous_credit),
            )
            diagnostics["response_validations"] += int(response_validated.sum().item())
            diagnostics["response_invalidations"] += int(response_invalidated.sum().item())
            diagnostics["response_credit_max"] = max(
                diagnostics["response_credit_max"], int(response_credit.max().item())
            )
            if temporal_mode == "topk_overlap":
                if previous_topk_ids is None:
                    raise ValueError("topk_overlap mode requires previous_topk_ids")
                previous_ids = previous_topk_ids[batch_idx, masked]
                valid_previous = previous_ids >= 0
                overlap_count = (
                    (current_topk_ids.unsqueeze(-1) == previous_ids.unsqueeze(-2))
                    & valid_previous.unsqueeze(-2)
                ).any(dim=-1).sum(dim=-1).to(torch.int16)
                has_continuity = overlap_count > 0
                temporal_tier = torch.where(
                    ~strong_dependency | ~changed,
                    torch.full_like(temporal_tier, 2),
                    torch.where(
                        has_continuity,
                        torch.full_like(temporal_tier, 1),
                        torch.zeros_like(temporal_tier),
                    ),
                )
                maturity = temporal_tier == 2
                diagnostics["candidate_continuity_candidates"] += int(
                    has_continuity.sum().item()
                )
                diagnostics["intermediate_candidates"] += int(
                    (temporal_tier == 1).sum().item()
                )
            else:
                maturity = ~(strong_dependency & changed)
                temporal_tier = torch.where(
                    maturity,
                    torch.full_like(temporal_tier, 2),
                    torch.zeros_like(temporal_tier),
                )
            diagnostics["unstable_candidates"] += int((~maturity).sum().item())
            diagnostics["changed_candidates"] += int(changed.sum().item())
            diagnostics["strongly_dependent_candidates"] += int(strong_dependency.sum().item())
        current_response_credit_full[batch_idx, masked] = response_credit

        mature_positions = masked[maturity]
        immature_positions = masked[~maturity]
        if mature_positions.numel() == 0:
            # The parent falls back to confidence when every candidate changed.
            # Revision-margin mode instead asks which change was most decisive
            # under the newly committed condition; no additional threshold is
            # introduced and confidence remains the deterministic tie breaker.
            if temporal_mode == "revision_margin" and history_available:
                position_to_masked_index = {
                    int(position): index
                    for index, position in enumerate(masked.tolist())
                }
                unstable = masked.tolist()
                unstable.sort(
                    key=lambda position: (
                        -float(
                            revision_margin[
                                position_to_masked_index[int(position)]
                            ].item()
                        ),
                        -float(confidence[batch_idx, position].item()),
                    )
                )
                ordered = torch.tensor(unstable, dtype=torch.long, device=x.device)
            else:
                ordered = masked[
                    torch.topk(
                        confidence[batch_idx, masked], k=masked.numel()
                    ).indices
                ]
            diagnostics["all_immature_fallback"] = True
        else:
            # Mature candidates must remain bit-for-bit ordered by the parent
            # method's confidence rule. Top-K history is allowed to refine only
            # the unstable tail that the parent would already rank second.
            if temporal_mode == "response_credit" and history_available:
                position_to_masked_index = {
                    int(position): index
                    for index, position in enumerate(masked.tolist())
                }
                mature = mature_positions.tolist()
                mature.sort(
                    key=lambda position: (
                        -int(
                            response_credit[
                                position_to_masked_index[int(position)]
                            ].item()
                        ),
                        -float(confidence[batch_idx, position].item()),
                    )
                )
                mature_order = torch.tensor(mature, dtype=torch.long, device=x.device)
            else:
                mature_order = mature_positions[
                    torch.topk(
                        confidence[batch_idx, mature_positions],
                        k=mature_positions.numel(),
                    ).indices
                ]
            if immature_positions.numel() > 0:
                if temporal_mode in {"topk_overlap", "revision_margin"} and history_available:
                    position_to_masked_index = {
                        int(position): index
                        for index, position in enumerate(masked.tolist())
                    }
                    unstable = immature_positions.tolist()
                    if temporal_mode == "topk_overlap":
                        unstable.sort(
                            key=lambda position: (
                                -int(
                                    overlap_count[
                                        position_to_masked_index[int(position)]
                                    ].item()
                                ),
                                -float(confidence[batch_idx, position].item()),
                            )
                        )
                    else:
                        unstable.sort(
                            key=lambda position: (
                                -float(
                                    revision_margin[
                                        position_to_masked_index[int(position)]
                                    ].item()
                                ),
                                -float(confidence[batch_idx, position].item()),
                            )
                        )
                    immature_order = torch.tensor(
                        unstable, dtype=torch.long, device=x.device
                    )
                else:
                    immature_order = immature_positions[
                        torch.topk(
                            confidence[batch_idx, immature_positions],
                            k=immature_positions.numel(),
                        ).indices
                    ]
                ordered = torch.cat((mature_order, immature_order))
            else:
                ordered = mature_order

        selected = []
        rejected = []
        forced = []
        changed_by_position = {
            int(position): bool(changed[index].item())
            for index, position in enumerate(masked.tolist())
        }
        tier_by_position = {
            int(position): int(temporal_tier[index].item())
            for index, position in enumerate(masked.tolist())
        }
        overlap_by_position = {
            int(position): int(overlap_count[index].item())
            for index, position in enumerate(masked.tolist())
        }
        for position in ordered.tolist():
            if len(selected) >= int(budget):
                break
            if selected:
                local_position = position - block_start
                selected_local = torch.tensor([item - block_start for item in selected], device=x.device)
                pair_dependency = dependency[batch_idx, local_position, selected_local]
                strong_pair_count = int((pair_dependency > dependency_threshold).sum().item())
                if strong_pair_count:
                    stable_conflict = (
                        history_available
                        and not changed_by_position[position]
                        and all(not changed_by_position[item] for item in selected)
                    )
                    if prune_stable_conflicts and stable_conflict:
                        diagnostics["stable_conflicts_pruned"] += strong_pair_count
                    else:
                        diagnostics["rejected_pairs"] += 1
                        rejected.append(position)
                        continue
            selected.append(position)
        target_count = min(int(budget), masked.numel())
        if fill_budget and len(selected) < target_count:
            def fill_priority(position):
                local_position = position - block_start
                if selected:
                    selected_local = torch.tensor(
                        [item - block_start for item in selected], device=x.device
                    )
                    dependency_score = float(
                        dependency[batch_idx, local_position, selected_local].max().item()
                    )
                else:
                    dependency_score = 0.0
                changed_priority = int(
                    not history_available or changed_by_position[position]
                )
                return (
                    -tier_by_position[position],
                    -overlap_by_position[position],
                    changed_priority,
                    dependency_score,
                    -float(confidence[batch_idx, position].item()),
                )

            for position in sorted(rejected, key=fill_priority):
                if len(selected) >= target_count:
                    break
                selected.append(position)
                forced.append(position)
            diagnostics["forced_budget_fills"] += len(forced)
        if len(selected) < target_count:
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
            "current_topk_token_ids": current_topk_ids.to(torch.int32).cpu(),
            "previous_topk_token_ids": (
                previous_topk_ids[batch_idx, masked].to(torch.int32).cpu()
                if previous_topk_ids is not None
                else torch.full(
                    (masked.numel(), temporal_topk), -1, dtype=torch.int32
                )
            ),
            "topk_overlap_count": overlap_count.cpu(),
            "temporal_tier": temporal_tier.cpu(),
            "response_credit": response_credit.cpu(),
            "revision_margin": revision_margin.cpu(),
            "max_dependency_to_previous": max_dependency.to(torch.float32).cpu(),
            "maturity": maturity.cpu(),
            "ordered_positions_global": ordered.to(torch.int32).cpu(),
            "selected_positions_global": torch.tensor(selected, dtype=torch.int32),
            "rejected_positions_global": torch.tensor(rejected, dtype=torch.int32),
            "forced_fill_positions_global": torch.tensor(forced, dtype=torch.int32),
        })

    diagnostics["_current_response_credit_full"] = current_response_credit_full
    return x0, transfer_index, diagnostics, current_topk_full


def score_committed_tokens(candidate_state):
    """Return log-confidence sum and count for positions committed this step."""
    candidate_positions = candidate_state["masked_positions_global"].to(torch.long)
    selected_positions = candidate_state["selected_positions_global"].to(torch.long)
    position_to_candidate = {
        int(position): index
        for index, position in enumerate(candidate_positions.tolist())
    }
    selected_confidences = torch.stack(
        [
            candidate_state["top1_confidences"][
                position_to_candidate[int(position)]
            ]
            for position in selected_positions.tolist()
        ]
    )
    return (
        float(selected_confidences.clamp_min(1e-12).log().sum().item()),
        int(selected_confidences.numel()),
    )


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
    dependency_mode="symmetric",
    temporal_mode="top1",
    temporal_topk=4,
    prune_stable_conflicts=False,
    fill_budget=False,
    collect_position_risk=False,
):
    """Baseline LLaDA decoding with the proposed attention/stability selection layered on top."""
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    block_steps = steps // num_blocks
    nfe = 0
    if dependency_mode not in {"symmetric", "directed_read"}:
        raise ValueError(f"unsupported dependency mode: {dependency_mode}")
    if temporal_mode not in {
        "top1",
        "topk_overlap",
        "response_credit",
        "revision_margin",
    }:
        raise ValueError(f"unsupported temporal mode: {temporal_mode}")
    summary = {
        "decoder": (
            "attention_stability_v1"
            if dependency_mode == "symmetric"
            and temporal_mode == "top1"
            and not prune_stable_conflicts
            and not fill_budget
            else "attention_stability_v2"
        ),
        "dependency_threshold": float(dependency_threshold),
        "dependency_mode": dependency_mode,
        "temporal_mode": temporal_mode,
        "temporal_topk": int(temporal_topk),
        "prune_stable_conflicts": bool(prune_stable_conflicts),
        "fill_budget": bool(fill_budget),
        "configured_steps": int(steps),
        "gen_length": int(gen_length),
        "block_length": int(block_length),
        "unstable_candidates": 0,
        "changed_candidates": 0,
        "candidate_continuity_candidates": 0,
        "intermediate_candidates": 0,
        "response_validations": 0,
        "response_invalidations": 0,
        "response_credit_max": 0,
        "revision_margin_max": 0.0,
        "revision_margin_sum": 0.0,
        "revision_margin_candidates": 0,
        "strongly_dependent_candidates": 0,
        "rejected_pairs": 0,
        "stable_conflicts_pruned": 0,
        "forced_budget_fills": 0,
        "underfilled_steps": 0,
        "all_immature_fallback_steps": 0,
        "dependency_max": 0.0,
        "dependency_mean_sum": 0.0,
        "dependency_observations": 0,
        "attention_asymmetry_max": 0.0,
        "attention_asymmetry_mean_sum": 0.0,
        "commit_logprob_sum": 0.0,
        "commit_token_count": 0,
    }
    step_records = []
    position_risk = None
    if collect_position_risk:
        sequence_length = x.shape[1]
        position_risk = {
            "response_invalidations": torch.zeros(sequence_length, dtype=torch.int32),
            "response_validations": torch.zeros(sequence_length, dtype=torch.int32),
            "commit_confidence": torch.full(
                (sequence_length,), float("nan"), dtype=torch.float32
            ),
            "commit_maturity": torch.ones(sequence_length, dtype=torch.bool),
            "commit_forced": torch.zeros(sequence_length, dtype=torch.bool),
            "commit_revision_margin": torch.zeros(
                sequence_length, dtype=torch.float32
            ),
            "final_directional_attention": [],
        }

    for block_idx in range(num_blocks):
        block_start = prompt.shape[1] + block_idx * block_length
        block_end = block_start + block_length
        initial_mask = x[:, block_start:block_end] == mask_id
        transfer_schedule = get_num_transfer_tokens(initial_mask, block_steps)
        step_idx = 0
        previous_top1 = None
        previous_topk_ids = None
        previous_response_credit = None
        previous_selected = None
        last_directional_attention = None
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
            logits, directional_attention, symmetric_dependency = _forward_with_block_attention(
                model, x, block_start, block_end
            )
            last_directional_attention = directional_attention[0].detach().to(
                torch.float32
            ).cpu()
            dependency = (
                symmetric_dependency
                if dependency_mode == "symmetric"
                else directional_attention
            )
            summary["dependency_max"] = max(summary["dependency_max"], float(dependency.max().item()))
            summary["dependency_mean_sum"] += float(dependency.mean().item())
            summary["dependency_observations"] += 1
            asymmetry = (directional_attention - directional_attention.transpose(-2, -1)).abs()
            summary["attention_asymmetry_max"] = max(
                summary["attention_asymmetry_max"], float(asymmetry.max().item())
            )
            summary["attention_asymmetry_mean_sum"] += float(asymmetry.mean().item())
            x0, transfer_index, step_diagnostics, current_topk_ids = select_attention_stability_tokens(
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
                previous_topk_ids=previous_topk_ids,
                previous_response_credit=previous_response_credit,
                temporal_mode=temporal_mode,
                temporal_topk=temporal_topk,
                prune_stable_conflicts=prune_stable_conflicts,
                fill_budget=fill_budget,
            )
            current_response_credit = step_diagnostics.pop(
                "_current_response_credit_full"
            )
            if not transfer_index.any():
                raise RuntimeError("attention-stability selector made no progress")
            summary["unstable_candidates"] += step_diagnostics["unstable_candidates"]
            summary["changed_candidates"] += step_diagnostics["changed_candidates"]
            summary["candidate_continuity_candidates"] += step_diagnostics[
                "candidate_continuity_candidates"
            ]
            summary["intermediate_candidates"] += step_diagnostics[
                "intermediate_candidates"
            ]
            summary["response_validations"] += step_diagnostics["response_validations"]
            summary["response_invalidations"] += step_diagnostics[
                "response_invalidations"
            ]
            summary["response_credit_max"] = max(
                summary["response_credit_max"], step_diagnostics["response_credit_max"]
            )
            summary["revision_margin_max"] = max(
                summary["revision_margin_max"],
                step_diagnostics["revision_margin_max"],
            )
            summary["revision_margin_sum"] += step_diagnostics[
                "revision_margin_sum"
            ]
            summary["revision_margin_candidates"] += step_diagnostics[
                "revision_margin_candidates"
            ]
            summary["strongly_dependent_candidates"] += step_diagnostics["strongly_dependent_candidates"]
            summary["rejected_pairs"] += step_diagnostics["rejected_pairs"]
            summary["stable_conflicts_pruned"] += step_diagnostics["stable_conflicts_pruned"]
            summary["forced_budget_fills"] += step_diagnostics["forced_budget_fills"]
            summary["underfilled_steps"] += int(step_diagnostics["underfilled"])
            summary["all_immature_fallback_steps"] += int(step_diagnostics["all_immature_fallback"])
            candidate_state = step_diagnostics["candidate_state"][0]
            step_logprob, step_token_count = score_committed_tokens(candidate_state)
            summary["commit_logprob_sum"] += step_logprob
            summary["commit_token_count"] += step_token_count
            if collect_position_risk:
                candidate_positions = candidate_state[
                    "masked_positions_global"
                ].to(torch.long)
                strong_response = (
                    candidate_state["max_dependency_to_previous"]
                    > dependency_threshold
                )
                changed_response = candidate_state["candidate_changed"]
                position_risk["response_invalidations"][candidate_positions] += (
                    strong_response & changed_response
                ).to(torch.int32)
                position_risk["response_validations"][candidate_positions] += (
                    strong_response & ~changed_response
                ).to(torch.int32)

                selected_positions = candidate_state[
                    "selected_positions_global"
                ].to(torch.long)
                forced_positions = set(
                    candidate_state["forced_fill_positions_global"].tolist()
                )
                position_to_candidate = {
                    int(position): index
                    for index, position in enumerate(candidate_positions.tolist())
                }
                for selected_position in selected_positions.tolist():
                    candidate_index = position_to_candidate[int(selected_position)]
                    position_risk["commit_confidence"][selected_position] = (
                        candidate_state["top1_confidences"][candidate_index]
                    )
                    position_risk["commit_maturity"][selected_position] = (
                        candidate_state["maturity"][candidate_index]
                    )
                    position_risk["commit_forced"][selected_position] = (
                        int(selected_position) in forced_positions
                    )
                    position_risk["commit_revision_margin"][selected_position] = (
                        candidate_state["revision_margin"][candidate_index]
                    )
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
                    "symmetric_dependency": symmetric_dependency[0].to(torch.float16).cpu(),
                    "selection_dependency": dependency[0].to(torch.float16).cpu(),
                    "dependency_mode": dependency_mode,
                    "dependency_min": float(dependency.min().item()),
                    "dependency_mean": float(dependency.mean().item()),
                    "dependency_max": float(dependency.max().item()),
                    "attention_asymmetry_mean": float(asymmetry.mean().item()),
                    "attention_asymmetry_max": float(asymmetry.max().item()),
                    "unstable_candidates": step_diagnostics["unstable_candidates"],
                    "changed_candidates": step_diagnostics["changed_candidates"],
                    "strongly_dependent_candidates": step_diagnostics["strongly_dependent_candidates"],
                    "response_validations": step_diagnostics["response_validations"],
                    "response_invalidations": step_diagnostics["response_invalidations"],
                    "response_credit_max": step_diagnostics["response_credit_max"],
                    "revision_margin_max": step_diagnostics["revision_margin_max"],
                    "revision_margin_sum": step_diagnostics["revision_margin_sum"],
                    "revision_margin_candidates": step_diagnostics[
                        "revision_margin_candidates"
                    ],
                    "rejected_pairs": step_diagnostics["rejected_pairs"],
                    "stable_conflicts_pruned": step_diagnostics["stable_conflicts_pruned"],
                    "forced_budget_fills": step_diagnostics["forced_budget_fills"],
                    "underfilled": step_diagnostics["underfilled"],
                    "all_immature_fallback": step_diagnostics["all_immature_fallback"],
                    **candidate_state,
                })
            previous_top1 = x0.detach().clone()
            previous_topk_ids = current_topk_ids.detach().clone()
            previous_response_credit = current_response_credit.detach().clone()
            previous_selected = torch.where(transfer_index[0])[0]
            x[transfer_index] = x0[transfer_index]
            step_idx += 1

        if collect_position_risk:
            if last_directional_attention is None:
                raise RuntimeError("missing final directional attention for block")
            position_risk["final_directional_attention"].append(
                last_directional_attention
            )

        if early_stop and (x[:, block_start:block_end] == eos_id).any():
            x[:, block_end:] = eos_id
            break

    if summary["dependency_observations"]:
        summary["dependency_mean"] = summary.pop("dependency_mean_sum") / summary["dependency_observations"]
        summary["attention_asymmetry_mean"] = (
            summary.pop("attention_asymmetry_mean_sum") / summary["dependency_observations"]
        )
    if summary["revision_margin_candidates"]:
        summary["revision_margin_mean"] = (
            summary["revision_margin_sum"]
            / summary["revision_margin_candidates"]
        )
    else:
        summary["revision_margin_mean"] = 0.0
    if summary["commit_token_count"]:
        summary["commit_logprob_mean"] = (
            summary["commit_logprob_sum"] / summary["commit_token_count"]
        )
    else:
        summary["commit_logprob_mean"] = float("-inf")
    if collect_step_diagnostics:
        summary["_step_records"] = step_records
    if collect_position_risk:
        position_risk["final_directional_attention"] = torch.stack(
            position_risk["final_directional_attention"], dim=0
        )
        summary["_position_risk_state"] = position_risk
    return x, nfe, summary


def select_likelihood_trajectory(
    candidate_summaries,
    block_length=None,
    selection_mode="mean",
):
    """Select the path with the best mean committed-token log confidence.

    Ties preserve the fixed-budget fast parent, making the selection deterministic
    and avoiding an extra threshold or task-specific preference.
    """
    ordered_names = ("fast", "accuracy")
    missing = [name for name in ordered_names if name not in candidate_summaries]
    if missing:
        raise ValueError(f"missing trajectory summaries: {missing}")
    if selection_mode == "mean":
        return max(
            ordered_names,
            key=lambda name: float(
                candidate_summaries[name]["commit_logprob_mean"]
            ),
        )
    if selection_mode == "block_evidence":
        if block_length is None or int(block_length) <= 0:
            raise ValueError("block_evidence requires a positive block_length")
        fast_score = float(candidate_summaries["fast"]["commit_logprob_mean"])
        accuracy_score = float(
            candidate_summaries["accuracy"]["commit_logprob_mean"]
        )
        # Require at least one nat of cumulative evidence per generation block
        # before abandoning the parallel parent.  This uses the existing block
        # structure rather than a fitted margin hyperparameter.
        evidence_per_block = (accuracy_score - fast_score) * int(block_length)
        return "accuracy" if evidence_per_block > 1.0 else "fast"
    raise ValueError(f"unsupported trajectory selection mode: {selection_mode}")


@torch.no_grad()
def generate_trajectory_likelihood_selection(
    model,
    prompt,
    dependency_threshold,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    eos_id=126081,
    early_stop=False,
    collect_step_diagnostics=False,
    dependency_mode="symmetric",
    temporal_mode="top1",
    temporal_topk=4,
    selection_mode="mean",
):
    """Generate fast and accuracy-first parents, then select by path evidence.

    Horizontal evidence differs through their treatment of strong within-step
    conflicts.  Vertical evidence is the mean log confidence of tokens at the
    exact denoising step where each token becomes an explicit condition.
    """
    shared = dict(
        model=model,
        prompt=prompt,
        dependency_threshold=dependency_threshold,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        remasking=remasking,
        mask_id=mask_id,
        eos_id=eos_id,
        early_stop=early_stop,
        collect_step_diagnostics=collect_step_diagnostics,
        dependency_mode=dependency_mode,
        temporal_mode=temporal_mode,
        temporal_topk=temporal_topk,
    )
    fast_x, fast_nfe, fast_summary = generate_attention_stability(
        **shared,
        prune_stable_conflicts=True,
        fill_budget=True,
    )
    accuracy_x, accuracy_nfe, accuracy_summary = generate_attention_stability(
        **shared,
        prune_stable_conflicts=False,
        fill_budget=False,
    )
    candidate_summaries = {
        "fast": fast_summary,
        "accuracy": accuracy_summary,
    }
    selected_name = select_likelihood_trajectory(
        candidate_summaries,
        block_length=block_length,
        selection_mode=selection_mode,
    )
    candidate_ids = {"fast": fast_x, "accuracy": accuracy_x}
    selected_summary = candidate_summaries[selected_name]
    selected_steps = selected_summary.pop("_step_records", None)
    fast_summary.pop("_step_records", None)
    accuracy_summary.pop("_step_records", None)
    summary = {
        "decoder": "trajectory_likelihood_selection_v1",
        "selection_rule": (
            "one_nat_per_block_evidence"
            if selection_mode == "block_evidence"
            else "max_mean_committed_token_log_confidence"
        ),
        "selection_mode": selection_mode,
        "block_evidence_margin": (
            float(
                candidate_summaries["accuracy"]["commit_logprob_mean"]
                - candidate_summaries["fast"]["commit_logprob_mean"]
            )
            * int(block_length)
        ),
        "selected_name": selected_name,
        "selected_score": float(
            candidate_summaries[selected_name]["commit_logprob_mean"]
        ),
        "candidate_scores": {
            name: float(value["commit_logprob_mean"])
            for name, value in candidate_summaries.items()
        },
        "candidate_nfe": {"fast": int(fast_nfe), "accuracy": int(accuracy_nfe)},
        "candidate_summaries": candidate_summaries,
        "_trajectory_candidate_token_ids": candidate_ids,
    }
    if selected_steps is not None:
        summary["_step_records"] = selected_steps
    return candidate_ids[selected_name], fast_nfe + accuracy_nfe, summary


@torch.no_grad()
def build_response_refine_mask(
    position_risk,
    prompt_length,
    gen_length,
    block_length,
    dependency_threshold,
    repair_steps,
    risk_gated=False,
    require_commit_risk=False,
):
    """Choose a fixed-budget remasking frontier from horizontal/vertical risk.

    The frontier size is determined by the number of available repair forwards,
    not by another score threshold.  Within each block, lexicographic risk is:
    forced commit, repeated conditioned invalidation, immature commit, dense
    directed-attention incidence, then low commit confidence.
    """
    if gen_length % block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")
    num_blocks = gen_length // block_length
    if repair_steps % num_blocks != 0:
        raise ValueError("repair_steps must be divisible by the number of blocks")
    remask_per_block = repair_steps // num_blocks
    if remask_per_block < 1 or remask_per_block > block_length:
        raise ValueError("repair budget must select between 1 and block_length tokens")

    sequence_length = prompt_length + gen_length
    remask = torch.zeros((1, sequence_length), dtype=torch.bool)
    block_records = []
    selected_invalidation_sum = 0
    selected_forced = 0
    selected_immature = 0
    selected_incident_sum = 0

    for block_idx in range(num_blocks):
        block_start = prompt_length + block_idx * block_length
        directional = position_risk["final_directional_attention"][block_idx]
        strong_edges = directional > dependency_threshold
        incoming_source_count = strong_edges.sum(dim=1).to(torch.int32)
        outgoing_reader_count = strong_edges.sum(dim=0).to(torch.int32)
        incident_count = incoming_source_count + outgoing_reader_count

        def risk_key(local_position):
            global_position = block_start + local_position
            confidence = float(
                position_risk["commit_confidence"][global_position].item()
            )
            if not np.isfinite(confidence):
                confidence = -float("inf")
            return (
                -int(position_risk["commit_forced"][global_position].item()),
                -int(
                    position_risk["response_invalidations"][global_position].item()
                ),
                -int(
                    not position_risk["commit_maturity"][global_position].item()
                ),
                -int(incident_count[local_position].item()),
                confidence,
                local_position,
            )

        candidate_local = list(range(block_length))
        if risk_gated:
            def observed_risk(local_position):
                global_position = block_start + local_position
                invalidated = int(
                    position_risk["response_invalidations"][global_position].item()
                ) > 0
                commit_risk = bool(
                    position_risk["commit_forced"][global_position].item()
                ) or not bool(
                    position_risk["commit_maturity"][global_position].item()
                )
                return (
                    invalidated and commit_risk
                    if require_commit_risk
                    else invalidated or commit_risk
                )

            candidate_local = [
                local_position
                for local_position in candidate_local
                if observed_risk(local_position)
            ]
        selected_local = sorted(candidate_local, key=risk_key)[:remask_per_block]
        selected_global = [block_start + item for item in selected_local]
        remask[0, torch.tensor(selected_global, dtype=torch.long)] = True
        selected_records = []
        for local_position, global_position in zip(
            selected_local, selected_global
        ):
            invalidations = int(
                position_risk["response_invalidations"][global_position].item()
            )
            forced = bool(position_risk["commit_forced"][global_position].item())
            immature = not bool(
                position_risk["commit_maturity"][global_position].item()
            )
            incident = int(incident_count[local_position].item())
            selected_invalidation_sum += invalidations
            selected_forced += int(forced)
            selected_immature += int(immature)
            selected_incident_sum += incident
            selected_records.append(
                {
                    "global_position": int(global_position),
                    "local_position": int(local_position),
                    "response_invalidations": invalidations,
                    "response_validations": int(
                        position_risk["response_validations"][global_position].item()
                    ),
                    "commit_forced": forced,
                    "commit_maturity": not immature,
                    "commit_confidence": float(
                        position_risk["commit_confidence"][global_position].item()
                    ),
                    "commit_revision_margin": float(
                        position_risk["commit_revision_margin"][global_position].item()
                    ),
                    "incoming_source_count": int(
                        incoming_source_count[local_position].item()
                    ),
                    "outgoing_reader_count": int(
                        outgoing_reader_count[local_position].item()
                    ),
                    "incident_count": incident,
                }
            )
        block_records.append(
            {
                "block_index": int(block_idx),
                "selected_positions": selected_records,
                "strong_directed_edge_count": int(strong_edges.sum().item()),
            }
        )

    selected_count = int(remask.sum().item())
    summary = {
        "frontier_rule": (
            "observed_risk_then_forced_invalidation_maturity_incidence_v2"
            if risk_gated
            else "forced_invalidation_maturity_incidence_confidence_v1"
        ),
        "risk_gated": bool(risk_gated),
        "require_commit_risk": bool(require_commit_risk),
        "remask_per_block": int(remask_per_block),
        "remasked_positions": selected_count,
        "selected_forced_commits": int(selected_forced),
        "selected_immature_commits": int(selected_immature),
        "selected_response_invalidations": int(selected_invalidation_sum),
        "selected_incident_count": int(selected_incident_sum),
        "mean_invalidations_per_remasked_position": (
            selected_invalidation_sum / selected_count if selected_count else 0.0
        ),
        "mean_incident_count_per_remasked_position": (
            selected_incident_sum / selected_count if selected_count else 0.0
        ),
        "blocks": block_records,
    }
    return remask, summary


@torch.no_grad()
def repair_response_refine_positions(
    model,
    draft,
    remask,
    prompt_length,
    dependency_threshold,
    repair_steps,
    gen_length,
    block_length,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    """Repair the selected frontier under a preserved full-draft skeleton.

    Attention remains directed.  The first token in every block is the
    high-outgoing, low-incoming source according to the live repair forward;
    later tokens use conditioned revision margin and therefore treat earlier
    top-1 candidates as probes rather than as labels.
    """
    if draft.shape != remask.shape:
        raise ValueError("draft and remask must have identical shapes")
    num_blocks = gen_length // block_length
    if repair_steps % num_blocks != 0:
        raise ValueError("repair_steps must be divisible by the number of blocks")
    allocated_steps = repair_steps // num_blocks
    x = draft.clone()
    mutable = remask.clone().to(dtype=torch.bool, device=x.device)
    mutable[:, :prompt_length] = False
    x[mutable] = mask_id
    nfe = 0
    selection_order = []
    source_records = []
    summary = {
        "decoder": "directed_response_refine_repair_v1",
        "dependency_mode": "directed_read",
        "temporal_mode": "revision_margin",
        "repair_nfe": 0,
        "remasked_positions": int(mutable.sum().item()),
        "response_validations": 0,
        "response_invalidations": 0,
        "revision_margin_candidates": 0,
        "revision_margin_sum": 0.0,
        "revision_margin_max": 0.0,
        "strongly_dependent_candidates": 0,
        "source_first_overrides": 0,
        "underfilled_steps": 0,
    }

    for block_idx in range(num_blocks):
        block_start = prompt_length + block_idx * block_length
        block_end = block_start + block_length
        planned_count = int(mutable[:, block_start:block_end].sum().item())
        if planned_count == 0:
            continue
        budget = max(1, int(np.ceil(planned_count / allocated_steps)))
        previous_top1 = None
        previous_topk_ids = None
        previous_response_credit = None
        previous_selected = None
        first_step = True
        while (x[:, block_start:block_end] == mask_id).any():
            nfe += 1
            mask_index = x == mask_id
            mask_index[:, :block_start] = False
            mask_index[:, block_end:] = False
            current_budget = min(budget, int(mask_index[0].sum().item()))
            logits, directional_attention, _ = _forward_with_block_attention(
                model, x, block_start, block_end
            )
            x0, transfer_index, diagnostics, current_topk_ids = (
                select_attention_stability_tokens(
                    logits=logits,
                    temperature=temperature,
                    remasking=remasking,
                    mask_index=mask_index,
                    x=x,
                    budget=current_budget,
                    dependency=directional_attention,
                    dependency_threshold=dependency_threshold,
                    block_start=block_start,
                    previous_top1=previous_top1,
                    previous_selected=previous_selected,
                    previous_topk_ids=previous_topk_ids,
                    temporal_mode="revision_margin",
                    temporal_topk=4,
                    prune_stable_conflicts=True,
                    fill_budget=True,
                    previous_response_credit=previous_response_credit,
                )
            )
            current_response_credit = diagnostics.pop(
                "_current_response_credit_full"
            )

            if first_step and current_budget == 1:
                candidate_state = diagnostics["candidate_state"][0]
                candidates = candidate_state["masked_positions_global"].tolist()
                confidence_by_position = {
                    int(position): float(confidence)
                    for position, confidence in zip(
                        candidates,
                        candidate_state["top1_confidences"].tolist(),
                    )
                }
                strong_edges = directional_attention[0] > dependency_threshold
                incoming_source_count = strong_edges.sum(dim=1)
                outgoing_reader_count = strong_edges.sum(dim=0)

                def source_key(global_position):
                    local_position = int(global_position) - block_start
                    return (
                        -int(outgoing_reader_count[local_position].item()),
                        int(incoming_source_count[local_position].item()),
                        -confidence_by_position[int(global_position)],
                        int(global_position),
                    )

                source_position = min(candidates, key=source_key)
                transfer_index.zero_()
                transfer_index[0, source_position] = True
                local_source = int(source_position) - block_start
                source_records.append(
                    {
                        "block_index": int(block_idx),
                        "global_position": int(source_position),
                        "local_position": int(local_source),
                        "outgoing_reader_count": int(
                            outgoing_reader_count[local_source].item()
                        ),
                        "incoming_source_count": int(
                            incoming_source_count[local_source].item()
                        ),
                        "confidence": confidence_by_position[int(source_position)],
                    }
                )
                summary["source_first_overrides"] += 1
            if not transfer_index.any():
                raise RuntimeError("response-refine repair made no progress")

            selected_positions = torch.where(transfer_index[0])[0]
            selection_order.extend(int(item) for item in selected_positions.tolist())
            summary["response_validations"] += diagnostics["response_validations"]
            summary["response_invalidations"] += diagnostics[
                "response_invalidations"
            ]
            summary["revision_margin_candidates"] += diagnostics[
                "revision_margin_candidates"
            ]
            summary["revision_margin_sum"] += diagnostics["revision_margin_sum"]
            summary["revision_margin_max"] = max(
                summary["revision_margin_max"], diagnostics["revision_margin_max"]
            )
            summary["strongly_dependent_candidates"] += diagnostics[
                "strongly_dependent_candidates"
            ]
            summary["underfilled_steps"] += int(diagnostics["underfilled"])
            previous_top1 = x0.detach().clone()
            previous_topk_ids = current_topk_ids.detach().clone()
            previous_response_credit = current_response_credit.detach().clone()
            previous_selected = selected_positions
            x[transfer_index] = x0[transfer_index]
            first_step = False

    summary["repair_nfe"] = int(nfe)
    summary["selection_order_global"] = selection_order
    summary["source_first"] = source_records
    summary["revised_token_count"] = int(((x != draft) & mutable).sum().item())
    summary["residual_mask_count"] = int((x == mask_id).sum().item())
    if summary["revision_margin_candidates"]:
        summary["revision_margin_mean"] = (
            summary["revision_margin_sum"]
            / summary["revision_margin_candidates"]
        )
    else:
        summary["revision_margin_mean"] = 0.0
    return x, nfe, summary


@torch.no_grad()
def retain_response_refine_blocks(
    model,
    draft,
    repaired,
    remask,
    prompt_length,
    gen_length,
    block_length,
    mask_id=126336,
    require_pareto=False,
):
    """Keep a repaired block only when it beats the retained draft.

    All positions changed by the repair are masked in one shared context.  The
    original and repaired token values are therefore compared under exactly the
    same explicit conditions.  Scores are accumulated per generation block so
    horizontally coupled changes are accepted or rejected together.  The rule
    has no learned verifier, score weight, or additional threshold.
    """
    if draft.shape != repaired.shape or draft.shape != remask.shape:
        raise ValueError("draft, repaired, and remask must have identical shapes")
    changed = (draft != repaired) & remask.to(device=draft.device, dtype=torch.bool)
    changed[:, :prompt_length] = False
    output = draft.clone()
    summary = {
        "selector": (
            "shared_mask_block_pareto_retention_v2"
            if require_pareto
            else "shared_mask_block_retention_v1"
        ),
        "changed_positions": int(changed.sum().item()),
        "accepted_blocks": 0,
        "rejected_blocks": 0,
        "accepted_positions": 0,
        "selector_nfe": 0,
        "blocks": [],
    }
    if not changed.any():
        return output, 0, summary

    score_context = draft.clone()
    score_context[changed] = mask_id
    log_probs = torch.log_softmax(model(score_context).logits.float(), dim=-1)
    summary["selector_nfe"] = 1
    num_blocks = gen_length // block_length
    for block_idx in range(num_blocks):
        block_start = prompt_length + block_idx * block_length
        block_end = block_start + block_length
        block_changed = changed[:, block_start:block_end]
        local_positions = torch.where(block_changed[0])[0]
        if local_positions.numel() == 0:
            continue
        global_positions = local_positions + block_start
        original_tokens = draft[0, global_positions]
        repaired_tokens = repaired[0, global_positions]
        original_token_scores = log_probs[0, global_positions, original_tokens]
        repaired_token_scores = log_probs[0, global_positions, repaired_tokens]
        token_margins = repaired_token_scores - original_token_scores
        original_score = float(original_token_scores.sum().item())
        repaired_score = float(repaired_token_scores.sum().item())
        accept = bool(
            (token_margins > 0).all().item()
            if require_pareto
            else repaired_score > original_score
        )
        if accept:
            output[0, global_positions] = repaired[0, global_positions]
            summary["accepted_blocks"] += 1
            summary["accepted_positions"] += int(global_positions.numel())
        else:
            summary["rejected_blocks"] += 1
        summary["blocks"].append(
            {
                "block_index": int(block_idx),
                "changed_positions": [int(item) for item in global_positions.tolist()],
                "original_log_score": original_score,
                "repaired_log_score": repaired_score,
                "score_margin": repaired_score - original_score,
                "minimum_token_margin": float(token_margins.min().item()),
                "accepted": bool(accept),
            }
        )
    return output, 1, summary


def _directed_cross_condition_partitions(directional_attention, local_positions):
    """Split changed positions into source/dependent views without symmetrizing.

    ``directional_attention[target, source]`` measures how strongly a target
    reads a source.  Net outgoing flow therefore orders source-like positions
    before dependent positions.  Alternating that order creates two
    complementary views in which each side is scored while the other side is
    present as explicit tokens.
    """
    if local_positions.numel() == 0:
        return [], []
    local_positions = local_positions.to(
        device=directional_attention.device, dtype=torch.long
    )
    sub_rows = directional_attention.index_select(0, local_positions)
    submatrix = sub_rows.index_select(1, local_positions).float()
    outgoing = submatrix.sum(dim=0)
    incoming = submatrix.sum(dim=1)
    net_flow = outgoing - incoming
    ordered = sorted(
        range(local_positions.numel()),
        key=lambda index: (
            -float(net_flow[index].item()),
            int(local_positions[index].item()),
        ),
    )
    source_view = [int(local_positions[index].item()) for index in ordered[::2]]
    dependent_view = [
        int(local_positions[index].item()) for index in ordered[1::2]
    ]
    return source_view, dependent_view


@torch.no_grad()
def retain_response_refine_blocks_cross_conditioned(
    model,
    draft,
    repaired,
    remask,
    prompt_length,
    gen_length,
    block_length,
    directional_attention,
    mask_id=126336,
):
    """Retain repairs only after complementary explicit-token validation.

    The shared-mask selector cannot test interactions among changed tokens.
    Here the directed-attention order divides each changed block into two
    views.  One view is masked while the other remains explicit, then the
    roles are exchanged.  Original and repaired drafts are evaluated in one
    batched forward per view.  A repair is retained only when every changed
    token improves in its cross-conditioned view.  This is a strict Pareto
    rule and introduces no score weight or acceptance threshold.
    """
    if draft.shape != repaired.shape or draft.shape != remask.shape:
        raise ValueError("draft, repaired, and remask must have identical shapes")
    num_blocks = gen_length // block_length
    if directional_attention.shape[:2] != (num_blocks, block_length):
        raise ValueError("directional attention does not match generation blocks")
    changed = (draft != repaired) & remask.to(device=draft.device, dtype=torch.bool)
    changed[:, :prompt_length] = False
    output = draft.clone()
    summary = {
        "selector": "directed_cross_conditioned_block_pareto_v1",
        "changed_positions": int(changed.sum().item()),
        "accepted_blocks": 0,
        "rejected_blocks": 0,
        "accepted_positions": 0,
        "selector_nfe": 0,
        "selector_candidate_evaluations": 0,
        "blocks": [],
    }
    if not changed.any():
        return output, 0, summary

    for block_idx in range(num_blocks):
        block_start = prompt_length + block_idx * block_length
        block_end = block_start + block_length
        local_positions = torch.where(changed[0, block_start:block_end])[0]
        if local_positions.numel() == 0:
            continue
        global_positions = local_positions + block_start
        source_view, dependent_view = _directed_cross_condition_partitions(
            directional_attention[block_idx], local_positions
        )
        views = [source_view]
        if dependent_view:
            views.append(dependent_view)

        original_candidate = draft.clone()
        repaired_candidate = draft.clone()
        repaired_candidate[0, global_positions] = repaired[0, global_positions]
        view_records = []
        all_token_margins = []
        for view_index, local_view in enumerate(views):
            view_global = torch.tensor(
                [block_start + position for position in local_view],
                dtype=torch.long,
                device=draft.device,
            )
            if len(views) == 1:
                context = original_candidate.clone()
                context[0, view_global] = mask_id
                logits = model(context).logits.float()
                log_probs = torch.log_softmax(logits, dim=-1)
                original_scores = log_probs[
                    0, view_global, original_candidate[0, view_global]
                ]
                repaired_scores = log_probs[
                    0, view_global, repaired_candidate[0, view_global]
                ]
                candidate_evaluations = 1
            else:
                contexts = torch.cat(
                    [original_candidate.clone(), repaired_candidate.clone()], dim=0
                )
                contexts[:, view_global] = mask_id
                logits = model(contexts).logits.float()
                log_probs = torch.log_softmax(logits, dim=-1)
                original_scores = log_probs[
                    0, view_global, original_candidate[0, view_global]
                ]
                repaired_scores = log_probs[
                    1, view_global, repaired_candidate[0, view_global]
                ]
                candidate_evaluations = 2
            margins = repaired_scores - original_scores
            all_token_margins.append(margins)
            summary["selector_nfe"] += 1
            summary["selector_candidate_evaluations"] += candidate_evaluations
            view_records.append(
                {
                    "view_index": int(view_index),
                    "masked_positions": [
                        int(position) for position in view_global.tolist()
                    ],
                    "original_log_score": float(original_scores.sum().item()),
                    "repaired_log_score": float(repaired_scores.sum().item()),
                    "score_margin": float(margins.sum().item()),
                    "minimum_token_margin": float(margins.min().item()),
                }
            )

        token_margins = torch.cat(all_token_margins)
        accept = bool((token_margins > 0).all().item())
        if accept:
            output[0, global_positions] = repaired[0, global_positions]
            summary["accepted_blocks"] += 1
            summary["accepted_positions"] += int(global_positions.numel())
        else:
            summary["rejected_blocks"] += 1
        summary["blocks"].append(
            {
                "block_index": int(block_idx),
                "changed_positions": [
                    int(position) for position in global_positions.tolist()
                ],
                "source_view_local_positions": source_view,
                "dependent_view_local_positions": dependent_view,
                "minimum_token_margin": float(token_margins.min().item()),
                "accepted": bool(accept),
                "views": view_records,
            }
        )
    return output, summary["selector_nfe"], summary


@torch.no_grad()
def generate_response_refine(
    model,
    prompt,
    dependency_threshold,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    eos_id=126081,
    early_stop=False,
    budget_mode="matched",
):
    """Full-draft response refinement with fixed horizontal/vertical rules.

    ``matched`` splits the original NFE evenly between draft and repair.
    ``extra`` keeps the full parent draft budget and adds a half-budget repair.
    These are discrete compute regimes rather than tuned score weights.
    """
    if budget_mode not in {
        "matched",
        "extra",
        "gated",
        "causal_pareto",
        "causal_cross_pareto",
    }:
        raise ValueError(
            "budget_mode must be matched, extra, gated, causal_pareto, "
            "or causal_cross_pareto"
        )
    num_blocks = gen_length // block_length
    repair_steps = steps // 2
    fill_steps = steps // 2 if budget_mode == "matched" else steps
    if steps % 2 or fill_steps % num_blocks or repair_steps % num_blocks:
        raise ValueError("fill and repair steps must divide evenly across blocks")

    draft, fill_nfe, fill_summary = generate_attention_stability(
        model=model,
        prompt=prompt,
        dependency_threshold=dependency_threshold,
        steps=fill_steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        remasking=remasking,
        mask_id=mask_id,
        eos_id=eos_id,
        early_stop=early_stop,
        collect_step_diagnostics=False,
        dependency_mode="symmetric",
        temporal_mode="top1",
        temporal_topk=4,
        prune_stable_conflicts=True,
        fill_budget=True,
        collect_position_risk=True,
    )
    position_risk = fill_summary.pop("_position_risk_state")
    remask, frontier_summary = build_response_refine_mask(
        position_risk=position_risk,
        prompt_length=prompt.shape[1],
        gen_length=gen_length,
        block_length=block_length,
        dependency_threshold=dependency_threshold,
        repair_steps=repair_steps,
        risk_gated=budget_mode in {
            "gated",
            "causal_pareto",
            "causal_cross_pareto",
        },
        require_commit_risk=budget_mode in {
            "causal_pareto",
            "causal_cross_pareto",
        },
    )
    repaired, repair_nfe, repair_summary = repair_response_refine_positions(
        model=model,
        draft=draft,
        remask=remask,
        prompt_length=prompt.shape[1],
        dependency_threshold=dependency_threshold,
        repair_steps=repair_steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        remasking=remasking,
        mask_id=mask_id,
    )
    retained = repaired
    selector_nfe = 0
    retention_summary = None
    if budget_mode == "causal_cross_pareto":
        retained, selector_nfe, retention_summary = (
            retain_response_refine_blocks_cross_conditioned(
                model=model,
                draft=draft,
                repaired=repaired,
                remask=remask,
                prompt_length=prompt.shape[1],
                gen_length=gen_length,
                block_length=block_length,
                directional_attention=position_risk["final_directional_attention"],
                mask_id=mask_id,
            )
        )
    elif budget_mode in {"gated", "causal_pareto"}:
        retained, selector_nfe, retention_summary = retain_response_refine_blocks(
            model=model,
            draft=draft,
            repaired=repaired,
            remask=remask,
            prompt_length=prompt.shape[1],
            gen_length=gen_length,
            block_length=block_length,
            mask_id=mask_id,
            require_pareto=budget_mode == "causal_pareto",
        )
    decoder = "response_refine_v1"
    if budget_mode == "gated":
        decoder = "response_refine_v2"
    elif budget_mode == "causal_pareto":
        decoder = "response_refine_v3"
    elif budget_mode == "causal_cross_pareto":
        decoder = "response_refine_v4"
    summary = {
        "decoder": decoder,
        "budget_mode": budget_mode,
        "dependency_threshold": float(dependency_threshold),
        "configured_steps": int(steps),
        "fill_steps": int(fill_steps),
        "repair_steps": int(repair_steps),
        "fill_nfe": int(fill_nfe),
        "repair_nfe": int(repair_nfe),
        "selector_nfe": int(selector_nfe),
        "total_nfe": int(fill_nfe + repair_nfe + selector_nfe),
        "gen_length": int(gen_length),
        "block_length": int(block_length),
        "frontier": frontier_summary,
        "fill": fill_summary,
        "repair": repair_summary,
        "retention": retention_summary,
        "draft_to_repaired_changes": int((draft != repaired).sum().item()),
        "draft_to_retained_changes": int((draft != retained).sum().item()),
        "residual_mask_count": int((retained == mask_id).sum().item()),
    }
    if budget_mode == "causal_cross_pareto":
        summary["_draft_candidate_token_ids"] = {
            "anchor": draft.detach().to(torch.int32).cpu(),
            "repaired": retained.detach().to(torch.int32).cpu(),
        }
    return retained, summary["total_nfe"], summary


@torch.no_grad()
def repair_draft_disagreements(
    model,
    draft,
    disagreement_mask,
    prompt_length,
    dependency_threshold,
    steps,
    gen_length,
    block_length,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    """Re-denoise only positions where two complete drafts disagree.

    Agreed tokens form a full-draft skeleton.  Disagreement positions are
    explicitly masked, so every repaired token is predicted again under the
    shared skeleton rather than copied from either parent.  The repair budget
    is derived from the original tokens-per-step schedule and introduces no
    additional selection threshold.
    """
    if draft.shape != disagreement_mask.shape:
        raise ValueError("draft and disagreement_mask must have identical shapes")
    if draft.shape[0] != 1:
        raise ValueError("draft exchange currently supports batch size one")
    if gen_length % block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError("steps must be divisible by the number of blocks")

    x = draft.clone()
    mutable = disagreement_mask.clone().to(dtype=torch.bool, device=x.device)
    mutable[:, :prompt_length] = False
    x[mutable] = mask_id
    block_steps = steps // num_blocks
    baseline_budget = max(1, int(np.ceil(block_length / block_steps)))
    nfe = 0
    summary = {
        "decoder": "response_credit_disagreement_repair_v1",
        "disagreement_positions": int(mutable.sum().item()),
        "agreement_positions": int(gen_length - mutable.sum().item()),
        "repair_nfe": 0,
        "response_validations": 0,
        "response_invalidations": 0,
        "response_credit_max": 0,
        "rejected_pairs": 0,
        "stable_conflicts_pruned": 0,
        "forced_budget_fills": 0,
        "underfilled_steps": 0,
    }

    for block_idx in range(num_blocks):
        block_start = prompt_length + block_idx * block_length
        block_end = block_start + block_length
        previous_top1 = None
        previous_topk_ids = None
        previous_response_credit = None
        previous_selected = None
        while (x[:, block_start:block_end] == mask_id).any():
            nfe += 1
            mask_index = x == mask_id
            mask_index[:, :block_start] = False
            mask_index[:, block_end:] = False
            budget = min(baseline_budget, int(mask_index[0].sum().item()))
            logits, _, dependency = _forward_with_block_attention(
                model, x, block_start, block_end
            )
            x0, transfer_index, diagnostics, current_topk_ids = select_attention_stability_tokens(
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
                previous_topk_ids=previous_topk_ids,
                temporal_mode="response_credit",
                temporal_topk=4,
                prune_stable_conflicts=True,
                fill_budget=True,
                previous_response_credit=previous_response_credit,
            )
            current_response_credit = diagnostics.pop(
                "_current_response_credit_full"
            )
            if not transfer_index.any():
                raise RuntimeError("draft-disagreement repair made no progress")
            summary["response_validations"] += diagnostics["response_validations"]
            summary["response_invalidations"] += diagnostics["response_invalidations"]
            summary["response_credit_max"] = max(
                summary["response_credit_max"], diagnostics["response_credit_max"]
            )
            summary["rejected_pairs"] += diagnostics["rejected_pairs"]
            summary["stable_conflicts_pruned"] += diagnostics[
                "stable_conflicts_pruned"
            ]
            summary["forced_budget_fills"] += diagnostics["forced_budget_fills"]
            summary["underfilled_steps"] += int(diagnostics["underfilled"])
            previous_top1 = x0.detach().clone()
            previous_topk_ids = current_topk_ids.detach().clone()
            previous_response_credit = current_response_credit.detach().clone()
            previous_selected = torch.where(transfer_index[0])[0]
            x[transfer_index] = x0[transfer_index]

    summary["repair_nfe"] = nfe
    summary["residual_mask_count"] = int((x == mask_id).sum().item())
    return x, nfe, summary


@torch.no_grad()
def generate_response_credit_exchange(
    model,
    prompt,
    dependency_threshold,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    eos_id=126081,
    early_stop=False,
    prune_stable_conflicts=False,
    fill_budget=False,
):
    """Generate two policy-diverse drafts and repair their disagreement set.

    The anchor is the established top-1 temporal parent.  The explorer uses
    response credit: a candidate gains credit only when it survives an actual
    strong-dependency conditioning event, and loses that credit when the event
    changes its top-1.  Their agreement is preserved as a full-draft skeleton;
    disagreements are masked and revalidated by the ordinary dLLM forward.
    """
    common = dict(
        model=model,
        prompt=prompt,
        dependency_threshold=dependency_threshold,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        remasking=remasking,
        mask_id=mask_id,
        eos_id=eos_id,
        early_stop=early_stop,
        collect_step_diagnostics=False,
        dependency_mode="symmetric",
        temporal_topk=4,
        prune_stable_conflicts=prune_stable_conflicts,
        fill_budget=fill_budget,
    )
    anchor, anchor_nfe, anchor_summary = generate_attention_stability(
        temporal_mode="top1", **common
    )
    explorer, explorer_nfe, explorer_summary = generate_attention_stability(
        temporal_mode="response_credit", **common
    )
    prompt_length = prompt.shape[1]
    disagreement = anchor != explorer
    disagreement[:, :prompt_length] = False
    fused = anchor.clone()
    repaired, repair_nfe, repair_summary = repair_draft_disagreements(
        model=model,
        draft=fused,
        disagreement_mask=disagreement,
        prompt_length=prompt_length,
        dependency_threshold=dependency_threshold,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        remasking=remasking,
        mask_id=mask_id,
    )
    summary = {
        "decoder": "response_credit_draft_exchange_v1",
        "dependency_threshold": float(dependency_threshold),
        "configured_steps": int(steps),
        "gen_length": int(gen_length),
        "block_length": int(block_length),
        "draft_count": 2,
        "anchor_nfe": int(anchor_nfe),
        "explorer_nfe": int(explorer_nfe),
        "repair_nfe": int(repair_nfe),
        "total_nfe": int(anchor_nfe + explorer_nfe + repair_nfe),
        "draft_disagreement_positions": int(disagreement.sum().item()),
        "draft_agreement_positions": int(gen_length - disagreement.sum().item()),
        "anchor": anchor_summary,
        "explorer": explorer_summary,
        "repair": repair_summary,
        "residual_mask_count": int((repaired == mask_id).sum().item()),
        "_draft_candidate_token_ids": {
            "anchor": anchor.detach().to(torch.int32).cpu(),
            "explorer": explorer.detach().to(torch.int32).cpu(),
            "repaired": repaired.detach().to(torch.int32).cpu(),
        },
    }
    return repaired, summary["total_nfe"], summary


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
