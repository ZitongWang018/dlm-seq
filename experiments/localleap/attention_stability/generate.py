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
    return output.logits, symmetric


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
    diagnostics = {"unstable_candidates": 0, "rejected_pairs": 0, "underfilled": False}

    for batch_idx in range(x.shape[0]):
        masked = torch.where(mask_index[batch_idx])[0]
        if masked.numel() == 0:
            continue
        maturity = torch.ones(masked.numel(), dtype=torch.bool, device=x.device)
        if previous_top1 is not None and previous_selected is not None and previous_selected.numel() > 0:
            masked_local = masked - block_start
            selected_local = previous_selected - block_start
            max_dependency = dependency[batch_idx].index_select(0, masked_local).index_select(1, selected_local).max(dim=1).values
            changed = x0[batch_idx, masked] != previous_top1[batch_idx, masked]
            maturity = ~((max_dependency > dependency_threshold) & changed)
            diagnostics["unstable_candidates"] += int((~maturity).sum().item())

        mature_positions = masked[maturity]
        immature_positions = masked[~maturity]
        if mature_positions.numel() == 0:
            # Explicit all-immature fallback from the algorithm definition.
            ordered = masked[torch.topk(confidence[batch_idx, masked], k=masked.numel()).indices]
        else:
            mature_order = mature_positions[torch.topk(confidence[batch_idx, mature_positions], k=mature_positions.numel()).indices]
            if immature_positions.numel() > 0:
                immature_order = immature_positions[torch.topk(confidence[batch_idx, immature_positions], k=immature_positions.numel()).indices]
                ordered = torch.cat((mature_order, immature_order))
            else:
                ordered = mature_order

        selected = []
        for position in ordered.tolist():
            if len(selected) >= int(budget):
                break
            if selected:
                local_position = position - block_start
                selected_local = torch.tensor([item - block_start for item in selected], device=x.device)
                if dependency[batch_idx, local_position, selected_local].max() > dependency_threshold:
                    diagnostics["rejected_pairs"] += 1
                    continue
            selected.append(position)
        if len(selected) < min(int(budget), masked.numel()):
            diagnostics["underfilled"] = True
        if selected:
            transfer_index[batch_idx, torch.tensor(selected, device=x.device)] = True

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
        "rejected_pairs": 0,
        "underfilled_steps": 0,
        "dependency_max": 0.0,
        "dependency_mean_sum": 0.0,
        "dependency_observations": 0,
    }

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
            logits, dependency = _forward_with_block_attention(model, x, block_start, block_end)
            summary["dependency_max"] = max(summary["dependency_max"], float(dependency.max().item()))
            summary["dependency_mean_sum"] += float(dependency.mean().item())
            summary["dependency_observations"] += 1
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
            summary["rejected_pairs"] += step_diagnostics["rejected_pairs"]
            summary["underfilled_steps"] += int(step_diagnostics["underfilled"])
            previous_top1 = x0.detach().clone()
            previous_selected = torch.where(transfer_index[0])[0]
            x[transfer_index] = x0[transfer_index]
            step_idx += 1

        if early_stop and (x[:, block_start:block_end] == eos_id).any():
            x[:, block_end:] = eos_id
            break

    if summary["dependency_observations"]:
        summary["dependency_mean"] = summary.pop("dependency_mean_sum") / summary["dependency_observations"]
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
