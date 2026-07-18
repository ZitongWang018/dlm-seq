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

'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import accelerate
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig
from model.modeling_llada import LLaDAModelLM
from generate import *
from stcc_generate import generate_stcc
from differential_selector import (
    has_public_checks,
    select_differential_candidate,
    select_public_example_guard,
)

import json
import hashlib
import time
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def decode_candidate_generations(
    tokenizer,
    candidate_token_ids,
    prompt_length,
    stop_tokens,
    preserve_full_humaneval_response=False,
):
    """Decode stored trajectory candidates using the selected-output policy.

    HumanEval selected responses deliberately keep the full decoded response so
    the official sanitizer can recover the function.  Applying the generic stop
    sequences only to diagnostic candidates made their trace text shorter than
    the actual selectable response and invalidated candidate-oracle analysis.
    """
    decoded = {}
    for candidate_name, candidate_ids in candidate_token_ids.items():
        candidate_text = tokenizer.decode(
            candidate_ids[0][prompt_length:],
            skip_special_tokens=True,
        )
        if not preserve_full_humaneval_response:
            for stop_seq in stop_tokens:
                if stop_seq in candidate_text:
                    candidate_text = candidate_text.split(stop_seq)[0]
        decoded[candidate_name] = candidate_text
    return decoded


def all_gather_cpu(val, group=None):
    """
    Gather scalar values from all ranks to rank0.
    """
    val_tensor = torch.tensor([val], dtype=torch.float32).cuda()
    gathered = [torch.zeros_like(val_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, val_tensor, group=group)
    gathered = [v.cpu().item() for v in gathered]
    return gathered


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        cfg=0.,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking='low_confidence',
        device="cuda",
        early_stop=False,
        use_cache=False,
        threshold=None,
        relaxed_threshold=None,
        radius=4,
        save_dir=None,
        show_speed=False,
        integrate_speed=False,
        dependency_threshold=None,
        dependency_trace_dir=None,
        dependency_diagnostics_dir=None,
        dependency_mode="symmetric",
        dependency_temporal_mode="top1",
        dependency_temporal_topk=4,
        dependency_prune_stable_conflicts=False,
        dependency_fill_budget=False,
        dependency_likelihood_selection=False,
        dependency_likelihood_selection_mode="mean",
        dependency_draft_exchange=False,
        dependency_differential_selection=False,
        dependency_response_refine=False,
        dependency_response_refine_budget_mode="matched",
        candidate_memory_topk=None,
        candidate_memory_confidence_threshold=0.0,
        candidate_memory_fallback="confidence",
        candidate_memory_exact_jsd=False,
        candidate_memory_trace_dir=None,
        candidate_memory_diagnostics_dir=None,
        stcc_mode=None,
        stcc_topk=8,
        stcc_jsd_threshold=0.01,
        stcc_attention_threshold=0.004,
        stcc_extra_multiplier=1,
        stcc_extra_jsd_threshold=None,
        stcc_min_topk_overlap=None,
        stcc_min_stability_streak=1,
        stcc_trace_dir=None,
        stcc_diagnostics_dir=None,
        **kwargs,
    ):
        '''
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which
                             returns a True/False judgment used for accuracy calculation.
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function.
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality,
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
        '''
        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})
        config = AutoConfig.from_pretrained(model_path)
        config.flash_attention = True

        self.model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, config=config, **model_kwargs)
        self.model.eval()
        self.model_path = model_path

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)
            self._rank = 0
            self._world_size = 1

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.cfg = cfg  # 【add】
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking
        self.early_stop = early_stop
        self.use_cache = use_cache
        self.threshold = threshold
        self.relaxed_threshold = relaxed_threshold
        self.radius = radius

        self.is_instruct = True if 'instruct' in model_path.lower() else False
        self.save_dir = save_dir
        self.show_speed = show_speed
        self.integrate_speed = integrate_speed
        self.dependency_threshold = dependency_threshold
        self.dependency_trace_dir = dependency_trace_dir
        self.dependency_diagnostics_dir = dependency_diagnostics_dir
        self.dependency_mode = dependency_mode
        self.dependency_temporal_mode = dependency_temporal_mode
        self.dependency_temporal_topk = int(dependency_temporal_topk)
        self.dependency_prune_stable_conflicts = (
            dependency_prune_stable_conflicts.lower() == "true"
            if isinstance(dependency_prune_stable_conflicts, str)
            else bool(dependency_prune_stable_conflicts)
        )
        self.dependency_fill_budget = (
            dependency_fill_budget.lower() == "true"
            if isinstance(dependency_fill_budget, str)
            else bool(dependency_fill_budget)
        )
        self.dependency_likelihood_selection = (
            dependency_likelihood_selection.lower() == "true"
            if isinstance(dependency_likelihood_selection, str)
            else bool(dependency_likelihood_selection)
        )
        self.dependency_likelihood_selection_mode = str(
            dependency_likelihood_selection_mode
        )
        if self.dependency_likelihood_selection_mode not in {
            "mean",
            "block_evidence",
            "disagreement_evidence",
            "consensus_block",
            "lazy_consensus_block",
            "coverage_consensus_block",
            "convergent_coverage_consensus_block",
            "shared_skeleton",
            "bidirectional_block",
            "confirmed_bidirectional_block",
            "early_confirmed_bidirectional_block",
            "confirmed_bidirectional_public_guard",
        }:
            raise ValueError(
                "dependency_likelihood_selection_mode must be mean, "
                "block_evidence, disagreement_evidence, consensus_block, "
                "lazy_consensus_block, coverage_consensus_block, or "
                "convergent_coverage_consensus_block, shared_skeleton, or "
                "bidirectional_block, confirmed_bidirectional_block, or "
                "early_confirmed_bidirectional_block, or "
                "confirmed_bidirectional_public_guard"
            )
        self.dependency_draft_exchange = (
            dependency_draft_exchange.lower() == "true"
            if isinstance(dependency_draft_exchange, str)
            else bool(dependency_draft_exchange)
        )
        self.dependency_differential_selection = (
            dependency_differential_selection.lower() == "true"
            if isinstance(dependency_differential_selection, str)
            else bool(dependency_differential_selection)
        )
        self.dependency_response_refine = (
            dependency_response_refine.lower() == "true"
            if isinstance(dependency_response_refine, str)
            else bool(dependency_response_refine)
        )
        self.dependency_response_refine_budget_mode = str(
            dependency_response_refine_budget_mode
        )
        if self.dependency_response_refine_budget_mode not in {
            "matched",
            "extra",
            "gated",
            "causal_pareto",
            "causal_cross_pareto",
        }:
            raise ValueError(
                "dependency_response_refine_budget_mode must be matched, extra, "
                "gated, causal_pareto, or causal_cross_pareto"
            )
        self.candidate_memory_topk = candidate_memory_topk
        self.candidate_memory_confidence_threshold = candidate_memory_confidence_threshold
        self.candidate_memory_fallback = candidate_memory_fallback
        self.candidate_memory_exact_jsd = (
            candidate_memory_exact_jsd.lower() == "true"
            if isinstance(candidate_memory_exact_jsd, str)
            else bool(candidate_memory_exact_jsd)
        )
        self.candidate_memory_trace_dir = candidate_memory_trace_dir
        self.candidate_memory_diagnostics_dir = candidate_memory_diagnostics_dir
        self.stcc_mode = stcc_mode
        self.stcc_topk = int(stcc_topk)
        self.stcc_jsd_threshold = float(stcc_jsd_threshold)
        self.stcc_attention_threshold = float(stcc_attention_threshold)
        self.stcc_extra_multiplier = int(stcc_extra_multiplier)
        self.stcc_extra_jsd_threshold = (
            None
            if stcc_extra_jsd_threshold in {None, "None", "none"}
            else float(stcc_extra_jsd_threshold)
        )
        self.stcc_min_topk_overlap = (
            None
            if stcc_min_topk_overlap in {None, "None", "none"}
            else int(stcc_min_topk_overlap)
        )
        self.stcc_min_stability_streak = int(stcc_min_stability_streak)
        self.stcc_trace_dir = stcc_trace_dir
        self.stcc_diagnostics_dir = stcc_diagnostics_dir

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))

                self.total_instances += 1
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError


    def generate_until(self, requests):
        output = []
        num_tokens = 0
        num_nfe = 0
        processed_count = 0
        num_instances = 0
        num_input_tokens = 0
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            rank = self.rank
            save_path = os.path.join(self.save_dir, f'rank_{rank}.jsonl')
            print(f"save_path: {save_path}")
            if os.path.exists(save_path):
                print(f"load from {save_path}")
                with open(save_path, 'r', encoding='utf-8') as f:
                    output = [json.loads(line) for line in f]
                    processed_count = len(output)
                print(f"processed_count: {processed_count}")
        start_time = time.time()
        for i, req in enumerate(tqdm(requests, desc="Generating...")):
            if i < processed_count:
                continue

            question = req.args[0]
            if self.is_instruct:
                m = [{"role": "user", "content": question}]
                user_input = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                input_ids = self.tokenizer(user_input)['input_ids']
            else:
                user_input = question
                input_ids = self.tokenizer(user_input)['input_ids']

            stop_tokens = req.args[1]['until']
            input_ids = torch.tensor(input_ids).to(self.device).unsqueeze(0)  # [1, prompt_len]
            num_input_tokens += input_ids.shape[1]

            decode_diagnostics = None
            draft_candidate_token_ids = None
            candidate_generations = None
            step_records = None
            active_trace_dir = None
            active_diagnostics_dir = None
            step_diagnostics_schema = None
            trace_evaluator_version = None
            public_guard_requested = False
            public_guard_active = False
            if self.stcc_mode is not None:
                generated_answer, nfe, decode_diagnostics = generate_stcc(
                    self.model,
                    input_ids,
                    steps=self.steps,
                    gen_length=self.gen_length,
                    block_length=self.block_length,
                    mask_id=self.mask_id,
                    early_stop=self.early_stop,
                    candidate_topk=self.stcc_topk,
                    jsd_threshold=self.stcc_jsd_threshold,
                    horizontal_mode=self.stcc_mode,
                    attention_threshold=self.stcc_attention_threshold,
                    extra_multiplier=self.stcc_extra_multiplier,
                    extra_jsd_threshold=self.stcc_extra_jsd_threshold,
                    min_topk_overlap=self.stcc_min_topk_overlap,
                    min_stability_streak=self.stcc_min_stability_streak,
                    collect_step_diagnostics=self.stcc_diagnostics_dir is not None,
                )
                step_records = decode_diagnostics.pop("_step_records", None)
                active_trace_dir = self.stcc_trace_dir
                active_diagnostics_dir = self.stcc_diagnostics_dir
                step_diagnostics_schema = "stcc_distribution_response_steps_v1"
                trace_evaluator_version = "stcc_distribution_response_trace_v1"
            elif self.candidate_memory_topk is not None:
                generated_answer, nfe, decode_diagnostics = generate_candidate_memory(
                    self.model,
                    input_ids,
                    steps=self.steps,
                    gen_length=self.gen_length,
                    block_length=self.block_length,
                    temperature=0,
                    remasking=self.remasking,
                    mask_id=self.mask_id,
                    early_stop=self.early_stop,
                    candidate_topk=int(self.candidate_memory_topk),
                    confidence_threshold=float(self.candidate_memory_confidence_threshold),
                    fallback_mode=self.candidate_memory_fallback,
                    collect_exact_jsd=self.candidate_memory_exact_jsd,
                    collect_step_diagnostics=self.candidate_memory_diagnostics_dir is not None,
                )
                step_records = decode_diagnostics.pop("_step_records", None)
                active_trace_dir = self.candidate_memory_trace_dir
                active_diagnostics_dir = self.candidate_memory_diagnostics_dir
                step_diagnostics_schema = "candidate_memory_steps_v2"
                trace_evaluator_version = "candidate_memory_trace_v2"
            elif self.dependency_threshold is not None:
                if self.dependency_likelihood_selection:
                    requested_selection_mode = (
                        self.dependency_likelihood_selection_mode
                    )
                    public_guard_requested = (
                        requested_selection_mode
                        == "confirmed_bidirectional_public_guard"
                    )
                    public_guard_active = public_guard_requested and bool(
                        has_public_checks(question, req.doc.get("entry_point"))
                    )
                    effective_selection_mode = (
                        requested_selection_mode
                        if not public_guard_requested or public_guard_active
                        else "confirmed_bidirectional_block"
                    )
                    generated_answer, nfe, decode_diagnostics = (
                        generate_trajectory_likelihood_selection(
                            self.model,
                            input_ids,
                            steps=self.steps,
                            gen_length=self.gen_length,
                            block_length=self.block_length,
                            temperature=0,
                            remasking=self.remasking,
                            mask_id=self.mask_id,
                            early_stop=self.early_stop,
                            dependency_threshold=self.dependency_threshold,
                            collect_step_diagnostics=(
                                self.dependency_diagnostics_dir is not None
                            ),
                            dependency_mode=self.dependency_mode,
                            temporal_mode=self.dependency_temporal_mode,
                            temporal_topk=self.dependency_temporal_topk,
                            selection_mode=effective_selection_mode,
                        )
                    )
                    draft_candidate_token_ids = decode_diagnostics.pop(
                        "_trajectory_candidate_token_ids"
                    )
                    step_records = decode_diagnostics.pop("_step_records", None)
                elif self.dependency_response_refine:
                    generated_answer, nfe, decode_diagnostics = generate_response_refine(
                        self.model,
                        input_ids,
                        steps=self.steps,
                        gen_length=self.gen_length,
                        block_length=self.block_length,
                        temperature=0,
                        remasking=self.remasking,
                        mask_id=self.mask_id,
                        early_stop=self.early_stop,
                        dependency_threshold=self.dependency_threshold,
                        budget_mode=self.dependency_response_refine_budget_mode,
                    )
                    draft_candidate_token_ids = decode_diagnostics.pop(
                        "_draft_candidate_token_ids", None
                    )
                elif self.dependency_draft_exchange:
                    generated_answer, nfe, decode_diagnostics = generate_response_credit_exchange(
                        self.model,
                        input_ids,
                        steps=self.steps,
                        gen_length=self.gen_length,
                        block_length=self.block_length,
                        temperature=0,
                        remasking=self.remasking,
                        mask_id=self.mask_id,
                        early_stop=self.early_stop,
                        dependency_threshold=self.dependency_threshold,
                        prune_stable_conflicts=self.dependency_prune_stable_conflicts,
                        fill_budget=self.dependency_fill_budget,
                    )
                    draft_candidate_token_ids = decode_diagnostics.pop(
                        "_draft_candidate_token_ids", None
                    )
                else:
                    generated_answer, nfe, decode_diagnostics = generate_attention_stability(
                        self.model,
                        input_ids,
                        steps=self.steps,
                        gen_length=self.gen_length,
                        block_length=self.block_length,
                        temperature=0,
                        remasking=self.remasking,
                        mask_id=self.mask_id,
                        early_stop=self.early_stop,
                        dependency_threshold=self.dependency_threshold,
                        collect_step_diagnostics=self.dependency_diagnostics_dir is not None,
                        dependency_mode=self.dependency_mode,
                        temporal_mode=self.dependency_temporal_mode,
                        temporal_topk=self.dependency_temporal_topk,
                        prune_stable_conflicts=self.dependency_prune_stable_conflicts,
                        fill_budget=self.dependency_fill_budget,
                    )
                    step_records = decode_diagnostics.pop("_step_records", None)
                active_trace_dir = self.dependency_trace_dir
                active_diagnostics_dir = self.dependency_diagnostics_dir
                extended_dependency_mode = (
                    self.dependency_mode != "symmetric"
                    or self.dependency_temporal_mode != "top1"
                    or self.dependency_prune_stable_conflicts
                    or self.dependency_fill_budget
                    or self.dependency_likelihood_selection
                    or self.dependency_draft_exchange
                    or self.dependency_response_refine
                )
                step_diagnostics_schema = (
                    "attention_stability_steps_v2"
                    if extended_dependency_mode
                    else "attention_stability_steps_v1"
                )
                trace_evaluator_version = (
                    (
                        "response_refine_trace_v4"
                        if self.dependency_response_refine_budget_mode
                        == "causal_cross_pareto"
                        else "response_refine_trace_v3"
                        if self.dependency_response_refine_budget_mode
                        == "causal_pareto"
                        else "response_refine_trace_v2"
                        if self.dependency_response_refine_budget_mode == "gated"
                        else "response_refine_trace_v1"
                    )
                    if self.dependency_response_refine
                    else "response_credit_draft_exchange_trace_v1"
                    if self.dependency_draft_exchange
                    else "attention_stability_trace_v2"
                    if extended_dependency_mode
                    else "attention_stability_trace_v1"
                )
            elif self.relaxed_threshold is not None:
                generated_answer, nfe = generate_localleap(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, temperature=0, remasking=self.remasking, mask_id=self.mask_id, early_stop=self.early_stop,
                    threshold=self.threshold, relaxed_threshold=self.relaxed_threshold, radius=self.radius)

            else:
                generated_answer, nfe = generate(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length,
                                        temperature=0, remasking=self.remasking, mask_id=self.mask_id, early_stop=self.early_stop, threshold=self.threshold)

            generated_token_ids_for_diagnostics = generated_answer.detach().to(torch.int32).cpu()
            is_humaneval_request = (
                self.is_instruct
                and 'task_id' in req.doc
                and str(req.doc['task_id']).lower().startswith('humaneval')
            )
            if is_humaneval_request:
                generated_answer = self.tokenizer.decode(generated_answer[0][input_ids.shape[1]:], skip_special_tokens=True)
                generated_answer_ids = torch.tensor(self.tokenizer(generated_answer)["input_ids"])
                if self.show_speed:
                    num_tokens += (generated_answer_ids != 126081).sum()
                    num_nfe += nfe
                    num_instances += 1
            else:
                generated_answer = self.tokenizer.decode(generated_answer[0][input_ids.shape[1]:], skip_special_tokens=False)
                for stop_seq in stop_tokens:
                    if stop_seq in generated_answer:
                        generated_answer = generated_answer.split(stop_seq)[0]

                # remove special tokens
                generated_answer_ids = torch.tensor(self.tokenizer(generated_answer)["input_ids"])
                if self.show_speed:
                    num_tokens += (generated_answer_ids != 126081).sum()
                    num_nfe += nfe
                    num_instances += 1
                generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)

            if draft_candidate_token_ids is not None:
                candidate_generations = decode_candidate_generations(
                    self.tokenizer,
                    draft_candidate_token_ids,
                    input_ids.shape[1],
                    stop_tokens,
                    preserve_full_humaneval_response=is_humaneval_request,
                )
                if public_guard_requested:
                    if public_guard_active:
                        parent_name = decode_diagnostics["selected_name"]
                        guard_name, public_guard_diagnostics = (
                            select_public_example_guard(
                                candidate_generations["baseline"],
                                candidate_generations[parent_name],
                                question,
                                req.doc.get("entry_point"),
                            )
                        )
                        final_name = (
                            "baseline" if guard_name == "baseline" else parent_name
                        )
                        old_token_count = int(
                            (generated_answer_ids != 126081).sum().item()
                        )
                        generated_answer = candidate_generations[final_name]
                        generated_token_ids_for_diagnostics = (
                            draft_candidate_token_ids[final_name]
                            .detach()
                            .to(torch.int32)
                            .cpu()
                        )
                        generated_answer_ids = torch.tensor(
                            self.tokenizer(generated_answer)["input_ids"]
                        )
                        if self.show_speed:
                            num_tokens += int(
                                (generated_answer_ids != 126081).sum().item()
                            ) - old_token_count
                        public_guard_diagnostics["parent_name"] = parent_name
                        public_guard_diagnostics["final_selected_name"] = final_name
                        decode_diagnostics["pre_guard_selected_name"] = parent_name
                        decode_diagnostics["selected_name"] = final_name
                        decode_diagnostics["public_example_guard"] = (
                            public_guard_diagnostics
                        )
                    else:
                        decode_diagnostics["public_example_guard"] = {
                            "selector": "strict_public_example_guard_v2",
                            "status": "skipped_no_public_examples",
                            "uses_hidden_tests": False,
                            "uses_reference_solution": False,
                        }
                if self.dependency_differential_selection:
                    candidate_names = [
                        name
                        for name in ("anchor", "explorer", "repaired")
                        if name in candidate_generations
                    ]
                    selected_index, differential_diagnostics = select_differential_candidate(
                        [candidate_generations[name] for name in candidate_names],
                        question,
                        req.doc.get("entry_point"),
                    )
                    selected_name = candidate_names[selected_index]
                    old_token_count = int((generated_answer_ids != 126081).sum().item())
                    generated_answer = candidate_generations[selected_name]
                    generated_token_ids_for_diagnostics = draft_candidate_token_ids[
                        selected_name
                    ].detach().to(torch.int32).cpu()
                    generated_answer_ids = torch.tensor(
                        self.tokenizer(generated_answer)["input_ids"]
                    )
                    if self.show_speed:
                        num_tokens += int((generated_answer_ids != 126081).sum().item()) - old_token_count
                    differential_diagnostics["selected_name"] = selected_name
                    decode_diagnostics["differential_selection"] = differential_diagnostics
            output.append(generated_answer)

            diagnostics_path = None
            diagnostics_bytes = None
            stable_task_id = (
                req.doc.get("task_id")
                or req.doc.get("id")
                or req.doc.get("problem_id")
                or req.doc.get("unique_id")
                or f"index_{i}"
            )
            raw_gold_for_trace = (
                req.doc.get("canonical_solution")
                or req.doc.get("code")
                or req.doc.get("answer")
                or req.doc.get("target")
            )
            if step_records is not None and active_diagnostics_dir is not None:
                os.makedirs(active_diagnostics_dir, exist_ok=True)
                task_id = str(stable_task_id)
                safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)
                diagnostics_path = os.path.join(
                    active_diagnostics_dir,
                    f"{i:03d}_{safe_task_id}.pt",
                )
                temporary_path = diagnostics_path + ".tmp"
                diagnostics_payload = {
                    "schema_version": step_diagnostics_schema,
                    "absolute_index": i,
                    "task_id": stable_task_id,
                    "prompt_hash": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                    "prompt_text": question,
                    "prompt_token_ids": input_ids[0].to(torch.int32).cpu(),
                    "prompt_length": int(input_ids.shape[1]),
                    "raw_gold": raw_gold_for_trace,
                    "entry_point": req.doc.get("entry_point"),
                    "test_code": req.doc.get("test"),
                    "final_sequence_token_ids": generated_token_ids_for_diagnostics[0],
                    "decoded_generation": generated_answer,
                    "generation_settings": {
                        "model_path": self.model_path,
                        "steps": self.steps,
                        "gen_length": self.gen_length,
                        "block_length": self.block_length,
                        "temperature": 0,
                        "remasking": self.remasking,
                        "mask_id": self.mask_id,
                        "dependency_threshold": self.dependency_threshold,
                        "dependency_mode": self.dependency_mode,
                        "dependency_temporal_mode": self.dependency_temporal_mode,
                        "dependency_temporal_topk": self.dependency_temporal_topk,
                        "dependency_prune_stable_conflicts": self.dependency_prune_stable_conflicts,
                        "dependency_fill_budget": self.dependency_fill_budget,
                        "dependency_likelihood_selection": self.dependency_likelihood_selection,
                        "dependency_likelihood_selection_mode": self.dependency_likelihood_selection_mode,
                        "dependency_draft_exchange": self.dependency_draft_exchange,
                        "dependency_differential_selection": self.dependency_differential_selection,
                        "dependency_response_refine": self.dependency_response_refine,
                        "dependency_response_refine_budget_mode": self.dependency_response_refine_budget_mode,
                        "candidate_memory_topk": self.candidate_memory_topk,
                        "candidate_memory_confidence_threshold": self.candidate_memory_confidence_threshold,
                        "candidate_memory_fallback": self.candidate_memory_fallback,
                        "candidate_memory_exact_jsd": self.candidate_memory_exact_jsd,
                        "stcc_mode": self.stcc_mode,
                        "stcc_topk": self.stcc_topk,
                        "stcc_jsd_threshold": self.stcc_jsd_threshold,
                        "stcc_attention_threshold": self.stcc_attention_threshold,
                        "stcc_extra_multiplier": self.stcc_extra_multiplier,
                        "stcc_extra_jsd_threshold": self.stcc_extra_jsd_threshold,
                        "stcc_min_topk_overlap": self.stcc_min_topk_overlap,
                        "stcc_min_stability_streak": self.stcc_min_stability_streak,
                    },
                    "decode_summary": decode_diagnostics,
                    "steps": step_records,
                }
                torch.save(diagnostics_payload, temporary_path)
                os.replace(temporary_path, diagnostics_path)
                diagnostics_bytes = os.path.getsize(diagnostics_path)

            if decode_diagnostics is not None and active_trace_dir is not None:
                os.makedirs(active_trace_dir, exist_ok=True)
                trace_path = os.path.join(active_trace_dir, f"rank_{self.rank}.jsonl")
                trace_record = {
                    "absolute_index": i,
                    "task_id": stable_task_id,
                    "prompt_hash": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                    "prompt_text": question,
                    "entry_point": req.doc.get("entry_point"),
                    "raw_gold": raw_gold_for_trace,
                    "normalized_gold": raw_gold_for_trace,
                    "decoded_generation": generated_answer,
                    "correct": None,
                    "nfe": nfe,
                    "generation_settings": {
                        "model_path": self.model_path if hasattr(self, "model_path") else None,
                        "steps": self.steps,
                        "gen_length": self.gen_length,
                        "block_length": self.block_length,
                        "temperature": 0,
                        "remasking": self.remasking,
                        "mask_id": self.mask_id,
                        "dependency_threshold": self.dependency_threshold,
                        "dependency_mode": self.dependency_mode,
                        "dependency_temporal_mode": self.dependency_temporal_mode,
                        "dependency_temporal_topk": self.dependency_temporal_topk,
                        "dependency_prune_stable_conflicts": self.dependency_prune_stable_conflicts,
                        "dependency_fill_budget": self.dependency_fill_budget,
                        "dependency_likelihood_selection": self.dependency_likelihood_selection,
                        "dependency_likelihood_selection_mode": self.dependency_likelihood_selection_mode,
                        "dependency_draft_exchange": self.dependency_draft_exchange,
                        "dependency_differential_selection": self.dependency_differential_selection,
                        "dependency_response_refine": self.dependency_response_refine,
                        "dependency_response_refine_budget_mode": self.dependency_response_refine_budget_mode,
                        "candidate_memory_topk": self.candidate_memory_topk,
                        "candidate_memory_confidence_threshold": self.candidate_memory_confidence_threshold,
                        "candidate_memory_fallback": self.candidate_memory_fallback,
                        "candidate_memory_exact_jsd": self.candidate_memory_exact_jsd,
                        "stcc_mode": self.stcc_mode,
                        "stcc_topk": self.stcc_topk,
                        "stcc_jsd_threshold": self.stcc_jsd_threshold,
                        "stcc_attention_threshold": self.stcc_attention_threshold,
                        "stcc_extra_multiplier": self.stcc_extra_multiplier,
                        "stcc_extra_jsd_threshold": self.stcc_extra_jsd_threshold,
                        "stcc_min_topk_overlap": self.stcc_min_topk_overlap,
                        "stcc_min_stability_streak": self.stcc_min_stability_streak,
                    },
                    "evaluator_version": trace_evaluator_version,
                    "seed": {"lm_eval_random": 0, "numpy": 1234, "torch": 1234, "fewshot": 1234},
                    "decode_diagnostics": decode_diagnostics,
                    "candidate_generations": candidate_generations,
                    "step_diagnostics_schema": (
                        step_diagnostics_schema if diagnostics_path is not None else None
                    ),
                    "step_diagnostics_path": diagnostics_path,
                    "step_diagnostics_bytes": diagnostics_bytes,
                }
                with open(trace_path, "a", encoding="utf-8") as trace_file:
                    trace_file.write(json.dumps(trace_record, ensure_ascii=False) + "\n")

            if self.save_dir is not None:
                with open(save_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(generated_answer, ensure_ascii=False) + '\n')

            print('=' * 20)
            print('question: ', question)
            print('answer: ', generated_answer)
            print('=' * 20, end='\n\n')

        end_time = time.time()

        del self.model
        torch.cuda.empty_cache()

        if self.show_speed:
            print(f"Total data items: {num_instances}")
            print(f"Total number of input tokens: {num_input_tokens}")
            print(f"Total number of generated tokens: {num_tokens}")
            print(f"Total time: {end_time - start_time:.3f} seconds")
            print(f"Total NFE: {num_nfe}")
            print(f"Average number of input tokens: {num_input_tokens / num_instances:.3f}")
            print(f"Average number of generated tokens: {num_tokens / num_instances:.3f}")
            print(f"Tokens per second: {num_tokens / (end_time - start_time):.3f}")
            print(f"Tokens per step: {num_tokens / num_nfe:.3f}")

            if self.integrate_speed and dist.is_available() and dist.is_initialized():
                # gather values
                tokens_input_list = all_gather_cpu(num_input_tokens)
                tokens_gen_list = all_gather_cpu(num_tokens)
                time_list = all_gather_cpu(end_time - start_time)
                nfe_list = all_gather_cpu(num_nfe)
                instances_list = all_gather_cpu(num_instances)

                if dist.get_rank() == 0:
                    total_tokens_input = sum(tokens_input_list)
                    total_tokens_gen = sum(tokens_gen_list)
                    total_time = sum(time_list)
                    total_nfe = sum(nfe_list)
                    total_instances = sum(instances_list)

                    print(f"[All GPUs] Total data items: {total_instances}")
                    print(f"[All GPUs] Total number of input tokens: {total_tokens_input}")
                    print(f"[All GPUs] Total number of generated tokens: {total_tokens_gen}")
                    print(f"[All GPUs] Total time: {total_time:.3f} seconds")
                    print(f"[All GPUs] Total NFE: {total_nfe}")
                    print(f"[All GPUs] Average number of input tokens: {total_tokens_input / total_instances:.3f}")
                    print(f"[All GPUs] Average number of generated tokens: {total_tokens_gen / total_instances:.3f}")
                    print(f"[All GPUs] Average NFE: {total_nfe / total_instances:.3f}")
                    print(f"[All GPUs] Tokens per second: {total_tokens_gen / total_time:.3f}")
                    print(f"[All GPUs] Tokens per step: {total_tokens_gen / total_nfe:.3f}")
        return output


if __name__ == "__main__":
    set_seed(42)
    cli_evaluate()
