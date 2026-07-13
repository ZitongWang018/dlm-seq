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

import json
import hashlib
import time
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
            if self.dependency_threshold is not None:
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
                )
            elif self.relaxed_threshold is not None:
                generated_answer, nfe = generate_localleap(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, temperature=0, remasking=self.remasking, mask_id=self.mask_id, early_stop=self.early_stop, 
                    threshold=self.threshold, relaxed_threshold=self.relaxed_threshold, radius=self.radius)

            else:
                generated_answer, nfe = generate(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, 
                                        temperature=0, remasking=self.remasking, mask_id=self.mask_id, early_stop=self.early_stop, threshold=self.threshold)

            if self.is_instruct and 'task_id' in req.doc and str(req.doc['task_id']).lower().startswith('humaneval'):
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
            output.append(generated_answer)

            if decode_diagnostics is not None and self.dependency_trace_dir is not None:
                os.makedirs(self.dependency_trace_dir, exist_ok=True)
                trace_path = os.path.join(self.dependency_trace_dir, f"rank_{self.rank}.jsonl")
                trace_record = {
                    "absolute_index": i,
                    "task_id": req.doc.get("task_id"),
                    "prompt_hash": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                    "raw_gold": req.doc.get("canonical_solution"),
                    "normalized_gold": req.doc.get("canonical_solution"),
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
                        "dependency_threshold": self.dependency_threshold,
                    },
                    "evaluator_version": "attention_stability_trace_v1",
                    "seed": {"lm_eval_random": 0, "numpy": 1234, "torch": 1234, "fewshot": 1234},
                    "decode_diagnostics": decode_diagnostics,
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
