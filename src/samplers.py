"""Samplers: LCR baseline, RCR, and trajectory-guided scheduling."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch

from src.distribution import _topk_l1, confidence, kl_divergence


MASK_ID = 126336
TrackMode = Literal["none", "light", "full"]


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@dataclass
class TrajectoryState:
    """Per-position distribution trajectory state."""
    mode: TrackMode = "light"
    first_logits: dict[int, torch.Tensor] = field(default_factory=dict)
    prev_logits: dict[int, torch.Tensor] = field(default_factory=dict)
    path_length: dict[int, float] = field(default_factory=dict)
    net_anchor: dict[int, torch.Tensor] = field(default_factory=dict)
    first_conf: dict[int, float] = field(default_factory=dict)
    flip_count: dict[int, int] = field(default_factory=dict)
    last_top1: dict[int, int] = field(default_factory=dict)
    recent_conf: dict[int, list[float]] = field(default_factory=dict)
    running_max_conf: dict[int, float] = field(default_factory=dict)
    coupling: dict[tuple[int, int], float] = field(default_factory=dict)
    step_records: list[dict[str, Any]] = field(default_factory=list)
    last_step_conf: dict[int, float] = field(default_factory=dict)
    pending_commits: list[int] = field(default_factory=list)

    def update_light(self, pos: int, logits: torch.Tensor):
        top1 = int(logits.argmax().item())
        conf = confidence(logits, top1)
        if pos not in self.first_conf:
            self.first_conf[pos] = conf
            self.path_length[pos] = 0.0
            self.flip_count[pos] = 0
            self.last_top1[pos] = top1
            self.running_max_conf[pos] = conf
            self.recent_conf[pos] = [conf]
            return

        prev_conf = self.recent_conf[pos][-1]
        self.path_length[pos] = self.path_length.get(pos, 0.0) + abs(conf - prev_conf)
        if top1 != self.last_top1[pos]:
            self.flip_count[pos] = self.flip_count.get(pos, 0) + 1
        self.last_top1[pos] = top1
        self.running_max_conf[pos] = max(self.running_max_conf.get(pos, 0.0), conf)
        self.recent_conf[pos].append(conf)

    def update_full(self, pos: int, logits: torch.Tensor, top_k: int = 8):
        top1 = int(logits.argmax().item())
        conf = confidence(logits, top1)
        if pos not in self.first_logits:
            self.first_logits[pos] = logits.detach().cpu()
            self.first_conf[pos] = conf
            self.path_length[pos] = 0.0
            self.flip_count[pos] = 0
            self.last_top1[pos] = top1
            self.running_max_conf[pos] = conf
            self.recent_conf[pos] = [conf]
            self.net_anchor[pos] = logits.detach().cpu()
            self.prev_logits[pos] = logits.detach().cpu()
            return

        kl_step = _topk_l1(self.prev_logits[pos], logits.detach().cpu(), k=top_k)
        self.path_length[pos] = self.path_length.get(pos, 0.0) + kl_step
        self.net_anchor[pos] = logits.detach().cpu()
        if top1 != self.last_top1[pos]:
            self.flip_count[pos] = self.flip_count.get(pos, 0) + 1
        self.last_top1[pos] = top1
        self.running_max_conf[pos] = max(self.running_max_conf.get(pos, 0.0), conf)
        self.recent_conf[pos].append(conf)
        self.prev_logits[pos] = logits.detach().cpu()

    def record_commit_response_light(self, current_conf: dict[int, float]):
        if not self.pending_commits or not self.last_step_conf:
            self.pending_commits = []
            return
        for pos, conf_now in current_conf.items():
            if pos in self.pending_commits or pos not in self.last_step_conf:
                continue
            delta = abs(conf_now - self.last_step_conf[pos])
            if delta < 1e-4:
                continue
            for cpos in self.pending_commits:
                key = (cpos, pos)
                self.coupling[key] = self.coupling.get(key, 0.0) + delta
        self.pending_commits = []

    def record_commit_response_full(self, current_logits: dict[int, torch.Tensor], prev_logits: dict[int, torch.Tensor], top_k: int = 8):
        if not self.pending_commits or not prev_logits:
            self.pending_commits = []
            return
        for pos, logits_now in current_logits.items():
            if pos in self.pending_commits or pos not in prev_logits:
                continue
            delta = _topk_l1(prev_logits[pos], logits_now, k=top_k)
            if delta < 1e-4:
                continue
            for cpos in self.pending_commits:
                key = (cpos, pos)
                self.coupling[key] = self.coupling.get(key, 0.0) + delta
        self.pending_commits = []

    def convergence_score(self, pos: int) -> float:
        if pos not in self.first_conf:
            return 0.0
        recent = self.recent_conf.get(pos, [0.0])
        first = self.first_conf[pos]
        last = recent[-1]
        net = max(last - first, 0.0)
        path = self.path_length.get(pos, 0.0)
        ratio = net / (path + 1e-6)
        stab = 1.0 / (1.0 + float(np.std(recent[-4:]))) if len(recent) >= 2 else 0.5
        flips = self.flip_count.get(pos, 0)
        oscillation_penalty = min(flips * 0.25, 1.0)
        return max(ratio * stab + last * 0.5 - oscillation_penalty, 0.0)

    def trajectory_type(self, pos: int) -> str:
        flips = self.flip_count.get(pos, 0)
        recent = self.recent_conf.get(pos, [0.0])
        if len(recent) < 2:
            return "unknown"
        trend = recent[-1] - recent[0]
        if flips >= 2:
            return "oscillating"
        if trend > 0.05 and flips <= 1:
            return "converging"
        if abs(trend) < 0.02 and float(np.std(recent)) < 0.02:
            return "frozen"
        return "mixed"

    def coupling_penalty(self, pos_i: int, pos_j: int) -> float:
        return self.coupling.get((pos_i, pos_j), 0.0) + self.coupling.get((pos_j, pos_i), 0.0)


def _model_forward(model, x, attention_mask=None, cfg_scale=0.0, prompt_index=None):
    if cfg_scale > 0.0:
        un_x = x.clone()
        un_x[prompt_index] = MASK_ID
        x_ = torch.cat([x, un_x], dim=0)
        if attention_mask is not None:
            attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
        logits = model(x_, attention_mask=attention_mask_).logits
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
    else:
        logits = model(x, attention_mask=attention_mask).logits
    return logits


def _select_transfer_indices(
    masked_positions: list[int],
    k: int,
    scores: dict[int, float],
    traj: TrajectoryState | None = None,
    lateral: bool = False,
    coupling_threshold: float = 0.08,
) -> list[int]:
    ranked = sorted(masked_positions, key=lambda p: scores.get(p, -1e9), reverse=True)
    if not lateral or traj is None or k <= 1:
        return ranked[:k]

    selected: list[int] = []
    for pos in ranked:
        if len(selected) >= k:
            break
        if all(traj.coupling_penalty(pos, s) < coupling_threshold for s in selected):
            selected.append(pos)
    if len(selected) < k:
        for pos in ranked:
            if pos not in selected:
                selected.append(pos)
            if len(selected) >= k:
                break
    return selected


def _resolve_track_mode(sampler: str, track_trajectory: bool, lateral: bool) -> TrackMode:
    if track_trajectory:
        return "full"
    if lateral and sampler == "traj":
        return "light"
    if sampler in ("traj", "rcr"):
        return "light"
    return "none"


@torch.no_grad()
def generate_with_sampler(
    model,
    prompt,
    attention_mask=None,
    steps: int = 64,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    mask_id: int = MASK_ID,
    sampler: str = "lcr",
    track_trajectory: bool = False,
    top_k_track: int = 8,
    lateral: bool = False,
    coupling_threshold: float = 0.08,
):
    device = prompt.device
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, : prompt.shape[1]] = prompt.clone()
    if attention_mask is not None:
        attention_mask = torch.cat(
            [attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=device)],
            dim=-1,
        )
    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    track_mode = _resolve_track_mode(sampler, track_trajectory, lateral)
    traj = TrajectoryState(mode=track_mode) if track_mode != "none" else None
    nfe = 0
    prev_logits_map: dict[int, torch.Tensor] = {}

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = x == mask_id
            logits = _model_forward(model, x, attention_mask, cfg_scale, prompt_index)
            nfe += 1

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            masked_positions = [int(p) for p in torch.nonzero(mask_index[0], as_tuple=False).flatten().tolist()]
            current_conf = {}
            current_logits = {}

            for pos in masked_positions:
                pos_logits = logits[0, pos]
                pred = int(x0[0, pos].item())
                current_conf[pos] = confidence(pos_logits, pred)
                if track_mode == "full":
                    current_logits[pos] = pos_logits.detach().cpu()

            if traj is not None:
                if track_mode == "full":
                    traj.record_commit_response_full(current_logits, prev_logits_map, top_k=top_k_track)
                    for pos in masked_positions:
                        traj.update_full(pos, logits[0, pos], top_k=top_k_track)
                    prev_logits_map = current_logits
                else:
                    traj.record_commit_response_light(current_conf)
                    for pos in masked_positions:
                        traj.update_light(pos, logits[0, pos])
                    traj.last_step_conf = current_conf

            scores: dict[int, float] = {}
            for pos in masked_positions:
                pos_logits = logits[0, pos]
                pred = int(x0[0, pos].item())
                cur = confidence(pos_logits, pred)
                if sampler == "lcr":
                    scores[pos] = cur
                elif sampler == "rcr":
                    scores[pos] = max(traj.running_max_conf.get(pos, cur), cur) if traj else cur
                elif sampler == "traj":
                    conv = traj.convergence_score(pos) if traj else 0.0
                    cur = confidence(pos_logits, pred)
                    rmax = traj.running_max_conf.get(pos, cur) if traj else cur
                    flips = traj.flip_count.get(pos, 0) if traj else 0
                    recent = traj.recent_conf.get(pos, [cur]) if traj else [cur]
                    ttype = traj.trajectory_type(pos) if traj else "mixed"
                    if ttype == "converging":
                        stability = 1.0 / (1.0 + 0.35 * flips)
                        recent_stable = 1.0 if len(recent) >= 3 and float(np.std(recent[-3:])) < 0.03 else 0.0
                        scores[pos] = (0.45 * rmax + 0.30 * cur + 0.15 * conv) * stability + 0.10 * recent_stable
                    else:
                        # Fallback to baseline-like scoring when trajectory is ambiguous
                        scores[pos] = 0.75 * cur + 0.25 * rmax
                else:
                    raise ValueError(sampler)

            k = int(num_transfer_tokens[0, i].item())
            selected = _select_transfer_indices(
                masked_positions, k, scores, traj=traj,
                lateral=(lateral and sampler == "traj"),
                coupling_threshold=coupling_threshold,
            )

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for pos in selected:
                transfer_index[0, pos] = True

            x0 = torch.where(mask_index, x0, x)
            x[transfer_index] = x0[transfer_index]

            if traj is not None:
                traj.pending_commits = selected.copy()
                traj.step_records.append({
                    "step": i + num_block * steps_per_block,
                    "committed": selected,
                    "num_masked": len(masked_positions),
                })

    return {"tokens": x, "nfe": nfe, "trajectory": traj}
