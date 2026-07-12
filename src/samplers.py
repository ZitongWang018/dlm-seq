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


def response_budget_transfer_count(
    remaining_tokens: int,
    remaining_steps: int,
    response_risk: float,
    threshold: float,
    factor: float,
) -> int:
    """Choose a smaller commit batch after a strong context response."""
    base = max(1, (remaining_tokens + remaining_steps - 1) // remaining_steps)
    if remaining_steps <= 1 or base <= 1 or response_risk < threshold:
        return base
    return max(1, int(np.floor(base * factor)))


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


def _topk_state(logits: torch.Tensor, k: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits.float(), dim=-1)
    vals, ids = torch.topk(probs, k=min(k, probs.shape[-1]))
    return ids.detach().cpu(), vals.detach().cpu()


def _topk_states_for_positions(
    logits: torch.Tensor,
    positions: list[int],
    k: int = 8,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    if not positions:
        return {}
    pos_tensor = torch.tensor(positions, device=logits.device, dtype=torch.long)
    pos_logits = logits.index_select(0, pos_tensor)
    probs = torch.softmax(pos_logits.float(), dim=-1)
    vals, ids = torch.topk(probs, k=min(k, probs.shape[-1]), dim=-1)
    ids = ids.detach().cpu()
    vals = vals.detach().cpu()
    return {pos: (ids[i], vals[i]) for i, pos in enumerate(positions)}


def _topk_state_l1(
    prev_state: tuple[torch.Tensor, torch.Tensor],
    cur_state: tuple[torch.Tensor, torch.Tensor],
) -> float:
    prev_ids, prev_vals = prev_state
    cur_ids, cur_vals = cur_state
    vals: dict[int, list[float]] = {}
    for tok, val in zip(prev_ids.tolist(), prev_vals.tolist()):
        vals[int(tok)] = [float(val), 0.0]
    for tok, val in zip(cur_ids.tolist(), cur_vals.tolist()):
        vals.setdefault(int(tok), [0.0, 0.0])[1] = float(val)
    return float(sum(abs(a - b) for a, b in vals.values()))


def _topk_path_efficiency(
    previous: tuple[torch.Tensor, torch.Tensor],
    current: tuple[torch.Tensor, torch.Tensor],
    proposed: tuple[torch.Tensor, torch.Tensor],
) -> tuple[float, float]:
    """Measure whether a proposed context change continues or reverses a trajectory."""
    before = _topk_state_l1(previous, current)
    after = _topk_state_l1(current, proposed)
    net = _topk_state_l1(previous, proposed)
    return net / (before + after + 1e-8), before


def _select_transfer_indices(
    masked_positions: list[int],
    k: int,
    scores: dict[int, float],
    traj: TrajectoryState | None = None,
    lateral: bool = False,
    coupling_threshold: float = 0.08,
    local_spacing: int = 0,
) -> list[int]:
    ranked = sorted(masked_positions, key=lambda p: scores.get(p, -1e9), reverse=True)
    if local_spacing > 0:
        selected: list[int] = []
        for pos in ranked:
            if len(selected) >= k:
                break
            if all(abs(pos - s) > local_spacing for s in selected):
                selected.append(pos)
        if len(selected) < k:
            for pos in ranked:
                if pos not in selected:
                    selected.append(pos)
                if len(selected) >= k:
                    break
        return selected

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
    local_spacing: int = 0,
    local_spacing_until: float = 1.0,
    response_weight: float = 0.1,
    response_cap: float = 1.0,
    response_conf_max: float = 0.85,
    response_min_delta: float = 0.0,
    response_local_window: int = 4,
    response_refresh_threshold: float = 0.30,
    response_lookahead_threshold: float = 0.40,
    response_lookahead_max_steps: int = 2,
    response_budget_threshold: float = 0.35,
    response_budget_factor: float = 0.5,
    response_persistence_max_drift: float = 0.15,
    wavefront_size: int = 8,
    wavefront_radius: int = 4,
    terminal_refine_tokens: int = 4,
    terminal_refine_threshold: float = 0.35,
    refine_max_tokens: int = 24,
    refine_radius: int = 1,
    branch_radius: int = 4,
    branch_tail_window: int = 48,
    eos_token_id: int | None = None,
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
    prev_topk_map: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    prev_committed_for_response: list[int] = []
    diagnostics: dict[str, Any] = {
        "response_steps": 0,
        "response_max_delta": 0.0,
        "response_mean_delta_sum": 0.0,
        "response_selected_delta_sum": 0.0,
        "response_selected_count": 0,
        "response_delay_candidate_count": 0,
        "response_delay_selected_count": 0,
        "response_delayed_count": 0,
        "response_near_delta_sum": 0.0,
        "response_near_delta_count": 0,
        "response_far_delta_sum": 0.0,
        "response_far_delta_count": 0,
        "response_confirmed_count": 0,
        "rewrite_event_count": 0,
        "refine_masked_count": 0,
        "refine_changed_count": 0,
        "branch_event_delta": 0.0,
        "branch_changed_count": 0,
        "effective_gen_length": gen_length,
        "branch_seed_relative": -1,
        "response_refresh_steps": 0,
        "response_refresh_tokens": 0,
        "response_lookahead_steps": 0,
        "response_lookahead_margin_sum": 0.0,
        "response_budget_steps": 0,
        "response_budget_deferred_tokens": 0,
        "response_persistence_candidates": 0,
        "response_persistence_confirmed": 0,
        "response_alignment_steps": 0,
        "response_alignment_margin_sum": 0.0,
        "response_alignment_winner_sum": 0.0,
        "response_alignment_runner_up_sum": 0.0,
        "block_summaries": [],
        "wavefront_expansion_steps": 0,
        "wavefront_narrow_steps": 0,
        "terminal_refine_triggered": 0,
        "terminal_refine_changed_count": 0,
    }
    diag_last_top1: dict[int, int] = {}
    diag_flip_count: dict[int, int] = {}
    pending_rewrites: dict[int, int] = {}
    pending_response_states: dict[int, tuple[tuple[torch.Tensor, torch.Tensor], float]] = {}
    rewrite_count: dict[int, int] = {}
    rewrite_max_delta: dict[int, float] = {}
    strongest_rewrite: tuple[float, int, int] | None = None
    rewrite_events: list[tuple[float, int, int]] = []
    gen_start = prompt.shape[1]
    gen_end = gen_start + gen_length
    wavefront = set(range(gen_start, min(gen_start + wavefront_size, gen_end)))
    previous_near_response: float | None = None

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_counter_start = {
            key: diagnostics[key]
            for key in (
                "response_steps", "response_selected_delta_sum", "response_selected_count",
                "response_near_delta_sum", "response_near_delta_count",
            )
        }
        block_flip_start = dict(diag_flip_count)
        block_near_response_trace: list[float] = []
        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = x == mask_id
            logits = _model_forward(model, x, attention_mask, cfg_scale, prompt_index)
            nfe += 1

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            masked_positions = [int(p) for p in torch.nonzero(mask_index[0], as_tuple=False).flatten().tolist()]
            selectable_positions = [
                pos for pos in masked_positions if block_start <= pos < block_end
            ]
            current_conf = {}
            current_logits = {}
            response_delta: dict[int, float] = {}
            step_near_responses: list[float] = []
            needs_topk = sampler in (
                "lcr_probe",
                "lcr_response",
                "lcr_response_gated",
                "lcr_response_strong",
                "lcr_response_delay",
                "lcr_response_flip_delay",
                "lcr_response_stability_delay",
                "lcr_response_refresh",
                "lcr_response_lookahead",
                "lcr_response_budget",
                "lcr_response_persistence",
                "lcr_response_alignment",
                "lcr_response_wavefront",
                "lcr_terminal_refine",
                "lcr_response_confirm",
                "lcr_rewrite_refine",
                "lcr_rewrite_branch",
            )
            current_topk = _topk_states_for_positions(logits[0], masked_positions, k=top_k_track) if needs_topk else {}

            for pos in masked_positions:
                pos_logits = logits[0, pos]
                pred = int(x0[0, pos].item())
                current_conf[pos] = confidence(pos_logits, pred)
                if pos in diag_last_top1 and diag_last_top1[pos] != pred:
                    diag_flip_count[pos] = diag_flip_count.get(pos, 0) + 1
                else:
                    diag_flip_count.setdefault(pos, 0)
                diag_last_top1[pos] = pred
                if track_mode == "full":
                    current_logits[pos] = pos_logits.detach().cpu()

            if sampler in (
                "lcr_probe",
                "lcr_response",
                "lcr_response_gated",
                "lcr_response_strong",
                "lcr_response_delay",
                "lcr_response_flip_delay",
                "lcr_response_stability_delay",
                "lcr_response_refresh",
                "lcr_response_lookahead",
                "lcr_response_budget",
                "lcr_response_persistence",
                "lcr_response_alignment",
                "lcr_response_wavefront",
                "lcr_terminal_refine",
                "lcr_response_confirm",
                "lcr_rewrite_refine",
                "lcr_rewrite_branch",
            ) and prev_committed_for_response and prev_topk_map:
                for pos in masked_positions:
                    if pos in prev_committed_for_response or pos not in prev_topk_map:
                        continue
                    response_delta[pos] = _topk_state_l1(prev_topk_map[pos], current_topk[pos])
                if response_delta:
                    vals = list(response_delta.values())
                    diagnostics["response_steps"] += 1
                    diagnostics["response_max_delta"] = max(diagnostics["response_max_delta"], max(vals))
                    diagnostics["response_mean_delta_sum"] += float(np.mean(vals))
                    for pos, delta in response_delta.items():
                        is_near = response_local_window <= 0 or any(
                            abs(pos - cpos) <= response_local_window for cpos in prev_committed_for_response
                        )
                        key = "response_near" if is_near else "response_far"
                        diagnostics[f"{key}_delta_sum"] += delta
                        diagnostics[f"{key}_delta_count"] += 1
                        if is_near:
                            step_near_responses.append(delta)
                        prev_state = prev_topk_map.get(pos)
                        cur_state = current_topk.get(pos)
                        top1_changed = (
                            prev_state is not None
                            and cur_state is not None
                            and int(prev_state[0][0]) != int(cur_state[0][0])
                        )
                        if sampler in ("lcr_rewrite_refine", "lcr_rewrite_branch") and is_near and top1_changed:
                            rewrite_count[pos] = rewrite_count.get(pos, 0) + 1
                            rewrite_max_delta[pos] = max(rewrite_max_delta.get(pos, 0.0), delta)
                            diagnostics["rewrite_event_count"] += 1
                            old_top1 = int(prev_state[0][0])
                            event = (delta, pos, old_top1)
                            rewrite_events.append(event)
                            if strongest_rewrite is None or event[0] > strongest_rewrite[0]:
                                strongest_rewrite = event

            block_near_response_trace.append(
                float(np.mean(step_near_responses)) if step_near_responses else 0.0
            )

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
            delay_candidates: set[int] = set()
            new_rewrites: dict[int, int] = {}
            for pos in masked_positions:
                pos_logits = logits[0, pos]
                pred = int(x0[0, pos].item())
                cur = confidence(pos_logits, pred)
                if sampler in (
                    "lcr", "lcr_probe", "lcr_spaced", "lcr_spaced_early", "lcr_wavefront", "lcr_response_wavefront", "lcr_response_refresh", "lcr_response_lookahead", "lcr_response_budget", "lcr_response_persistence", "lcr_response_alignment", "lcr_terminal_refine",
                    "lcr_rewrite_refine", "lcr_rewrite_branch",
                ):
                    scores[pos] = cur
                    if sampler == "lcr_response_persistence" and pos in pending_response_states:
                        prior_state, prior_delta = pending_response_states[pos]
                        current_state = current_topk.get(pos)
                        if current_state is not None:
                            drift = _topk_state_l1(prior_state, current_state)
                            same_top1 = int(prior_state[0][0]) == int(current_state[0][0])
                            if same_top1 and drift <= response_persistence_max_drift:
                                retention = 1.0 - drift / max(response_persistence_max_drift, 1e-6)
                                scores[pos] += response_weight * min(prior_delta, response_cap) * retention
                                diagnostics["response_persistence_confirmed"] += 1
                elif sampler in (
                    "lcr_response", "lcr_response_gated", "lcr_response_strong",
                    "lcr_response_delay", "lcr_response_flip_delay",
                    "lcr_response_stability_delay",
                    "lcr_response_confirm",
                ):
                    allow_boost = sampler == "lcr_response" or cur <= response_conf_max
                    delta = response_delta.get(pos, 0.0)
                    if sampler == "lcr_response_strong":
                        allow_boost = delta >= response_min_delta
                    if sampler in (
                        "lcr_response_delay",
                        "lcr_response_flip_delay",
                        "lcr_response_stability_delay",
                        "lcr_response_confirm",
                    ):
                        near_recent_commit = (
                            response_local_window <= 0
                            or any(abs(pos - cpos) <= response_local_window for cpos in prev_committed_for_response)
                        )
                        should_delay = (
                            near_recent_commit
                            and delta >= response_min_delta
                            and cur <= response_conf_max
                        )
                        if sampler in (
                            "lcr_response_flip_delay",
                            "lcr_response_stability_delay",
                            "lcr_response_confirm",
                        ):
                            prev_state = prev_topk_map.get(pos)
                            cur_state = current_topk.get(pos)
                            top1_changed = (
                                prev_state is not None
                                and cur_state is not None
                                and int(prev_state[0][0]) != int(cur_state[0][0])
                            )
                            should_delay = should_delay and top1_changed
                        if sampler == "lcr_response_stability_delay":
                            should_delay = should_delay and diag_flip_count.get(pos, 0) > 0
                        if sampler == "lcr_response_confirm":
                            confirmed = pending_rewrites.get(pos) == pred
                            if confirmed:
                                diagnostics["response_confirmed_count"] += 1
                                scores[pos] = cur + response_weight * response_cap
                            else:
                                penalty = min(delta, response_cap) * response_weight if should_delay else 0.0
                                scores[pos] = cur - penalty
                                if should_delay:
                                    delay_candidates.add(pos)
                                    new_rewrites[pos] = pred
                        else:
                            penalty = min(delta, response_cap) * response_weight if should_delay else 0.0
                            if should_delay:
                                delay_candidates.add(pos)
                            scores[pos] = cur - penalty
                    else:
                        boost = min(delta, response_cap) * response_weight if allow_boost else 0.0
                        scores[pos] = cur + boost
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

            remaining_steps = steps_per_block - i
            remaining_in_block = int((x[:, block_start:block_end] == mask_id).sum().item())
            k = response_budget_transfer_count(
                remaining_in_block, remaining_steps, 0.0,
                response_budget_threshold, response_budget_factor,
            )
            global_step = i + num_block * steps_per_block
            use_local_spacing = sampler == "lcr_spaced" or (
                sampler == "lcr_spaced_early"
                and global_step < int(steps * local_spacing_until)
            )
            if sampler in ("lcr_wavefront", "lcr_response_wavefront"):
                scoped = [pos for pos in selectable_positions if pos in wavefront]
                ranked_scoped = sorted(scoped, key=lambda pos: scores.get(pos, -1e9), reverse=True)
                if len(ranked_scoped) < k:
                    ranked_global = sorted(
                        selectable_positions, key=lambda pos: scores.get(pos, -1e9), reverse=True
                    )
                    ranked_scoped.extend(pos for pos in ranked_global if pos not in ranked_scoped)
                selected = ranked_scoped[:k]
            else:
                selected = _select_transfer_indices(
                    selectable_positions, k, scores, traj=traj,
                    lateral=(lateral and sampler == "traj"),
                    coupling_threshold=coupling_threshold,
                    local_spacing=local_spacing if use_local_spacing else 0,
                )
            if sampler == "lcr_response_budget" and len(selected) > 1 and remaining_steps > 1:
                selected_deltas = [response_delta[pos] for pos in selected if pos in response_delta]
                selected_risk = float(np.mean(selected_deltas)) if selected_deltas else 0.0
                reduced_k = response_budget_transfer_count(
                    remaining_in_block, remaining_steps, selected_risk,
                    response_budget_threshold, response_budget_factor,
                )
                if reduced_k < k:
                    selected = selected[:reduced_k]
                    diagnostics["response_budget_steps"] += 1
                    diagnostics["response_budget_deferred_tokens"] += k - reduced_k
            if (
                sampler == "lcr_response_alignment"
                and len(selected) > 1
                and diagnostics["response_alignment_steps"] < response_lookahead_max_steps
            ):
                selected_deltas = [response_delta[pos] for pos in selected if pos in response_delta]
                selected_risk = float(np.mean(selected_deltas)) if selected_deltas else 0.0
                if selected_risk >= response_lookahead_threshold:
                    probes: list[tuple[float, int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
                    for candidate in selected[:2]:
                        probe_x = x.clone()
                        probe_x[0, candidate] = x0[0, candidate]
                        probe_logits = _model_forward(model, probe_x, attention_mask, cfg_scale, prompt_index)
                        nfe += 1
                        probe_x0 = torch.argmax(add_gumbel_noise(probe_logits, temperature=temperature), dim=-1)
                        local_positions = [
                            pos for pos in masked_positions
                            if pos != candidate and abs(pos - candidate) <= response_local_window
                            and pos in prev_topk_map and pos in current_topk
                        ]
                        if not local_positions:
                            local_positions = [
                                pos for pos in masked_positions
                                if pos != candidate and pos in prev_topk_map and pos in current_topk
                            ]
                        probe_topk = _topk_states_for_positions(probe_logits[0], local_positions, k=top_k_track)
                        weighted_score = 0.0
                        weight_sum = 0.0
                        for pos in local_positions:
                            efficiency, movement = _topk_path_efficiency(
                                prev_topk_map[pos], current_topk[pos], probe_topk[pos]
                            )
                            weighted_score += efficiency * movement
                            weight_sum += movement
                        score = weighted_score / max(weight_sum, 1e-8)
                        probes.append((score, candidate, probe_x, probe_logits, probe_x0))
                    probes.sort(key=lambda item: item[0], reverse=True)
                    winner, anchor, x, logits, x0 = probes[0]
                    runner_up = probes[1][0]
                    mask_index = x == mask_id
                    refreshed_positions = [int(p) for p in torch.nonzero(mask_index[0], as_tuple=False).flatten().tolist()]
                    refreshed_selectable = [
                        pos for pos in refreshed_positions if block_start <= pos < block_end
                    ]
                    refreshed_scores = {
                        pos: confidence(logits[0, pos], int(x0[0, pos].item()))
                        for pos in refreshed_positions
                    }
                    selected = [anchor] + _select_transfer_indices(
                        refreshed_selectable, len(selected) - 1, refreshed_scores,
                        traj=traj, lateral=False, coupling_threshold=coupling_threshold, local_spacing=0,
                    )
                    current_topk = _topk_states_for_positions(logits[0], refreshed_positions, k=top_k_track)
                    diagnostics["response_alignment_steps"] += 1
                    diagnostics["response_alignment_margin_sum"] += winner - runner_up
                    diagnostics["response_alignment_winner_sum"] += winner
                    diagnostics["response_alignment_runner_up_sum"] += runner_up
            if sampler == "lcr_response_refresh" and len(selected) > 1:
                selected_deltas = [response_delta[p] for p in selected if p in response_delta]
                selected_risk = float(np.mean(selected_deltas)) if selected_deltas else 0.0
                if selected_risk >= response_refresh_threshold:
                    # Commit one anchor, then recompute the remainder under its visible context.
                    anchor = selected[0]
                    x0 = torch.where(mask_index, x0, x)
                    x[0, anchor] = x0[0, anchor]
                    refreshed_logits = _model_forward(model, x, attention_mask, cfg_scale, prompt_index)
                    nfe += 1
                    refreshed_noise = add_gumbel_noise(refreshed_logits, temperature=temperature)
                    refreshed_x0 = torch.argmax(refreshed_noise, dim=-1)
                    refreshed_mask = x == mask_id
                    refreshed_positions = [int(p) for p in torch.nonzero(refreshed_mask[0], as_tuple=False).flatten().tolist()]
                    refreshed_selectable = [
                        pos for pos in refreshed_positions if block_start <= pos < block_end
                    ]
                    refreshed_scores = {
                        pos: confidence(refreshed_logits[0, pos], int(refreshed_x0[0, pos].item()))
                        for pos in refreshed_positions
                    }
                    selected = [anchor] + _select_transfer_indices(
                        refreshed_selectable, len(selected) - 1, refreshed_scores, traj=traj,
                        lateral=False, coupling_threshold=coupling_threshold, local_spacing=0,
                    )
                    logits = refreshed_logits
                    x0 = refreshed_x0
                    mask_index = refreshed_mask
                    current_topk = _topk_states_for_positions(logits[0], refreshed_positions, k=top_k_track)
                    diagnostics["response_refresh_steps"] += 1
                    diagnostics["response_refresh_tokens"] += len(selected) - 1
            if (
                sampler == "lcr_response_lookahead"
                and len(selected) > 1
                and diagnostics["response_lookahead_steps"] < response_lookahead_max_steps
            ):
                selected_deltas = [response_delta[p] for p in selected if p in response_delta]
                selected_risk = float(np.mean(selected_deltas)) if selected_deltas else 0.0
                if selected_risk >= response_lookahead_threshold:
                    candidates = selected[:2]
                    probes: list[tuple[float, int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
                    for candidate in candidates:
                        probe_x = x.clone()
                        probe_x[0, candidate] = x0[0, candidate]
                        probe_logits = _model_forward(model, probe_x, attention_mask, cfg_scale, prompt_index)
                        nfe += 1
                        probe_x0 = torch.argmax(add_gumbel_noise(probe_logits, temperature=temperature), dim=-1)
                        local_positions = [
                            pos for pos in masked_positions
                            if pos != candidate and abs(pos - candidate) <= response_local_window
                        ]
                        if not local_positions:
                            local_positions = [pos for pos in masked_positions if pos != candidate]
                        local_index = torch.tensor(local_positions, device=probe_logits.device, dtype=torch.long)
                        local_conf = torch.softmax(probe_logits[0].index_select(0, local_index).float(), dim=-1).max(dim=-1).values.mean()
                        probes.append((float(local_conf.item()), candidate, probe_x, probe_logits, probe_x0))
                    probes.sort(key=lambda item: item[0], reverse=True)
                    winner, anchor, x, logits, x0 = probes[0]
                    runner_up = probes[1][0]
                    mask_index = x == mask_id
                    refreshed_positions = [int(p) for p in torch.nonzero(mask_index[0], as_tuple=False).flatten().tolist()]
                    refreshed_selectable = [
                        pos for pos in refreshed_positions if block_start <= pos < block_end
                    ]
                    refreshed_scores = {
                        pos: confidence(logits[0, pos], int(x0[0, pos].item()))
                        for pos in refreshed_positions
                    }
                    selected = [anchor] + _select_transfer_indices(
                        refreshed_selectable, len(selected) - 1, refreshed_scores, traj=traj,
                        lateral=False, coupling_threshold=coupling_threshold, local_spacing=0,
                    )
                    current_topk = _topk_states_for_positions(logits[0], refreshed_positions, k=top_k_track)
                    diagnostics["response_lookahead_steps"] += 1
                    diagnostics["response_lookahead_margin_sum"] += winner - runner_up
            if sampler in (
                "lcr_response_delay",
                "lcr_response_flip_delay",
                "lcr_response_stability_delay",
                "lcr_response_refresh",
                "lcr_response_lookahead",
                "lcr_response_budget",
                "lcr_response_persistence",
                "lcr_response_alignment",
                "lcr_response_wavefront",
                "lcr_terminal_refine",
                "lcr_response_confirm",
            ):
                selected_delay_candidates = delay_candidates.intersection(selected)
                diagnostics["response_delay_candidate_count"] += len(delay_candidates)
                diagnostics["response_delay_selected_count"] += len(selected_delay_candidates)
                diagnostics["response_delayed_count"] += len(delay_candidates) - len(selected_delay_candidates)
                if sampler == "lcr_response_confirm":
                    pending_rewrites = {
                        pos: token for pos, token in new_rewrites.items() if pos not in selected
                    }

            if sampler == "lcr_response_persistence":
                next_pending: dict[int, tuple[tuple[torch.Tensor, torch.Tensor], float]] = {}
                for pos, delta in response_delta.items():
                    if pos in selected or delta < response_min_delta:
                        continue
                    is_near = response_local_window <= 0 or any(
                        abs(pos - cpos) <= response_local_window for cpos in prev_committed_for_response
                    )
                    if is_near and pos in current_topk:
                        next_pending[pos] = (current_topk[pos], delta)
                pending_response_states = next_pending
                diagnostics["response_persistence_candidates"] += len(next_pending)

            assert all(block_start <= pos < block_end for pos in selected)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for pos in selected:
                transfer_index[0, pos] = True

            x0 = torch.where(mask_index, x0, x)
            x[transfer_index] = x0[transfer_index]

            if sampler in ("lcr_wavefront", "lcr_response_wavefront"):
                current_near_response = (
                    float(np.mean(step_near_responses)) if step_near_responses else 0.0
                )
                radius = wavefront_radius
                if sampler == "lcr_response_wavefront" and previous_near_response is not None:
                    if current_near_response > previous_near_response:
                        radius = 1
                        diagnostics["wavefront_narrow_steps"] += 1
                    else:
                        diagnostics["wavefront_expansion_steps"] += 1
                previous_near_response = current_near_response
                for pos in selected:
                    for neighbor in range(max(gen_start, pos - radius), min(gen_end, pos + radius + 1)):
                        if bool(x[0, neighbor] == mask_id):
                            wavefront.add(neighbor)
                wavefront = {pos for pos in wavefront if bool(x[0, pos] == mask_id)}
                if len(wavefront) > wavefront_size:
                    wavefront = set(sorted(
                        wavefront, key=lambda pos: scores.get(pos, -1e9), reverse=True
                    )[:wavefront_size])

            if traj is not None:
                traj.pending_commits = selected.copy()
                traj.step_records.append({
                    "step": i + num_block * steps_per_block,
                    "committed": selected,
                    "num_masked": len(masked_positions),
                })
            if sampler in (
                "lcr_probe",
                "lcr_response",
                "lcr_response_gated",
                "lcr_response_strong",
                "lcr_response_delay",
                "lcr_response_flip_delay",
                "lcr_response_stability_delay",
                "lcr_response_refresh",
                "lcr_response_lookahead",
                "lcr_response_budget",
                "lcr_response_persistence",
                "lcr_response_alignment",
                "lcr_response_wavefront",
                "lcr_terminal_refine",
                "lcr_response_confirm",
                "lcr_rewrite_refine",
                "lcr_rewrite_branch",
            ):
                prev_topk_map = current_topk
                prev_committed_for_response = selected.copy()
                selected_deltas = [response_delta[p] for p in selected if p in response_delta]
                diagnostics["response_selected_delta_sum"] += float(sum(selected_deltas))
                diagnostics["response_selected_count"] += len(selected_deltas)

        block_selected_count = diagnostics["response_selected_count"] - block_counter_start["response_selected_count"]
        block_selected_sum = diagnostics["response_selected_delta_sum"] - block_counter_start["response_selected_delta_sum"]
        block_near_count = diagnostics["response_near_delta_count"] - block_counter_start["response_near_delta_count"]
        block_near_sum = diagnostics["response_near_delta_sum"] - block_counter_start["response_near_delta_sum"]
        block_flips = [
            diag_flip_count.get(pos, 0) - block_flip_start.get(pos, 0)
            for pos in range(block_start, block_end)
        ]
        midpoint = max(1, len(block_near_response_trace) // 2)
        early_response = float(np.mean(block_near_response_trace[:midpoint]))
        late_response = float(np.mean(block_near_response_trace[midpoint:])) if midpoint < len(block_near_response_trace) else 0.0
        diagnostics["block_summaries"].append({
            "block_index": num_block,
            "block_length": block_length,
            "response_steps": diagnostics["response_steps"] - block_counter_start["response_steps"],
            "selected_delta_mean": block_selected_sum / block_selected_count if block_selected_count else 0.0,
            "near_delta_mean": block_near_sum / block_near_count if block_near_count else 0.0,
            "early_near_response": early_response,
            "late_near_response": late_response,
            "response_retention": late_response / (early_response + 1e-8),
            "flip_mean": float(np.mean(block_flips)) if block_flips else 0.0,
        })

    if sampler == "lcr_rewrite_refine" and rewrite_count and refine_max_tokens > 0:
        ranked_seeds = sorted(
            rewrite_count,
            key=lambda pos: (rewrite_count[pos], rewrite_max_delta.get(pos, 0.0)),
            reverse=True,
        )
        gen_start = prompt.shape[1]
        gen_end = gen_start + gen_length
        refine_positions: list[int] = []
        for seed in ranked_seeds:
            for pos in range(seed - refine_radius, seed + refine_radius + 1):
                if gen_start <= pos < gen_end and pos not in refine_positions:
                    refine_positions.append(pos)
                    if len(refine_positions) >= refine_max_tokens:
                        break
            if len(refine_positions) >= refine_max_tokens:
                break

        original_tokens = x[0, refine_positions].clone()
        x[0, refine_positions] = mask_id
        diagnostics["refine_masked_count"] = len(refine_positions)
        for _ in range(len(refine_positions)):
            mask_index = x == mask_id
            logits = _model_forward(model, x, attention_mask, cfg_scale, prompt_index)
            nfe += 1
            x0 = torch.argmax(logits, dim=-1)
            remaining = [pos for pos in refine_positions if bool(mask_index[0, pos])]
            scores = {
                pos: confidence(logits[0, pos], int(x0[0, pos].item())) for pos in remaining
            }
            selected_pos = max(remaining, key=scores.get)
            x[0, selected_pos] = x0[0, selected_pos]
        diagnostics["refine_changed_count"] = int(
            (x[0, refine_positions] != original_tokens).sum().item()
        )

    if sampler == "lcr_rewrite_branch" and strongest_rewrite is not None:
        gen_start = prompt.shape[1]
        gen_end = gen_start + gen_length
        effective_end = gen_end
        if eos_token_id is not None:
            eos_offsets = torch.nonzero(x[0, gen_start:gen_end] == eos_token_id, as_tuple=False).flatten()
            if eos_offsets.numel():
                effective_end = gen_start + int(eos_offsets[0].item())
        diagnostics["effective_gen_length"] = effective_end - gen_start
        tail_start = max(gen_start, effective_end - branch_tail_window)
        eligible_events = [event for event in rewrite_events if tail_start <= event[1] < effective_end]
        if not eligible_events:
            eligible_events = [event for event in rewrite_events if gen_start <= event[1] < effective_end]
        if eligible_events:
            event_delta, seed, old_top1 = max(eligible_events, key=lambda event: event[0])
        else:
            event_delta, seed, old_top1 = strongest_rewrite
        branch_positions = list(range(seed, effective_end))
        original_tokens = x[0, branch_positions].clone()
        x[0, branch_positions] = mask_id
        x[0, seed] = old_top1
        diagnostics["branch_event_delta"] = event_delta
        diagnostics["branch_seed_relative"] = seed - gen_start
        for _ in range(len(branch_positions) - 1):
            mask_index = x == mask_id
            logits = _model_forward(model, x, attention_mask, cfg_scale, prompt_index)
            nfe += 1
            x0 = torch.argmax(logits, dim=-1)
            remaining = [pos for pos in branch_positions if bool(mask_index[0, pos])]
            scores = {
                pos: confidence(logits[0, pos], int(x0[0, pos].item())) for pos in remaining
            }
            selected_pos = max(remaining, key=scores.get)
            x[0, selected_pos] = x0[0, selected_pos]
        diagnostics["branch_changed_count"] = int(
            (x[0, branch_positions] != original_tokens).sum().item()
        )

    terminal_risk = (
        diagnostics["response_selected_delta_sum"] / diagnostics["response_selected_count"]
        if diagnostics["response_selected_count"] else 0.0
    )
    if sampler == "lcr_terminal_refine" and terminal_risk >= terminal_refine_threshold:
        gen_start = prompt.shape[1]
        gen_end = gen_start + gen_length
        effective_end = gen_end
        if eos_token_id is not None:
            eos_offsets = torch.nonzero(x[0, gen_start:gen_end] == eos_token_id, as_tuple=False).flatten()
            if eos_offsets.numel():
                effective_end = gen_start + int(eos_offsets[0].item())
        refine_start = max(gen_start, effective_end - terminal_refine_tokens)
        terminal_positions = list(range(refine_start, effective_end))
        if terminal_positions:
            original_tokens = x[0, terminal_positions].clone()
            x[0, terminal_positions] = mask_id
            diagnostics["terminal_refine_triggered"] = 1
            for _ in range(len(terminal_positions)):
                mask_index = x == mask_id
                logits = _model_forward(model, x, attention_mask, cfg_scale, prompt_index)
                nfe += 1
                x0 = torch.argmax(logits, dim=-1)
                remaining = [pos for pos in terminal_positions if bool(mask_index[0, pos])]
                scores = {pos: confidence(logits[0, pos], int(x0[0, pos].item())) for pos in remaining}
                chosen = max(remaining, key=scores.get)
                x[0, chosen] = x0[0, chosen]
            diagnostics["terminal_refine_changed_count"] = int((x[0, terminal_positions] != original_tokens).sum().item())

    if diagnostics["response_steps"]:
        diagnostics["response_mean_delta"] = diagnostics["response_mean_delta_sum"] / diagnostics["response_steps"]
    else:
        diagnostics["response_mean_delta"] = 0.0
    if diagnostics["response_selected_count"]:
        diagnostics["response_selected_delta_mean"] = diagnostics["response_selected_delta_sum"] / diagnostics["response_selected_count"]
    else:
        diagnostics["response_selected_delta_mean"] = 0.0
    for locality in ("near", "far"):
        count = diagnostics[f"response_{locality}_delta_count"]
        total = diagnostics[f"response_{locality}_delta_sum"]
        diagnostics[f"response_{locality}_delta_mean"] = total / count if count else 0.0
    gen_start = prompt.shape[1]
    answer_positions = range(gen_start + max(gen_length - 15, 0), gen_start + gen_length)
    answer_flips = [diag_flip_count.get(pos, 0) for pos in answer_positions]
    all_flips = list(diag_flip_count.values())
    diagnostics["answer_flip_mean"] = float(np.mean(answer_flips)) if answer_flips else 0.0
    diagnostics["all_flip_mean"] = float(np.mean(all_flips)) if all_flips else 0.0
    return {"tokens": x, "nfe": nfe, "trajectory": traj, "diagnostics": diagnostics}
