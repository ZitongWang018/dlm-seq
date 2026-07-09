"""Samplers v2: argmax-identity stability gating and Jaccard top-K stability.

New sampler modes:
  - 'ids'  : Identity-Stable scoring — boost positions whose argmax has been
             consistent for >= k consecutive steps; penalise recent flippers.
  - 'jac'  : Jaccard top-K stability — score based on top-K set consistency.
  - 'ids+' : Identity-Stable with confidence blend (adaptive gate).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

from src.samplers import (
    MASK_ID,
    TrajectoryState,
    add_gumbel_noise,
    get_num_transfer_tokens,
    _model_forward,
    _select_transfer_indices,
)
from src.distribution import confidence


# ── Identity-Stability State ─────────────────────────────────────────────────

@dataclass
class IdentityState:
    """Track argmax token identity per position across steps."""
    # argmax_history[pos] = deque of last `window` argmax token ids
    argmax_history: dict[int, list[int]] = field(default_factory=dict)
    # topk_history[pos] = deque of last `window` frozensets of top-K ids
    topk_history: dict[int, list[frozenset]] = field(default_factory=dict)
    conf_history: dict[int, list[float]] = field(default_factory=dict)
    window: int = 4

    def update(self, pos: int, logits: torch.Tensor, top_k: int = 8):
        probs = torch.softmax(logits.float(), dim=-1)
        top1 = int(logits.argmax().item())
        conf = float(probs[top1].item())
        topk_ids = frozenset(torch.topk(probs, min(top_k, probs.shape[0])).indices.tolist())

        if pos not in self.argmax_history:
            self.argmax_history[pos] = []
            self.topk_history[pos] = []
            self.conf_history[pos] = []

        hist = self.argmax_history[pos]
        hist.append(top1)
        if len(hist) > self.window:
            hist.pop(0)

        tk = self.topk_history[pos]
        tk.append(topk_ids)
        if len(tk) > self.window:
            tk.pop(0)

        ch = self.conf_history[pos]
        ch.append(conf)
        if len(ch) > self.window:
            ch.pop(0)

    def identity_stability(self, pos: int) -> float:
        """Fraction of last-window steps where argmax == current argmax.
        Returns 1.0 if only 1 observation (no history yet).
        """
        hist = self.argmax_history.get(pos, [])
        if len(hist) < 2:
            return 0.5  # neutral when no history
        current = hist[-1]
        return sum(t == current for t in hist) / len(hist)

    def jaccard_stability(self, pos: int) -> float:
        """Mean pairwise Jaccard of top-K sets over last-window steps."""
        tk = self.topk_history.get(pos, [])
        if len(tk) < 2:
            return 0.5
        scores = []
        for i in range(len(tk) - 1):
            a, b = tk[i], tk[i + 1]
            inter = len(a & b)
            union = len(a | b)
            scores.append(inter / union if union > 0 else 1.0)
        return float(np.mean(scores))

    def current_conf(self, pos: int) -> float:
        ch = self.conf_history.get(pos, [0.0])
        return ch[-1] if ch else 0.0

    def conf_trend(self, pos: int) -> float:
        """Recent confidence trend: last - first in window. Positive = improving."""
        ch = self.conf_history.get(pos, [])
        if len(ch) < 2:
            return 0.0
        return ch[-1] - ch[0]


# ── Scoring functions ─────────────────────────────────────────────────────────

def score_ids(pos: int, state: IdentityState, alpha: float = 0.5, beta: float = 0.5) -> float:
    """Identity-Stable score: blend of current confidence and identity stability.

    Positions with high identity stability AND rising confidence score highest.
    Positions that recently flipped (low stability) are penalised regardless of
    their current confidence level.

    alpha: weight of current confidence
    beta:  weight of identity stability
    """
    conf = state.current_conf(pos)
    stab = state.identity_stability(pos)
    trend = max(state.conf_trend(pos), 0.0)  # only reward upward trend
    # penalise positions that are stable on LOW confidence (frozen-wrong)
    # by multiplying stability by confidence so high-conf-stable scores highest
    return alpha * conf + beta * (stab * conf) + 0.1 * trend


def score_jac(pos: int, state: IdentityState, alpha: float = 0.4, beta: float = 0.6) -> float:
    """Jaccard top-K stability score."""
    conf = state.current_conf(pos)
    jac = state.jaccard_stability(pos)
    trend = max(state.conf_trend(pos), 0.0)
    return alpha * conf + beta * (jac * conf) + 0.1 * trend


def score_ids_plus(pos: int, state: IdentityState) -> float:
    """Adaptive blend: trust stability more when confidence is moderate."""
    conf = state.current_conf(pos)
    stab = state.identity_stability(pos)
    jac = state.jaccard_stability(pos)
    trend = state.conf_trend(pos)

    # Gate: if conf is already very high (>0.9), just use conf (LCR behaviour)
    if conf > 0.90:
        return conf

    # If conf is moderate (0.4-0.9), use stability as tiebreaker
    stab_bonus = 0.3 * stab * jac  # both must be high to get bonus
    trend_bonus = 0.1 * max(trend, 0.0)
    return conf + stab_bonus + trend_bonus


# ── Main generator ─────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_v2(
    model,
    prompt,
    attention_mask=None,
    steps: int = 64,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    mask_id: int = MASK_ID,
    sampler: str = "ids",   # 'ids' | 'jac' | 'ids+' | 'lcr'
    top_k: int = 8,
    stability_window: int = 4,
):
    """Unified generation with v2 samplers."""
    device = prompt.device
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length), mask_id,
        dtype=torch.long, device=device
    )
    x[:, :prompt.shape[1]] = prompt.clone()
    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=device),
        ], dim=-1)
    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    state = IdentityState(window=stability_window)
    nfe = 0

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = x == mask_id
            logits = _model_forward(model, x, attention_mask, cfg_scale, prompt_index)
            nfe += 1

            logits_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_noise, dim=-1)

            masked_positions = torch.nonzero(mask_index[0], as_tuple=False).flatten().tolist()

            # Update state with current step's logits
            for pos in masked_positions:
                state.update(pos, logits[0, pos], top_k=top_k)

            # Compute scores
            scores: dict[int, float] = {}
            for pos in masked_positions:
                if sampler == "lcr":
                    pred = int(x0[0, pos].item())
                    scores[pos] = confidence(logits[0, pos], pred)
                elif sampler == "ids":
                    scores[pos] = score_ids(pos, state)
                elif sampler == "jac":
                    scores[pos] = score_jac(pos, state)
                elif sampler == "ids+":
                    scores[pos] = score_ids_plus(pos, state)
                else:
                    raise ValueError(f"Unknown sampler: {sampler}")

            k = int(num_transfer[0, i].item())
            selected = sorted(masked_positions, key=lambda p: scores.get(p, -1e9), reverse=True)[:k]

            transfer = torch.zeros_like(x0, dtype=torch.bool)
            for pos in selected:
                transfer[0, pos] = True
            x0_full = torch.where(mask_index, x0, x)
            x[transfer] = x0_full[transfer]

    return {"tokens": x, "nfe": nfe}
