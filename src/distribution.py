"""Distribution utilities for trajectory analysis."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def softmax_dist(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits.float(), dim=-1)


def topk_dist(logits: torch.Tensor, k: int = 32):
    """Return top-k token ids and normalized probabilities for a 1D logit vector."""
    probs = softmax_dist(logits)
    vals, idx = torch.topk(probs, k=min(k, probs.shape[-1]))
    vals = vals / vals.sum().clamp_min(1e-12)
    return idx.cpu(), vals.cpu()


def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor, top_k: int = 32) -> float:
    """KL(p||q) on union support of top-k from both distributions."""
    p = softmax_dist(p_logits)
    q = softmax_dist(q_logits)
    _, pi = torch.topk(p, k=min(top_k, p.shape[-1]))
    _, qi = torch.topk(q, k=min(top_k, q.shape[-1]))
    support = torch.unique(torch.cat([pi, qi])).long()
    p_s = p[support].clamp_min(1e-12)
    q_s = q[support].clamp_min(1e-12)
    p_s = p_s / p_s.sum()
    q_s = q_s / q_s.sum()
    return float((p_s * (p_s.log() - q_s.log())).sum().item())


def _topk_l1(p_logits: torch.Tensor, q_logits: torch.Tensor, k: int = 8) -> float:
    p = softmax_dist(p_logits)
    q = softmax_dist(q_logits)
    _, pi = torch.topk(p, k=min(k, p.shape[-1]))
    _, qi = torch.topk(q, k=min(k, q.shape[-1]))
    support = torch.unique(torch.cat([pi, qi])).long()
    return float((p[support] - q[support]).abs().sum().item())


def confidence(logits: torch.Tensor, token_id: int | None = None) -> float:
    probs = softmax_dist(logits)
    if token_id is None:
        token_id = int(probs.argmax().item())
    return float(probs[token_id].item())


def margin(logits: torch.Tensor) -> float:
    probs = softmax_dist(logits)
    top2 = torch.topk(probs, k=2).values
    return float((top2[0] - top2[1]).item())
