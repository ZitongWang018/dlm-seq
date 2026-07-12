#!/usr/bin/env python3
"""Reframe observation: longitudinal and lateral DLM trajectory evidence.

This script is intentionally observation-only. It runs standard LCR decoding
and records unused distributions so we can inspect what they mean before
designing another sampler.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import extract_number, load_gsm8k
from src.distribution import _topk_l1, confidence
from src.model_loader import encode_prompt, load_llada
from src.samplers import MASK_ID, get_num_transfer_tokens


def _decode_token(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=True)
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text if text.strip() else f"<{token_id}>"


def _topk_snapshot(logits: torch.Tensor, k: int) -> list[dict]:
    probs = torch.softmax(logits.float(), dim=-1)
    vals, ids = torch.topk(probs, k=min(k, probs.shape[-1]))
    return [
        {"id": int(tok), "prob": float(prob)}
        for tok, prob in zip(ids.detach().cpu().tolist(), vals.detach().cpu().tolist())
    ]


def _find_number_positions(token_ids: list[int], tokenizer, start_offset: int) -> list[int]:
    positions = []
    for i, tok in enumerate(token_ids):
        text = tokenizer.decode([tok], skip_special_tokens=True).strip()
        if re.fullmatch(r"-?\d+(\.\d+)?", text):
            positions.append(start_offset + i)
    return positions


def _plot_longitudinal(sample_dir: Path, tokenizer, histories: dict[int, list[dict]], focus_positions: list[int]):
    for pos in focus_positions:
        hist = histories.get(pos, [])
        if not hist:
            continue

        token_scores: defaultdict[int, float] = defaultdict(float)
        token_counts: Counter[int] = Counter()
        for step in hist:
            for cand in step["topk"]:
                token_scores[cand["id"]] += cand["prob"]
                token_counts[cand["id"]] += 1
        keep = [tok for tok, _ in sorted(token_scores.items(), key=lambda kv: kv[1], reverse=True)[:6]]

        xs = [step["step"] for step in hist]
        plt.figure(figsize=(8, 4.5))
        for tok in keep:
            ys = []
            for step in hist:
                prob = 0.0
                for cand in step["topk"]:
                    if cand["id"] == tok:
                        prob = cand["prob"]
                        break
                ys.append(prob)
            label = _decode_token(tokenizer, tok)
            plt.plot(xs, ys, marker="o", markersize=2, linewidth=1.4, label=label[:18])

        commit_step = hist[-1].get("commit_step")
        if commit_step is not None:
            plt.axvline(commit_step, color="#222222", linestyle="--", linewidth=1.0, label="Commit step")
        plt.xlabel("Denoising step")
        plt.ylabel("Candidate probability")
        plt.title(f"Longitudinal token trajectories at position {pos}")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(sample_dir / f"longitudinal_pos_{pos}.png", dpi=160)
        plt.close()


def _plot_lateral(sample_dir: Path, tokenizer, response_events: list[dict]):
    if not response_events:
        return
    best = max(response_events, key=lambda e: e["max_delta"])
    responses = best["responses"][:12]
    if not responses:
        return

    labels = [str(r["pos"]) for r in responses]
    vals = [r["delta"] for r in responses]
    commit_text = ", ".join(_decode_token(tokenizer, t) for t in best.get("commit_token_ids", [])[:4])

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, vals, color="#4C78A8")
    plt.xlabel("Response position")
    plt.ylabel("Distribution change")
    plt.title(f"Response after token commitment at step {best['step']} ({commit_text})")
    plt.tight_layout()
    plt.savefig(sample_dir / "lateral_response_top_event.png", dpi=160)
    plt.close()


def _response_distance(event: dict) -> float | None:
    commits = event.get("committed_positions", [])
    responses = event.get("responses", [])
    if not commits or not responses:
        return None
    top_pos = int(responses[0]["pos"])
    return float(min(abs(top_pos - int(cpos)) for cpos in commits))


@torch.no_grad()
def observe_item(model, tokenizer, item: dict, steps: int, gen_length: int, top_k: int):
    device = next(model.parameters()).device
    input_ids, _ = encode_prompt(tokenizer, item["question"], device=device)
    prompt_len = input_ids.shape[1]

    x = torch.cat(
        [
            input_ids,
            torch.full((1, gen_length), MASK_ID, dtype=torch.long, device=device),
        ],
        dim=1,
    )

    block_mask = x[:, prompt_len : prompt_len + gen_length] == MASK_ID
    num_transfer = get_num_transfer_tokens(block_mask, steps)

    histories: dict[int, list[dict]] = defaultdict(list)
    commit_step_by_pos: dict[int, int] = {}
    prev_logits: dict[int, torch.Tensor] = {}
    prev_committed: list[int] = []
    response_events: list[dict] = []

    for step in range(steps):
        mask_index = x == MASK_ID
        logits = model(x).logits
        x0 = torch.argmax(logits[0], dim=-1)

        masked_pos = torch.nonzero(mask_index[0], as_tuple=False).flatten().tolist()

        if prev_committed and prev_logits:
            responses = []
            for pos in masked_pos:
                if pos in prev_committed or pos not in prev_logits:
                    continue
                delta = _topk_l1(prev_logits[pos], logits[0, pos].detach().cpu(), k=top_k)
                if delta > 1e-5:
                    responses.append({"pos": int(pos), "delta": float(delta)})
            responses.sort(key=lambda r: r["delta"], reverse=True)
            if responses:
                response_events.append(
                    {
                        "step": step,
                        "committed_positions": [int(p) for p in prev_committed],
                        "commit_token_ids": [int(x[0, p].item()) for p in prev_committed],
                        "max_delta": float(responses[0]["delta"]),
                        "responses": responses[:20],
                    }
                )

        current_logits = {}
        scores = {}
        for pos in masked_pos:
            pos_logits = logits[0, pos]
            pred = int(x0[pos].item())
            scores[int(pos)] = confidence(pos_logits, pred)
            current_logits[int(pos)] = pos_logits.detach().cpu()
            histories[int(pos)].append(
                {
                    "step": step,
                    "top1": pred,
                    "top1_prob": scores[int(pos)],
                    "topk": _topk_snapshot(pos_logits, top_k),
                }
            )

        k = int(num_transfer[0, step].item())
        selected = sorted([int(p) for p in masked_pos], key=lambda p: scores.get(p, -1e9), reverse=True)[:k]

        transfer = torch.zeros_like(x0, dtype=torch.bool)
        for pos in selected:
            transfer[pos] = True
            commit_step_by_pos[pos] = step
        x0_full = torch.where(mask_index[0], x0, x[0])
        x[0][transfer] = x0_full[transfer]

        prev_logits = current_logits
        prev_committed = selected

    for pos, step in commit_step_by_pos.items():
        if pos in histories and histories[pos]:
            histories[pos][-1]["commit_step"] = step

    gen_tokens = x[0, prompt_len:].tolist()
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    pred = extract_number(gen_text)
    correct = pred is not None and pred.strip() == item["answer"].strip()
    digit_positions = _find_number_positions(gen_tokens, tokenizer, prompt_len)
    answer_positions = list(range(prompt_len + gen_length - 15, prompt_len + gen_length))
    focus_positions = []
    for pos in digit_positions[-6:] + answer_positions[-4:]:
        if pos in histories and pos not in focus_positions:
            focus_positions.append(pos)
    focus_positions = focus_positions[:8]

    return {
        "correct": bool(correct),
        "pred": pred,
        "gold": item["answer"],
        "gen_text": gen_text,
        "prompt_len": prompt_len,
        "focus_positions": focus_positions,
        "histories": histories,
        "response_events": response_events,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--model_path", default="/root/autodl-tmp/model/LLaDA/instruct")
    parser.add_argument("--dataset_root", default="/root/autodl-tmp/dataset")
    parser.add_argument("--out_dir", default="results/reframe_observation")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_llada(args.model_path)
    items = load_gsm8k(args.dataset_root, limit=args.limit)

    records = []
    for idx, item in enumerate(items):
        print(f"[{idx + 1}/{len(items)}] observing", flush=True)
        sample_dir = out_dir / f"sample_{idx:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        obs = observe_item(model, tokenizer, item, args.steps, args.gen_length, args.top_k)
        _plot_longitudinal(sample_dir, tokenizer, obs["histories"], obs["focus_positions"])
        _plot_lateral(sample_dir, tokenizer, obs["response_events"])

        compact = {
            "idx": idx,
            "correct": obs["correct"],
            "pred": obs["pred"],
            "gold": obs["gold"],
            "focus_positions": obs["focus_positions"],
            "num_response_events": len(obs["response_events"]),
            "top_response_event": max(obs["response_events"], key=lambda e: e["max_delta"]) if obs["response_events"] else None,
            "gen_text_snippet": obs["gen_text"][:300],
        }
        if compact["top_response_event"] is not None:
            compact["top_response_distance"] = _response_distance(compact["top_response_event"])
        records.append(compact)
        with open(sample_dir / "record.json", "w") as f:
            json.dump(compact, f, indent=2)

    max_deltas = [
        r["top_response_event"]["max_delta"]
        for r in records
        if r.get("top_response_event") is not None
    ]
    top_distances = [
        r["top_response_distance"]
        for r in records
        if r.get("top_response_distance") is not None
    ]
    summary = {
        "num_samples": len(records),
        "num_correct": sum(1 for r in records if r["correct"]),
        "lateral_response": {
            "mean_top_delta": float(np.mean(max_deltas)) if max_deltas else 0.0,
            "median_top_delta": float(np.median(max_deltas)) if max_deltas else 0.0,
            "mean_top_distance": float(np.mean(top_distances)) if top_distances else 0.0,
            "median_top_distance": float(np.median(top_distances)) if top_distances else 0.0,
        },
        "records": records,
        "figure_captions": {
            "longitudinal": "Longitudinal token trajectories. Top candidates for a fixed position across denoising steps.",
            "lateral": "Response after token commitment. Distribution changes at still-masked positions before and after a selected token is committed.",
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved observation artifacts to {out_dir}")


if __name__ == "__main__":
    main()
