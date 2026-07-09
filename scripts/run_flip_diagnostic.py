"""Node 6: trajectory flip-count diagnostic.

For GSM8K samples, track argmax flip counts across 64 steps.
Compare correct vs incorrect samples at intermediate vs final positions.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import load_gsm8k, extract_number
from src.model_loader import load_llada, encode_prompt
from src.samplers import MASK_ID, get_num_transfer_tokens


def _fwd(model, x):
    return model(x).logits


def find_digit_positions(token_ids: list[int], tokenizer, start_offset: int) -> list[int]:
    positions = []
    for i, tok in enumerate(token_ids):
        s = tokenizer.decode([tok], skip_special_tokens=True).strip()
        if re.fullmatch(r"-?\d+\.?\d*", s):
            positions.append(start_offset + i)
    return positions


@torch.no_grad()
def run_with_flip_tracking(model, tokenizer, item: dict,
                            steps: int = 64, gen_length: int = 128):
    device = next(model.parameters()).device
    input_ids, _ = encode_prompt(tokenizer, item["question"], device=device)
    prompt_len = input_ids.shape[1]

    x = torch.cat([
        input_ids,
        torch.full((1, gen_length), MASK_ID, dtype=torch.long, device=device),
    ], dim=1)

    bmi = x[:, prompt_len:prompt_len + gen_length] == MASK_ID
    num_transfer = get_num_transfer_tokens(bmi, steps)

    argmax_history: dict[int, list[int]] = {}

    for i in range(steps):
        mask_index = x == MASK_ID
        logits = _fwd(model, x)
        probs = torch.softmax(logits[0].float(), dim=-1)
        x0 = torch.argmax(logits[0], dim=-1)

        masked_pos = torch.nonzero(mask_index[0], as_tuple=False).flatten().tolist()
        for pos in masked_pos:
            if pos not in argmax_history:
                argmax_history[pos] = []
            argmax_history[pos].append(int(x0[pos].item()))

        scores = {pos: float(probs[pos, int(x0[pos].item())].item()) for pos in masked_pos}
        k = int(num_transfer[0, i].item())
        selected = sorted(masked_pos, key=lambda p: scores.get(p, -1e9), reverse=True)[:k]

        transfer = torch.zeros_like(x0, dtype=torch.bool)
        for pos in selected:
            transfer[pos] = True
        x0_full = torch.where(mask_index[0], x0, x[0])
        x[0][transfer] = x0_full[transfer]

    gen_tokens = x[0, prompt_len:].tolist()
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    pred = extract_number(gen_text)
    is_correct = pred is not None and pred.strip() == item["answer"].strip()

    flip_counts: dict[int, int] = {}
    for pos, hist in argmax_history.items():
        flips = sum(1 for a, b in zip(hist[:-1], hist[1:]) if a != b)
        flip_counts[pos] = flips

    digit_positions = find_digit_positions(gen_tokens, tokenizer, prompt_len)
    answer_positions = set(range(prompt_len + gen_length - 15, prompt_len + gen_length))
    reasoning_positions = set(digit_positions) - answer_positions

    def mean_flips(positions):
        vals = [flip_counts.get(p, 0) for p in positions if p in flip_counts]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "correct": is_correct,
        "pred": pred,
        "gold": item["answer"],
        "answer_flip_mean": mean_flips(answer_positions),
        "reasoning_flip_mean": mean_flips(reasoning_positions),
        "all_flip_mean": mean_flips(flip_counts.keys()),
        "n_digit_positions": len(digit_positions),
        "n_reasoning_digit_pos": len(reasoning_positions),
        "gen_text_snippet": gen_text[:200],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--model_path", default="/root/autodl-tmp/model/LLaDA/instruct")
    parser.add_argument("--dataset_root", default="/root/autodl-tmp/dataset")
    parser.add_argument("--out", default="results/flip_diagnostic/gsm8k_flip.json")
    args = parser.parse_args()

    print("Loading model ...")
    model, tokenizer = load_llada(args.model_path)

    print(f"Loading GSM8K (limit={args.limit}) ...")
    items = load_gsm8k(args.dataset_root, limit=args.limit)

    records = []
    correct_flips: dict[str, list[float]] = {"answer": [], "reasoning": [], "all": []}
    incorrect_flips: dict[str, list[float]] = {"answer": [], "reasoning": [], "all": []}

    for idx, item in enumerate(items):
        print(f"[{idx+1}/{len(items)}]", end="\r", flush=True)
        try:
            res = run_with_flip_tracking(model, tokenizer, item)
            res["idx"] = idx
            records.append(res)
            bucket = correct_flips if res["correct"] else incorrect_flips
            bucket["answer"].append(res["answer_flip_mean"])
            bucket["reasoning"].append(res["reasoning_flip_mean"])
            bucket["all"].append(res["all_flip_mean"])
        except Exception as e:
            print(f"\n[ERROR] idx={idx}: {e}")

    def summarize(bucket: dict) -> dict:
        return {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
            for k, v in bucket.items() if v
        }

    c_stats = summarize(correct_flips)
    i_stats = summarize(incorrect_flips)

    print(f"\n=== Flip Diagnostic ===")
    print(f"Correct ({len(correct_flips['all'])} samples):")
    for k, s in c_stats.items():
        print(f"  {k:20s}: {s['mean']:.3f} ± {s['std']:.3f}")
    print(f"Incorrect ({len(incorrect_flips['all'])} samples):")
    for k, s in i_stats.items():
        print(f"  {k:20s}: {s['mean']:.3f} ± {s['std']:.3f}")

    all_reasoning = [r["reasoning_flip_mean"] for r in records if "correct" in r]
    all_correct_flag = [int(r["correct"]) for r in records if "correct" in r]
    if len(all_reasoning) > 5:
        x_arr = np.array(all_reasoning)
        y_arr = np.array(all_correct_flag)
        if x_arr.std() > 1e-9:
            r_val = float(np.corrcoef(x_arr, y_arr)[0, 1])
        else:
            r_val = 0.0
        print(f"\nPearson r(reasoning_flips, correctness) = {r_val:.3f}")

    result = {
        "correct_stats": c_stats,
        "incorrect_stats": i_stats,
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved to {out}")
    acc = len(correct_flips["all"]) / max(len(records), 1)
    print(f"SCORE: {acc * 100:.1f}")


if __name__ == "__main__":
    main()
