"""Node 6.1: Flip-gated Best-of-2 sampler for GSM8K.

Algorithm:
1. Run LCR generation, tracking flip counts at answer positions
2. If mean answer-region flip count > threshold T, run a second generation
3. Compare the two predictions:
   - If they agree → output that answer
   - If they disagree → output the one with lower flip count (more confident)
4. If flip count <= T → output first generation directly (no extra NFE)
"""
from __future__ import annotations

import argparse
import json
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


@torch.no_grad()
def generate_lcr_tracked(model, tokenizer, question: str,
                          steps: int = 64, gen_length: int = 128,
                          temperature: float = 0.0):
    """Run LCR and return (gen_text, answer_flip_mean, pred)."""
    device = next(model.parameters()).device
    input_ids, _ = encode_prompt(tokenizer, question, device=device)
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

    # Answer region = last 15 positions
    answer_positions = range(prompt_len + gen_length - 15, prompt_len + gen_length)
    flip_vals = []
    for pos in answer_positions:
        hist = argmax_history.get(pos, [])
        if len(hist) > 1:
            flips = sum(1 for a, b in zip(hist[:-1], hist[1:]) if a != b)
            flip_vals.append(flips)
    answer_flip_mean = float(np.mean(flip_vals)) if flip_vals else 0.0

    return gen_text, answer_flip_mean, pred


def eval_gsm8k_bo2(model, tokenizer, limit: int, steps: int, gen_length: int,
                   threshold: float) -> dict:
    items = load_gsm8k("/root/autodl-tmp/dataset", limit=limit)
    correct_lcr = 0
    correct_bo2 = 0
    records = []
    extra_nfe = 0

    for idx, item in enumerate(items):
        print(f"[{idx+1}/{len(items)}] threshold={threshold:.1f}", end="\r", flush=True)
        q = item["question"]
        gold = item["answer"]

        # First generation
        gen1, flip1, pred1 = generate_lcr_tracked(model, tokenizer, q, steps, gen_length)
        is_correct_lcr = pred1 is not None and pred1.strip() == gold.strip()
        correct_lcr += int(is_correct_lcr)

        # Flip-gated second generation
        if flip1 > threshold:
            gen2, flip2, pred2 = generate_lcr_tracked(model, tokenizer, q, steps, gen_length)
            extra_nfe += 1
            # Selection: agree → output; disagree → lower flip count wins
            if pred1 == pred2:
                final_pred = pred1
            else:
                final_pred = pred1 if flip1 <= flip2 else pred2
        else:
            final_pred = pred1
            flip2 = None
            pred2 = None

        is_correct_bo2 = final_pred is not None and final_pred.strip() == gold.strip()
        correct_bo2 += int(is_correct_bo2)

        records.append({
            "idx": idx, "gold": gold,
            "pred1": pred1, "flip1": flip1,
            "pred2": pred2, "flip2": flip2,
            "final_pred": final_pred,
            "correct_lcr": is_correct_lcr,
            "correct_bo2": is_correct_bo2,
            "triggered": flip1 > threshold,
        })

    n = len(items)
    acc_lcr = correct_lcr / n
    acc_bo2 = correct_bo2 / n
    n_triggered = sum(r["triggered"] for r in records)

    print(f"\n=== Flip-gated Best-of-2 (threshold={threshold}) ===")
    print(f"LCR:         {acc_lcr:.1%} ({correct_lcr}/{n})")
    print(f"BO2 (flip-gated): {acc_bo2:.1%} ({correct_bo2}/{n})")
    print(f"Triggered:   {n_triggered}/{n} samples ({n_triggered/n:.0%})")
    print(f"Extra NFE:   {extra_nfe} extra generations")
    print(f"Gain:        {(acc_bo2 - acc_lcr)*100:+.1f}%")
    print(f"SCORE: {acc_bo2 * 100:.1f}")

    return {
        "threshold": threshold,
        "accuracy_lcr": acc_lcr,
        "accuracy_bo2": acc_bo2,
        "correct_lcr": correct_lcr,
        "correct_bo2": correct_bo2,
        "total": n,
        "n_triggered": n_triggered,
        "extra_nfe": extra_nfe,
        "gain": acc_bo2 - acc_lcr,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=3.5,
                        help="Answer flip mean threshold for triggering 2nd generation. "
                             "Wrong samples have mean=4.27, correct=2.51, so 3.5 is midpoint.")
    parser.add_argument("--model_path", default="/root/autodl-tmp/model/LLaDA/instruct")
    parser.add_argument("--out_dir", default="results/round4_bo2")
    args = parser.parse_args()

    print("Loading model ...")
    model, tokenizer = load_llada(args.model_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = eval_gsm8k_bo2(
        model, tokenizer,
        limit=args.limit, steps=args.steps,
        gen_length=args.gen_length, threshold=args.threshold,
    )

    out = out_dir / f"gsm8k_bo2_t{args.threshold:.1f}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
