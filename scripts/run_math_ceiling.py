"""MATH LCR baseline + H4 ceiling measurement in one pass.

Runs LCR on MATH level 1-3 (dev), measures:
1. pass@1 accuracy (boxed answer match)
2. H4 ceiling: for wrong samples, was gold answer token ever argmax?
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets_v2 import load_math, math_prompt, check_math_answer, extract_math_answer
from src.model_loader import load_llada, encode_prompt
from src.samplers import MASK_ID, get_num_transfer_tokens


def _fwd(model, x):
    return model(x).logits


@torch.no_grad()
def run_math_with_ceiling(
    model, tokenizer, item: dict,
    steps: int = 256, gen_length: int = 256, block_length: int = 256,
):
    device = next(model.parameters()).device
    prompt = math_prompt(item)
    input_ids, _ = encode_prompt(tokenizer, prompt, device=device)
    prompt_len = input_ids.shape[1]

    x = torch.cat([
        input_ids,
        torch.full((1, gen_length), MASK_ID, dtype=torch.long, device=device),
    ], dim=1)

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    # Track argmax per position per step
    argmax_history: dict[int, list[int]] = {}

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length
        bmi = x[:, block_start:block_end] == MASK_ID
        num_transfer = get_num_transfer_tokens(bmi, steps_per_block)

        for i in range(steps_per_block):
            mask_index = x == MASK_ID
            logits = _fwd(model, x)
            x0 = torch.argmax(logits[0], dim=-1)
            probs = torch.softmax(logits[0].float(), dim=-1)

            masked_pos = torch.nonzero(mask_index[0], as_tuple=False).flatten().tolist()
            for pos in masked_pos:
                top1 = int(x0[pos].item())
                if pos not in argmax_history:
                    argmax_history[pos] = []
                argmax_history[pos].append(top1)

            # LCR: commit by confidence
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

    gold_solution = item["solution"]
    is_correct = check_math_answer(gen_text, gold_solution)
    gold_answer = extract_math_answer(gold_solution)

    # Ceiling measurement: tokenize gold boxed answer, search in generated region
    ever_correct_ratio = 0.0
    if not is_correct and gold_answer is not None:
        gold_tokens = tokenizer.encode(gold_answer, add_special_tokens=False)
        # Search last 30 positions for answer
        answer_positions = list(range(prompt_len + gen_length - 30, prompt_len + gen_length))
        ever_correct_count = 0
        for pos in answer_positions:
            if pos not in argmax_history:
                continue
            hist = argmax_history[pos]
            rel = pos - (prompt_len + gen_length - len(gold_tokens))
            if 0 <= rel < len(gold_tokens):
                gold_tok = gold_tokens[rel]
                if gold_tok in hist:
                    ever_correct_count += 1
        ever_correct_ratio = ever_correct_count / max(len(answer_positions), 1)

    return {
        "correct": is_correct,
        "pred": gen_text[-100:],  # last 100 chars
        "gold": gold_answer,
        "level": item["level"],
        "type": item["type"],
        "ever_correct_ratio": ever_correct_ratio,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--model_path", default="/root/autodl-tmp/model/LLaDA/instruct")
    parser.add_argument("--dataset_root", default="/root/autodl-tmp/dataset")
    parser.add_argument("--out", default="results/math_baseline/math_ceiling.json")
    args = parser.parse_args()

    print("Loading model ...")
    model, tokenizer = load_llada(args.model_path)

    print(f"Loading MATH dev (limit={args.limit}) ...")
    items = load_math(args.dataset_root, split="dev", limit=args.limit)

    records = []
    n_correct, n_incorrect, n_ceiling = 0, 0, 0

    for idx, item in enumerate(items):
        print(f"[{idx+1}/{len(items)}] level={item['level']}", end="\r", flush=True)
        try:
            res = run_math_with_ceiling(
                model, tokenizer, item,
                steps=args.steps, gen_length=args.gen_length,
            )
            res["idx"] = idx
            records.append(res)
            if res["correct"]:
                n_correct += 1
            else:
                n_incorrect += 1
                if res["ever_correct_ratio"] > 0.3:
                    n_ceiling += 1
        except Exception as e:
            print(f"\n[ERROR] idx={idx}: {e}")
            records.append({"idx": idx, "error": str(e)})

    acc = n_correct / max(len(items), 1)
    ceiling = n_ceiling / max(n_incorrect, 1)

    summary = {
        "dataset": "MATH-dev-L1-3",
        "total": len(items),
        "correct": n_correct,
        "incorrect": n_incorrect,
        "accuracy": acc,
        "ceiling_ratio": ceiling,
        "ceiling_interpretation": "HIGH" if ceiling > 0.20 else "LOW",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2, default=str)

    print(f"\n=== MATH Baseline + Ceiling ===")
    print(f"Accuracy:      {acc:.1%} ({n_correct}/{len(items)})")
    print(f"Ceiling ratio: {ceiling:.1%} ({n_ceiling}/{n_incorrect} wrong samples)")
    print(f"Interpretation: {summary['ceiling_interpretation']}")
    print(f"Saved to {out}")
    print(f"SCORE: {acc * 100:.1f}")


if __name__ == "__main__":
    main()
