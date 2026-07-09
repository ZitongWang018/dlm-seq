"""Run LCR baseline on MATH (level 1-3) to establish new benchmark point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets_v2 import load_math, math_prompt, check_math_answer, extract_math_answer
from src.model_loader import load_llada, encode_prompt
from src.samplers import generate_with_sampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--model_path", default="/root/autodl-tmp/model/LLaDA/instruct")
    parser.add_argument("--dataset_root", default="/root/autodl-tmp/dataset")
    parser.add_argument("--out", default="results/math_baseline/math_lcr.json")
    args = parser.parse_args()

    print(f"Loading model ...")
    model, tokenizer = load_llada(args.model_path)

    print(f"Loading MATH {args.split} (limit={args.limit}) ...")
    items = load_math(args.dataset_root, split=args.split, limit=args.limit)

    records = []
    correct = 0
    for idx, item in enumerate(items):
        print(f"[{idx+1}/{len(items)}] level={item['level']}", end="\r", flush=True)
        prompt = math_prompt(item)
        input_ids, attn = encode_prompt(tokenizer, prompt)
        out = generate_with_sampler(
            model, input_ids, attn,
            steps=args.steps, gen_length=args.gen_length,
            block_length=args.gen_length, temperature=0.0,
            mask_id=126336, sampler="lcr",
        )
        gen_text = tokenizer.decode(out["tokens"][0, input_ids.shape[1]:], skip_special_tokens=True)
        is_correct = check_math_answer(gen_text, item["solution"])
        correct += int(is_correct)
        records.append({
            "idx": idx, "correct": is_correct, "level": item["level"],
            "type": item["type"], "gen_text": gen_text[:300],
            "gold": extract_math_answer(item["solution"]),
        })

    acc = correct / max(len(items), 1)
    print(f"\nMATH-{args.split} LCR accuracy: {acc:.1%} ({correct}/{len(items)})")
    print(f"SCORE: {acc * 100:.1f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"accuracy": acc, "correct": correct, "total": len(items), "records": records}, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
