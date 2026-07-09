"""Evaluate v2 samplers (ids, jac, ids+) vs LCR baseline on GSM8K."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import load_gsm8k, extract_number
from src.model_loader import load_llada, encode_prompt
from src.samplers_v2 import generate_v2


def eval_gsm8k(model, tokenizer, sampler: str, limit: int, steps: int = 64,
               gen_length: int = 128, block_length: int = 128,
               stability_window: int = 4, top_k: int = 8) -> dict:
    items = load_gsm8k("/root/autodl-tmp/dataset", limit=limit)
    correct = 0
    records = []

    for idx, item in enumerate(items):
        print(f"[{idx+1}/{len(items)}] sampler={sampler}", end="\r", flush=True)
        input_ids, attn = encode_prompt(tokenizer, item["question"])
        out = generate_v2(
            model, input_ids, attn,
            steps=steps, gen_length=gen_length, block_length=block_length,
            temperature=0.0, mask_id=126336,
            sampler=sampler, top_k=top_k, stability_window=stability_window,
        )
        gen_text = tokenizer.decode(out["tokens"][0, input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_number(gen_text)
        is_correct = pred is not None and pred.strip() == item["answer"].strip()
        correct += int(is_correct)
        records.append({"idx": idx, "correct": is_correct, "pred": pred, "gold": item["answer"]})

    acc = correct / max(len(items), 1)
    print(f"\nSampler={sampler} | Accuracy={acc:.1%} ({correct}/{len(items)})")
    print(f"SCORE: {acc * 100:.1f}")
    return {"sampler": sampler, "accuracy": acc, "correct": correct, "total": len(items), "records": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samplers", nargs="+", default=["lcr", "ids", "jac", "ids+"])
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--model_path", default="/root/autodl-tmp/model/LLaDA/instruct")
    parser.add_argument("--out_dir", default="results/round3_v2")
    args = parser.parse_args()

    print(f"Loading {args.model_path} ...")
    model, tokenizer = load_llada(args.model_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for s in args.samplers:
        res = eval_gsm8k(model, tokenizer, s, args.limit, args.steps,
                         args.gen_length, args.gen_length, args.window, args.top_k)
        results[s] = res["accuracy"]
        with open(out_dir / f"gsm8k_{s}.json", "w") as f:
            json.dump(res, f, indent=2)

    print("\n=== Summary ===")
    for s, acc in results.items():
        marker = " ← BASELINE" if s == "lcr" else (" ✓ BEAT" if acc > results.get("lcr", 0) else " ✗")
        print(f"  {s:8s}: {acc:.1%}{marker}")

    with open(out_dir / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
