#!/usr/bin/env python3
"""Add a third independent derivation and accept only a numerical majority on risk-triggered samples."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import save_json
from src.datasets import extract_number, load_gsm8k
from src.model_loader import encode_prompt, load_llada
from src.runner import load_config
from src.samplers import generate_with_sampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--gen_length", type=int, default=64)
    args = parser.parse_args()
    cfg = load_config("configs/default.yaml")
    prior = json.load(open(args.input))
    rows = prior["records"]
    items = load_gsm8k(cfg["dataset_root"], limit=len(rows), start_index=args.start_index)
    model, tokenizer = load_llada(cfg["model_path"])
    output, extra_nfe = [], 0
    for row, item in zip(rows, items):
        third_pred, third_text, nfe = None, "", 0
        if row["triggered"]:
            prompt = "Solve this math problem from scratch and end with the final numeric answer.\nQuestion: " + item["question"]
            input_ids, attn = encode_prompt(tokenizer, prompt)
            generated = generate_with_sampler(model, input_ids, attn, steps=args.steps, gen_length=args.gen_length, block_length=args.gen_length, temperature=args.temperature, mask_id=cfg["mask_id"], sampler="lcr", eos_token_id=tokenizer.eos_token_id)
            third_text = tokenizer.decode(generated["tokens"][0, input_ids.shape[1]:], skip_special_tokens=True)
            third_pred, nfe = extract_number(third_text), int(generated["nfe"])
        candidates = [pred for pred in (row["pred_a"], row["pred_b"], third_pred) if pred is not None]
        majority_pred, count = (Counter(candidates).most_common(1)[0] if candidates else (None, 0))
        accepted = count >= 2
        pred = majority_pred if accepted else row["base_pred"]
        correct = pred is not None and pred == item["answer"]
        extra_nfe += nfe
        output.append({**row, "third_pred": third_pred, "third_text": third_text, "third_nfe": nfe, "accepted": accepted, "pred": pred, "correct": correct})
    result = {"total": len(output), "base_accuracy": sum(row["base_correct"] for row in output) / len(output), "accuracy": sum(row["correct"] for row in output) / len(output), "accept_rate": sum(row["accepted"] for row in output) / len(output), "avg_nfe": prior["avg_nfe"] + extra_nfe / len(output), "records": output}
    save_json(result, Path(args.out))
    print({key: result[key] for key in ("base_accuracy", "accuracy", "accept_rate", "avg_nfe")})


if __name__ == "__main__":
    main()
