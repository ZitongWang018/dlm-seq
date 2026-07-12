#!/usr/bin/env python3
"""Run a short second-pass numerical verifier on trajectory-selected GSM8K outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import save_json
from src.datasets import extract_number, load_gsm8k
from src.model_loader import encode_prompt, load_llada
from src.runner import load_config
from src.samplers import generate_with_sampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--risk_threshold", type=float, default=-1.0)
    parser.add_argument("--risk_key", default="response_selected_delta_mean")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--gen_length", type=int, default=32)
    args = parser.parse_args()

    cfg = load_config("configs/default.yaml")
    base = json.load(open(args.base))
    records = base["records"]
    items = load_gsm8k(cfg["dataset_root"], limit=len(records), start_index=cfg.get("gsm8k_start_index", 0))
    model, tokenizer = load_llada(cfg["model_path"])
    output = []
    total_nfe = 0
    for record, item in zip(records, items):
        risk = float(record.get(args.risk_key, 0.0) or 0.0)
        triggered = record.get("pred") is not None and risk >= args.risk_threshold
        verifier_pred = None
        verifier_text = ""
        nfe = 0
        if triggered:
            prompt = (
                f"Question: {item['question']}\n"
                f"A proposed final answer is {record['pred']}.\n"
                "Check the arithmetic independently. Return only the correct final number."
            )
            input_ids, attn = encode_prompt(tokenizer, prompt)
            out = generate_with_sampler(
                model, input_ids, attn, steps=args.steps, gen_length=args.gen_length,
                block_length=args.gen_length, temperature=0.0, mask_id=cfg["mask_id"], sampler="lcr",
                eos_token_id=tokenizer.eos_token_id,
            )
            verifier_text = tokenizer.decode(out["tokens"][0, input_ids.shape[1]:], skip_special_tokens=True)
            verifier_pred = extract_number(verifier_text)
            nfe = out["nfe"]
        final_pred = verifier_pred if verifier_pred is not None else record.get("pred")
        correct = final_pred is not None and final_pred == item["answer"]
        total_nfe += int(record["nfe"]) + nfe
        output.append({
            "idx": record["idx"], "gold": item["answer"], "base_pred": record.get("pred"),
            "base_correct": bool(record["correct"]), "risk": risk, "triggered": triggered,
            "verifier_pred": verifier_pred, "verifier_text": verifier_text, "verifier_nfe": nfe,
            "pred": final_pred, "correct": correct,
        })
    result = {
        "base": args.base, "risk_key": args.risk_key, "risk_threshold": args.risk_threshold,
        "total": len(output), "accuracy": sum(row["correct"] for row in output) / len(output),
        "base_accuracy": sum(row["base_correct"] for row in output) / len(output),
        "trigger_rate": sum(row["triggered"] for row in output) / len(output),
        "avg_nfe": total_nfe / len(output), "records": output,
    }
    save_json(result, Path(args.out))
    print({key: result[key] for key in ("accuracy", "base_accuracy", "trigger_rate", "avg_nfe")})


if __name__ == "__main__":
    main()
