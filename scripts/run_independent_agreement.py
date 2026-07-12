#!/usr/bin/env python3
"""Test whether two independently prompted DLM derivations provide a useful answer agreement signal."""
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


PROMPTS = (
    "Solve this math word problem independently. Show the arithmetic, then give the final numeric answer.\nQuestion: {question}",
    "Re-derive the result from scratch for this problem. Do not assume any earlier answer. End with the correct number.\nProblem: {question}",
)


def build_result(output: list[dict], total_nfe: int, args: argparse.Namespace) -> dict:
    total = len(output)
    return {
        "total": total,
        "base_accuracy": sum(row["base_correct"] for row in output) / total,
        "accuracy": sum(row["correct"] for row in output) / total,
        "agreement_rate": sum(row["agree"] for row in output) / total,
        "trigger_rate": sum(row["triggered"] for row in output) / total,
        "avg_nfe": total_nfe / total,
        "risk_key": args.risk_key,
        "risk_threshold": args.risk_threshold,
        "temperature": args.temperature,
        "steps": args.steps,
        "gen_length": args.gen_length,
        "records": output,
    }


def generate(model, tokenizer, cfg: dict, prompt: str, steps: int, gen_length: int, temperature: float) -> tuple[str, int]:
    input_ids, attn = encode_prompt(tokenizer, prompt)
    out = generate_with_sampler(
        model, input_ids, attn, steps=steps, gen_length=gen_length, block_length=gen_length,
        temperature=temperature, mask_id=cfg["mask_id"], sampler="lcr", eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out["tokens"][0, input_ids.shape[1]:], skip_special_tokens=True)
    return text, int(out["nfe"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--risk_threshold", type=float, default=-1.0)
    parser.add_argument("--risk_key", default="response_selected_delta_mean")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    args = parser.parse_args()
    cfg = load_config("configs/default.yaml")
    base = json.load(open(args.base))
    records = base["records"]
    items = load_gsm8k(cfg["dataset_root"], limit=len(records), start_index=args.start_index)
    model, tokenizer = load_llada(cfg["model_path"])
    output = []
    total_nfe = 0
    for offset, (row, item) in enumerate(zip(records, items), start=1):
        risk_value = float(row.get(args.risk_key, 0.0) or 0.0)
        triggered = risk_value >= args.risk_threshold
        texts, preds, nfes = [], [], []
        if triggered:
            for template in PROMPTS:
                text, nfe = generate(model, tokenizer, cfg, template.format(question=item["question"]), args.steps, args.gen_length, args.temperature)
                texts.append(text)
                preds.append(extract_number(text))
                nfes.append(nfe)
        else:
            texts, preds, nfes = ["", ""], [None, None], [0, 0]
        agree = triggered and preds[0] is not None and preds[0] == preds[1]
        final_pred = preds[0] if agree else row.get("pred")
        correct = final_pred is not None and final_pred == item["answer"]
        total_nfe += int(row["nfe"]) + sum(nfes)
        output.append({
            "idx": row["idx"], "gold": item["answer"], "base_pred": row.get("pred"), "base_correct": bool(row["correct"]), "triggered": triggered,
            "risk_value": risk_value,
            "pred_a": preds[0], "pred_b": preds[1], "text_a": texts[0], "text_b": texts[1],
            "agree": agree, "agreed_pred": preds[0] if agree else None,
            "pred": final_pred, "correct": correct, "nfe_a": nfes[0], "nfe_b": nfes[1],
        })
        if offset % args.checkpoint_every == 0 or offset == len(records):
            result = build_result(output, total_nfe, args)
            result["complete"] = offset == len(records)
            result["processed"] = offset
            save_json(result, Path(args.out))
            print({
                "processed": offset,
                "triggered": sum(record["triggered"] for record in output),
                "agreements": sum(record["agree"] for record in output),
                "base_accuracy": result["base_accuracy"],
                "accuracy": result["accuracy"],
            }, flush=True)
    result = build_result(output, total_nfe, args)
    result["complete"] = True
    result["processed"] = len(output)
    save_json(result, Path(args.out))
    print({key: result[key] for key in ("base_accuracy", "accuracy", "agreement_rate", "avg_nfe")})


if __name__ == "__main__":
    main()
