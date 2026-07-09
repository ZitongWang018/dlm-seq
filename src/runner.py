"""Evaluation runner."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from src.analysis import plot_coupling_heatmap, plot_method_comparison, plot_path_scatter, save_json, summarize_trajectory
from src.datasets import extract_code_block, extract_number, load_gsm8k, load_mbpp, mbpp_prompt, run_mbpp_tests
from src.model_loader import encode_prompt, load_llada
from src.samplers import generate_with_sampler


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def eval_gsm8k(model, tokenizer, cfg: dict, sampler: str, limit: int, track: bool = False, lateral: bool = False) -> dict[str, Any]:
    items = load_gsm8k(cfg["dataset_root"], limit=limit)
    correct = 0
    records = []
    traj_summaries = []

    for idx, item in enumerate(tqdm(items, desc=f"GSM8K-{sampler}")):
        input_ids, attn = encode_prompt(tokenizer, item["question"])
        out = generate_with_sampler(
            model, input_ids, attn,
            steps=cfg["steps"], gen_length=cfg["gen_length"], block_length=cfg["block_length"],
            temperature=cfg["temperature"], mask_id=cfg["mask_id"],
            sampler=sampler, track_trajectory=track, top_k_track=cfg.get("top_k_track", 32),
            lateral=lateral,
        )
        gen_text = tokenizer.decode(out["tokens"][0, input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_number(gen_text)
        is_correct = pred is not None and pred == item["answer"]
        correct += int(is_correct)
        rec = {"idx": idx, "correct": is_correct, "pred": pred, "gold": item["answer"], "nfe": out["nfe"]}
        records.append(rec)
        if out["trajectory"] is not None:
            traj_summaries.append(summarize_trajectory(out["trajectory"]))

    return {
        "dataset": "gsm8k",
        "sampler": sampler,
        "accuracy": correct / len(items),
        "correct": correct,
        "total": len(items),
        "records": records,
        "trajectory_summaries": traj_summaries,
    }


def eval_mbpp(model, tokenizer, cfg: dict, sampler: str, limit: int, track: bool = False, lateral: bool = False) -> dict[str, Any]:
    items = load_mbpp(cfg["dataset_root"], limit=limit)
    correct = 0
    records = []

    for idx, item in enumerate(tqdm(items, desc=f"MBPP-{sampler}")):
        prompt = mbpp_prompt(item)
        input_ids, attn = encode_prompt(tokenizer, prompt)
        gen_len = cfg.get("mbpp_gen_length", 256)
        mbpp_steps = cfg.get("mbpp_steps", gen_len)
        mbpp_block = cfg.get("mbpp_block_length", gen_len)
        out = generate_with_sampler(
            model, input_ids, attn,
            steps=mbpp_steps, gen_length=gen_len, block_length=mbpp_block,
            temperature=cfg["temperature"], mask_id=cfg["mask_id"],
            sampler=sampler, track_trajectory=track, top_k_track=cfg.get("top_k_track", 32),
            lateral=lateral,
        )
        gen_text = tokenizer.decode(out["tokens"][0, input_ids.shape[1]:], skip_special_tokens=True)
        code = extract_code_block(gen_text)
        passed = run_mbpp_tests(code, item["test_list"], item.get("test_setup_code", ""))
        correct += int(passed)
        records.append({"idx": idx, "task_id": item["task_id"], "correct": passed, "nfe": out["nfe"]})

    return {
        "dataset": "mbpp",
        "sampler": sampler,
        "accuracy": correct / len(items),
        "correct": correct,
        "total": len(items),
        "records": records,
    }


def run_experiment(config_path: str, methods: list[str], datasets: list[str], run_tag: str):
    cfg = load_config(config_path)
    results_root = Path(cfg["results_root"]) / run_tag
    results_root.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_llada(cfg["model_path"])
    all_metrics = {}

    for dataset in datasets:
        limit = cfg["gsm8k_limit"] if dataset == "gsm8k" else cfg["mbpp_limit"]
        for method in methods:
            lateral = method == "traj_lateral"
            sampler = "traj" if method.startswith("traj") else method
            track = method.startswith("traj")
            t0 = time.time()
            if dataset == "gsm8k":
                result = eval_gsm8k(model, tokenizer, cfg, sampler, limit, track=track, lateral=lateral)
            else:
                result = eval_mbpp(model, tokenizer, cfg, sampler, limit, track=track, lateral=lateral)
            result["elapsed_sec"] = time.time() - t0
            save_json(result, results_root / f"{dataset}_{method}.json")
            all_metrics[f"{dataset}_{method}"] = result["accuracy"]

    plot_method_comparison(all_metrics, results_root / "method_comparison.png")
    save_json(all_metrics, results_root / "summary.json")
    print(f"Results saved to {results_root}")
    return all_metrics
