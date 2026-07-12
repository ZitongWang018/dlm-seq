#!/usr/bin/env python3
"""Evaluate LCR against the local spacing commit rule."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import plot_method_comparison, save_json
from src.model_loader import load_llada
from src.runner import eval_gsm8k, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--out_dir", default="results/round_reframe_lcr_spaced_20")
    parser.add_argument("--methods", nargs="+", default=["lcr", "lcr_spaced"])
    parser.add_argument("--response_weight", type=float, default=None)
    parser.add_argument("--response_cap", type=float, default=None)
    parser.add_argument("--response_conf_max", type=float, default=None)
    parser.add_argument("--response_min_delta", type=float, default=None)
    parser.add_argument("--response_local_window", type=int, default=None)
    parser.add_argument("--response_refresh_threshold", type=float, default=None)
    parser.add_argument("--response_lookahead_threshold", type=float, default=None)
    parser.add_argument("--response_lookahead_max_steps", type=int, default=None)
    parser.add_argument("--response_budget_threshold", type=float, default=None)
    parser.add_argument("--response_budget_factor", type=float, default=None)
    parser.add_argument("--response_persistence_max_drift", type=float, default=None)
    parser.add_argument("--wavefront_size", type=int, default=None)
    parser.add_argument("--wavefront_radius", type=int, default=None)
    parser.add_argument("--terminal_refine_tokens", type=int, default=None)
    parser.add_argument("--terminal_refine_threshold", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--block_length", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config("configs/default.yaml")
    if args.response_weight is not None:
        cfg["response_weight"] = args.response_weight
    if args.response_cap is not None:
        cfg["response_cap"] = args.response_cap
    if args.response_conf_max is not None:
        cfg["response_conf_max"] = args.response_conf_max
    if args.response_min_delta is not None:
        cfg["response_min_delta"] = args.response_min_delta
    if args.response_local_window is not None:
        cfg["response_local_window"] = args.response_local_window
    if args.response_refresh_threshold is not None:
        cfg["response_refresh_threshold"] = args.response_refresh_threshold
    if args.response_lookahead_threshold is not None:
        cfg["response_lookahead_threshold"] = args.response_lookahead_threshold
    if args.response_lookahead_max_steps is not None:
        cfg["response_lookahead_max_steps"] = args.response_lookahead_max_steps
    if args.response_budget_threshold is not None:
        cfg["response_budget_threshold"] = args.response_budget_threshold
    if args.response_budget_factor is not None:
        cfg["response_budget_factor"] = args.response_budget_factor
    if args.response_persistence_max_drift is not None:
        cfg["response_persistence_max_drift"] = args.response_persistence_max_drift
    if args.wavefront_size is not None:
        cfg["wavefront_size"] = args.wavefront_size
    if args.wavefront_radius is not None:
        cfg["wavefront_radius"] = args.wavefront_radius
    if args.terminal_refine_tokens is not None:
        cfg["terminal_refine_tokens"] = args.terminal_refine_tokens
    if args.terminal_refine_threshold is not None:
        cfg["terminal_refine_threshold"] = args.terminal_refine_threshold
    if args.temperature is not None:
        cfg["temperature"] = args.temperature
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.block_length is not None:
        cfg["block_length"] = args.block_length
    cfg["gsm8k_start_index"] = args.start_index
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_llada(cfg["model_path"])
    metrics = {}
    for method in args.methods:
        result = eval_gsm8k(model, tokenizer, cfg, method, limit=args.limit, track=False, lateral=False)
        save_json(result, out_dir / f"gsm8k_{method}.json")
        metrics[f"gsm8k_{method}"] = result["accuracy"]

    save_json(metrics, out_dir / "summary.json")
    plot_method_comparison(metrics, out_dir / "method_comparison.png")
    print(metrics)


if __name__ == "__main__":
    main()
