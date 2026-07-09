#!/usr/bin/env python3
"""Run observation study with trajectory recording and visualizations."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from tqdm import tqdm

from src.analysis import (
    aggregate_observation_stats,
    plot_coupling_heatmap,
    plot_path_scatter,
    plot_trajectory_type_distribution,
    save_json,
    summarize_trajectory,
)
from src.datasets import load_gsm8k
from src.model_loader import encode_prompt, load_llada
from src.samplers import generate_with_sampler


def main():
    with open(ROOT / "configs" / "default.yaml") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["results_root"]) / "round1_observation"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_llada(cfg["model_path"])
    items = load_gsm8k(cfg["dataset_root"], limit=cfg.get("observation_limit", 10))
    summaries = []

    for idx, item in enumerate(tqdm(items, desc="Observation")):
        input_ids, attn = encode_prompt(tokenizer, item["question"])
        out = generate_with_sampler(
            model, input_ids, attn,
            steps=cfg["steps"], gen_length=cfg["gen_length"], block_length=cfg["block_length"],
            temperature=cfg["temperature"], mask_id=cfg["mask_id"],
            sampler="lcr", track_trajectory=True, top_k_track=cfg.get("top_k_track", 32),
        )
        traj = out["trajectory"]
        if traj is None:
            continue
        summary = summarize_trajectory(traj)
        summaries.append(summary)
        plot_path_scatter(traj, out_dir / f"path_scatter_{idx:03d}.png", title=f"Sample {idx} Path Geometry")
        plot_coupling_heatmap(traj, out_dir / f"coupling_{idx:03d}.png")

    save_json(summaries, out_dir / "trajectory_summaries.json")
    agg = aggregate_observation_stats(summaries)
    save_json(agg, out_dir / "aggregate_stats.json")
    plot_trajectory_type_distribution(agg.get("aggregate_class_counts", {}), out_dir / "trajectory_type_distribution.png")
    print(f"Observation results saved to {out_dir}")
    print("Aggregate stats:", agg)


if __name__ == "__main__":
    main()
