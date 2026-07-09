#!/usr/bin/env python3
"""Resume method comparison from checkpoint."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import plot_method_comparison, save_json
from src.runner import load_config, eval_gsm8k, eval_mbpp
from src.model_loader import load_llada

RUN_TAG = sys.argv[1] if len(sys.argv) > 1 else "round2_methods"
METHODS = sys.argv[2:] if len(sys.argv) > 2 else ["rcr", "traj", "traj_lateral"]
DATASETS = ["gsm8k", "mbpp"]


def main():
    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    out_dir = Path(cfg["results_root"]) / RUN_TAG
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    model, tokenizer = load_llada(cfg["model_path"])

    for dataset in DATASETS:
        limit = cfg["gsm8k_limit"] if dataset == "gsm8k" else cfg["mbpp_limit"]
        for method in METHODS:
            key = f"{dataset}_{method}"
            result_path = out_dir / f"{dataset}_{method}.json"
            if result_path.exists():
                with open(result_path) as f:
                    all_metrics[key] = json.load(f)["accuracy"]
                print(f"skip existing {key}")
                continue

            lateral = method == "traj_lateral"
            sampler = "traj" if method.startswith("traj") else method
            if dataset == "gsm8k":
                result = eval_gsm8k(model, tokenizer, cfg, sampler, limit, track=False, lateral=lateral)
            else:
                result = eval_mbpp(model, tokenizer, cfg, sampler, limit, track=False, lateral=lateral)
            save_json(result, result_path)
            all_metrics[key] = result["accuracy"]
            print(f"{key}: {result['accuracy']:.2%}")

    plot_method_comparison(all_metrics, out_dir / "method_comparison.png")
    save_json(all_metrics, summary_path)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
