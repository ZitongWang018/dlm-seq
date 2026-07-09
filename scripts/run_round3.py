#!/usr/bin/env python3
"""Round 3: rerun improved traj on GSM8K and complete MBPP baselines."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runner import run_experiment

if __name__ == "__main__":
    # Force overwrite traj results with hybrid v2 scoring
    import json
    from src.analysis import plot_method_comparison, save_json
    from src.runner import load_config, eval_gsm8k, eval_mbpp
    from src.model_loader import load_llada

    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    tag = "round3_traj_v2"
    out = Path(cfg["results_root"]) / tag
    out.mkdir(parents=True, exist_ok=True)

    # seed from round2
    prev = Path(cfg["results_root"]) / "round2_methods"
    metrics = {}
    if (prev / "summary.json").exists():
        with open(prev / "summary.json") as f:
            metrics = json.load(f)
    for name in ["gsm8k_lcr", "gsm8k_rcr"]:
        p = prev / f"{name}.json"
        if p.exists():
            with open(p) as f:
                metrics[name] = json.load(f)["accuracy"]

    model, tok = load_llada(cfg["model_path"])
    for method, lateral in [("traj", False), ("traj", True)]:
        suffix = "traj_lateral" if lateral else "traj"
        key = f"gsm8k_{suffix}"
        r = eval_gsm8k(model, tok, cfg, "traj", cfg["gsm8k_limit"], lateral=lateral)
        save_json(r, out / f"{key}.json")
        metrics[key] = r["accuracy"]
        print(f"{key}: {r['accuracy']:.2%}")

    for method in ["lcr", "traj", "traj_lateral"]:
        key = f"mbpp_{method}"
        prev_p = prev / f"{key}.json"
        if prev_p.exists() and method != "traj":
            with open(prev_p) as f:
                metrics[key] = json.load(f)["accuracy"]
            print(f"reuse {key}")
            continue
        lateral = method == "traj_lateral"
        sampler = "traj" if method.startswith("traj") else method
        r = eval_mbpp(model, tok, cfg, sampler, cfg["mbpp_limit"], lateral=lateral)
        save_json(r, out / f"{key}.json")
        metrics[key] = r["accuracy"]
        print(f"{key}: {r['accuracy']:.2%}")

    plot_method_comparison(metrics, out / "method_comparison.png")
    save_json(metrics, out / "summary.json")
    print(f"round3 done -> {out}")
