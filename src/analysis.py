"""Trajectory analysis and visualization."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.samplers import TrajectoryState


def summarize_trajectory(traj: TrajectoryState) -> dict[str, Any]:
    classes: dict[str, int] = {}
    ratios = []
    for pos in traj.first_conf:
        cls = traj.trajectory_type(pos)
        classes[cls] = classes.get(cls, 0) + 1
        recent = traj.recent_conf.get(pos, [0.0])
        net = recent[-1] - recent[0] if recent else 0.0
        path = traj.path_length.get(pos, 0.0)
        ratios.append(path / (abs(net) + 1e-6))

    coupling_vals = list(traj.coupling.values())
    return {
        "num_positions": len(traj.first_conf),
        "class_counts": classes,
        "mean_path_net_ratio": float(np.mean(ratios)) if ratios else 0.0,
        "coupling_mean": float(np.mean(coupling_vals)) if coupling_vals else 0.0,
        "coupling_sparsity": float(np.mean([v < 0.01 for v in coupling_vals])) if coupling_vals else 1.0,
        "num_steps": len(traj.step_records),
    }


def plot_path_scatter(traj: TrajectoryState, out_path: Path, title: str = "Distribution Path Geometry"):
    nets, paths, colors = [], [], []
    for pos in traj.first_conf:
        recent = traj.recent_conf.get(pos, [0.0])
        net = abs(recent[-1] - recent[0]) if recent else 0.0
        path = traj.path_length.get(pos, 0.0)
        nets.append(net)
        paths.append(path)
        colors.append(traj.trajectory_type(pos))

    plt.figure(figsize=(7, 5))
    palette = {"converging": "#2ecc71", "oscillating": "#e74c3c", "frozen": "#95a5a6", "mixed": "#f39c12", "unknown": "#bdc3c7"}
    for cls in set(colors):
        idx = [i for i, c in enumerate(colors) if c == cls]
        plt.scatter([nets[i] for i in idx], [paths[i] for i in idx], label=cls, alpha=0.7, s=30, c=palette.get(cls, "#333"))
    plt.xlabel("Net Confidence Change")
    plt.ylabel("Path Length (sum |delta conf|)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_coupling_heatmap(traj: TrajectoryState, out_path: Path, max_pos: int = 40):
    positions = sorted(traj.first_conf.keys())[:max_pos]
    if len(positions) < 2:
        return
    mat = np.zeros((len(positions), len(positions)))
    pos_to_i = {p: i for i, p in enumerate(positions)}
    for (src, dst), val in traj.coupling.items():
        if src in pos_to_i and dst in pos_to_i:
            mat[pos_to_i[src], pos_to_i[dst]] = val

    plt.figure(figsize=(8, 6))
    sns.heatmap(mat, cmap="mako", xticklabels=positions, yticklabels=positions)
    plt.xlabel("Response Position")
    plt.ylabel("Commit Position")
    plt.title("Confidence Response Matrix After Commit Events")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_trajectory_type_distribution(class_counts: dict[str, int], out_path: Path):
    labels = list(class_counts.keys())
    vals = [class_counts[k] for k in labels]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, vals, color=["#2ecc71", "#e74c3c", "#95a5a6", "#f39c12", "#3498db"][: len(labels)])
    plt.ylabel("Count")
    plt.xlabel("Trajectory Type")
    plt.title("Distribution Trajectory Types Across Positions")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def aggregate_observation_stats(summaries: list[dict]) -> dict[str, Any]:
    if not summaries:
        return {}
    total_classes: dict[str, int] = {}
    for s in summaries:
        for k, v in s.get("class_counts", {}).items():
            total_classes[k] = total_classes.get(k, 0) + v
    return {
        "num_samples": len(summaries),
        "aggregate_class_counts": total_classes,
        "mean_path_net_ratio": float(np.mean([s.get("mean_path_net_ratio", 0) for s in summaries])),
        "mean_coupling_sparsity": float(np.mean([s.get("coupling_sparsity", 0) for s in summaries])),
        "mean_coupling_strength": float(np.mean([s.get("coupling_mean", 0) for s in summaries])),
    }


def plot_method_comparison(results: dict[str, float], out_path: Path):
    methods = list(results.keys())
    accs = [results[m] for m in methods]
    plt.figure(figsize=(8, 4))
    bars = plt.bar(methods, accs, color=["#3498db", "#9b59b6", "#e67e22", "#2ecc71", "#1abc9c"][: len(methods)])
    plt.ylabel("Accuracy")
    plt.title("Method Comparison")
    plt.ylim(0, 1.0)
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{acc:.2%}", ha="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_json(data: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
