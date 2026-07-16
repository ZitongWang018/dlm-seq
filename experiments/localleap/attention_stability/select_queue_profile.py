"""Auditable promotion rules for sampled experiment queues."""

import argparse
import json
import os


HEALTH_KEYS = (
    "prompt_hash_mismatches",
    "target_hash_mismatches",
    "duplicate_or_missing_ids",
)


def read_summary(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def healthy(summary):
    return all(summary.get(key) == 0 for key in HEALTH_KEYS)


def choose_best(candidates):
    available = []
    for profile, path in candidates:
        if not os.path.exists(path):
            continue
        summary = read_summary(path)
        if healthy(summary):
            available.append({"profile": profile, "summary": summary, "path": path})
    if not available:
        return {
            "profile": "none",
            "reason": "no_healthy_pair_summaries",
            "candidates": [],
        }

    def rank(item):
        summary = item["summary"]
        nfe = summary.get("nfe_ratio_method_over_baseline")
        wall = summary.get("wall_speedup_baseline_over_method")
        return (
            summary["method_correct"],
            int(item["profile"] == "symmetric_fast"),
            -(nfe if nfe is not None else 10**9),
            wall if wall is not None else -1,
        )

    winner = max(available, key=rank)
    return {
        "profile": winner["profile"],
        "reason": "correct_then_parent_tie_then_nfe_then_wall",
        "candidates": [
            {
                "profile": item["profile"],
                "correct": item["summary"]["method_correct"],
                "total": item["summary"]["total"],
                "nfe_ratio": item["summary"].get("nfe_ratio_method_over_baseline"),
                "wall_speedup": item["summary"].get(
                    "wall_speedup_baseline_over_method"
                ),
            }
            for item in available
        ],
    }


def beats(candidate, parent, required_gain=1):
    return (
        healthy(candidate)
        and healthy(parent)
        and candidate["method_correct"]
        >= parent["method_correct"] + required_gain
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    choose_parser = subparsers.add_parser("choose")
    choose_parser.add_argument("--output", required=True)
    choose_parser.add_argument("--candidate", action="append", required=True)
    beats_parser = subparsers.add_parser("beats")
    beats_parser.add_argument("candidate")
    beats_parser.add_argument("parent")
    beats_parser.add_argument("--required-gain", type=int, default=1)
    args = parser.parse_args()

    if args.command == "choose":
        candidates = [item.split("=", 1) for item in args.candidate]
        selected = choose_best(candidates)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(selected, handle, indent=2)
        print(selected["profile"])
        return

    candidate = read_summary(args.candidate)
    parent = read_summary(args.parent)
    raise SystemExit(
        0 if beats(candidate, parent, required_gain=args.required_gain) else 1
    )


if __name__ == "__main__":
    main()
