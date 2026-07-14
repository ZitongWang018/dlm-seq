import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-env", required=True)
    args = parser.parse_args()

    # Tie order is pre-registered: one-step central threshold, then neighboring
    # thresholds, then the more conservative two-step streak.
    configs = [
        ("he_vertical_eps0p01_s1", 0.01, 1, 0),
        ("he_vertical_eps0p005_s1", 0.005, 1, 1),
        ("he_vertical_eps0p02_s1", 0.02, 1, 2),
        ("he_vertical_eps0p01_s2", 0.01, 2, 3),
    ]
    rows = []
    root = Path(args.results_root)
    for tag, epsilon, streak, tie_rank in configs:
        path = root / tag / "audit" / "audit_summary.json"
        if not path.exists():
            raise SystemExit(f"missing completed audit: {path}")
        audit = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "run_tag": tag,
            "epsilon": epsilon,
            "streak": streak,
            "tie_rank": tie_rank,
            "correct": int(audit["correct"]),
            "total": int(audit["total"]),
            "accuracy": float(audit["accuracy"]),
        })
    winner = sorted(rows, key=lambda row: (-row["correct"], row["tie_rank"]))[0]
    output = {
        "selector_version": "stcc_humaneval_selector_v1",
        "selection_scope": "exploratory_humaneval_only",
        "tie_rule": "central epsilon streak1, neighboring epsilons, then streak2",
        "rows": rows,
        "winner": winner,
    }
    Path(args.output_json).write_text(json.dumps(output, indent=2), encoding="utf-8")
    Path(args.output_env).write_text(
        f"WINNER_EPS={winner['epsilon']}\nWINNER_STREAK={winner['streak']}\n"
        f"WINNER_TAG={winner['run_tag']}\n",
        encoding="utf-8",
    )
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
