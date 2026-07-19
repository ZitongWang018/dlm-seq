#!/usr/bin/env python3
"""Verify that two LocalLeap runs differ only in algorithm-specific options."""

import argparse
import json
from pathlib import Path


VERSION = "localleap_run_config_fairness_v1"
TOP_LEVEL_FIELDS = (
    "task",
    "steps",
    "gen_length",
    "baseline_budget_per_step",
    "expected_records",
    "num_fewshot",
    "seeds",
)
MODEL_FIELDS = (
    "model_path",
    "gen_length",
    "steps",
    "block_length",
    "remasking",
    "early_stop",
    "show_speed",
    "integrate_speed",
)


def read_config(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    if "model_args" not in result:
        raise ValueError(f"missing model_args in {path}")
    return result


def parse_model_args(value):
    result = {}
    for item in value.split(","):
        if "=" not in item:
            raise ValueError(f"invalid model argument: {item}")
        key, argument = item.split("=", 1)
        if key in result:
            raise ValueError(f"duplicate model argument: {key}")
        result[key] = argument
    return result


def compare(reference, candidate):
    reference_model = parse_model_args(reference["model_args"])
    candidate_model = parse_model_args(candidate["model_args"])
    differences = []
    for field in TOP_LEVEL_FIELDS:
        if reference.get(field) != candidate.get(field):
            differences.append(
                {
                    "scope": "run_config",
                    "field": field,
                    "reference": reference.get(field),
                    "candidate": candidate.get(field),
                }
            )
    for field in MODEL_FIELDS:
        if reference_model.get(field) != candidate_model.get(field):
            differences.append(
                {
                    "scope": "model_args",
                    "field": field,
                    "reference": reference_model.get(field),
                    "candidate": candidate_model.get(field),
                }
            )
    algorithm_arguments = {
        key: value
        for key, value in candidate_model.items()
        if key.startswith("dependency_")
    }
    unexpected_candidate_arguments = sorted(
        set(candidate_model) - set(reference_model) - set(algorithm_arguments)
    )
    missing_reference_arguments = sorted(
        key for key in reference_model if key not in candidate_model
    )
    if unexpected_candidate_arguments:
        differences.append(
            {
                "scope": "model_args",
                "field": "unexpected_candidate_arguments",
                "candidate": unexpected_candidate_arguments,
            }
        )
    if missing_reference_arguments:
        differences.append(
            {
                "scope": "model_args",
                "field": "missing_reference_arguments",
                "candidate": missing_reference_arguments,
            }
        )
    return {
        "schema": VERSION,
        "all_equal_core": not differences,
        "reference_profile": reference.get("profile"),
        "candidate_profile": candidate.get("profile"),
        "algorithm_arguments": algorithm_arguments,
        "core_differences": differences,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(read_config(args.reference), read_config(args.candidate))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["all_equal_core"] else 2)


if __name__ == "__main__":
    main()
