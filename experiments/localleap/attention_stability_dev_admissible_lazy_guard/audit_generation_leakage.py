#!/usr/bin/env python3
"""Reject hidden-target data paths in generation and in saved selector traces."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


VERSION = "generation_information_leakage_audit_v1"
FORBIDDEN_GENERATION_IDENTIFIERS = {
    "correct",
    "gold_answer",
    "hidden_test",
    "hidden_tests",
    "normalized_gold",
    "raw_gold",
    "reference_solution",
    "target_hash",
    "test_list",
}
FORBIDDEN_TRACE_KEYS = {
    "correct",
    "gold_answer",
    "hidden_test",
    "hidden_tests",
    "normalized_gold",
    "raw_gold",
    "reference_solution",
    "target_hash",
    "test_list",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identifiers(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.arg):
            found.add(node.arg)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
    return found, tree


def audit_sources(root):
    root = Path(root)
    audited = [root / "generate.py", root / "differential_selector.py"]
    violations = []
    hashes = {}
    for path in audited:
        names, _ = identifiers(path)
        bad = sorted(names & FORBIDDEN_GENERATION_IDENTIFIERS)
        if bad:
            violations.append({"file": str(path), "forbidden_identifiers": bad})
        hashes[str(path)] = sha256(path)

    eval_path = root / "eval_llada.py"
    _, tree = identifiers(eval_path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name != "generate":
            continue
        keyword_names = {item.arg for item in node.keywords if item.arg}
        forbidden = sorted(keyword_names & FORBIDDEN_GENERATION_IDENTIFIERS)
        if forbidden:
            violations.append({"file": str(eval_path), "generate_keywords": forbidden})
    hashes[str(eval_path)] = sha256(eval_path)
    return hashes, violations


def walk_trace(value, location, violations):
    if isinstance(value, dict):
        for key, child in value.items():
            next_location = f"{location}.{key}"
            if key in FORBIDDEN_TRACE_KEYS:
                violations.append({"location": next_location, "reason": "forbidden_key"})
            if key in {"uses_hidden_tests", "uses_reference_solution"} and child is not False:
                violations.append({"location": next_location, "reason": "must_be_false"})
            walk_trace(child, next_location, violations)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            walk_trace(child, f"{location}[{index}]", violations)


def audit_run(run_root, expected_profile):
    run_root = Path(run_root)
    config = (run_root / "run_config.txt").read_text(encoding="utf-8")
    if f"profile={expected_profile}\n" not in config:
        raise ValueError(f"profile mismatch in {run_root}")
    trace = run_root / "trace" / "rank_0.jsonl"
    violations = []
    count = 0
    with trace.open(encoding="utf-8") as handle:
        for count, line in enumerate(handle, start=1):
            walk_trace(json.loads(line), f"record[{count}]", violations)
    return {"run_root": str(run_root), "trace_records": count, "trace_sha256": sha256(trace)}, violations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--run-root")
    parser.add_argument("--expected-profile")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source_hashes, violations = audit_sources(args.source_root)
    run = None
    if args.run_root:
        if not args.expected_profile:
            raise SystemExit("--expected-profile is required with --run-root")
        run, run_violations = audit_run(args.run_root, args.expected_profile)
        violations.extend(run_violations)
    report = {
        "evaluator_version": VERSION,
        "source_hashes": source_hashes,
        "run": run,
        "violation_count": len(violations),
        "violations": violations,
        "pass": not violations,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()
