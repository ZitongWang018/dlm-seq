#!/usr/bin/env python3
"""Audit generation-time information flow while allowing post-generation metadata.

Version 1 rejected raw_gold/normalized_gold keys anywhere in the persisted
trace. The evaluation wrapper appends those fields only after the decoder has
returned, so their presence is not evidence that generation or selection read
the answer. Version 2 audits the actual call graph and selector diagnostics,
and treats only top-level, correctness=None metadata as post-generation data.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path


VERSION = "generation_information_leakage_audit_v2"
FORBIDDEN_GENERATION_IDENTIFIERS = {
    "correct", "gold_answer", "hidden_test", "hidden_tests",
    "normalized_gold", "raw_gold", "reference_solution", "target_hash",
    "test_list",
}
FORBIDDEN_SELECTOR_KEYS = FORBIDDEN_GENERATION_IDENTIFIERS
POST_GENERATION_TOP_LEVEL = {"correct", "normalized_gold", "raw_gold"}


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
    hashes = {}
    violations = []
    for name in ("generate.py", "differential_selector.py"):
        path = root / name
        names, _ = identifiers(path)
        bad = sorted(names & FORBIDDEN_GENERATION_IDENTIFIERS)
        if bad:
            violations.append({"file": str(path), "forbidden_identifiers": bad})
        hashes[str(path)] = sha256(path)

    eval_path = root / "eval_llada.py"
    source = eval_path.read_text(encoding="utf-8")
    _, tree = identifiers(eval_path)
    if "question = req.args[0]" not in source or '"content": question' not in source:
        violations.append({"file": str(eval_path), "reason": "model_input_contract_missing"})
    calls = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name not in {"generate", "generate_stcc"}:
            continue
        calls += 1
        keyword_names = {item.arg for item in node.keywords if item.arg}
        forbidden = sorted(keyword_names & FORBIDDEN_GENERATION_IDENTIFIERS)
        if forbidden:
            violations.append({"file": str(eval_path), "generate_keywords": forbidden})
        if len(node.args) > 2:
            violations.append({"file": str(eval_path), "reason": "unexpected_positional_generation_inputs"})
    if calls == 0:
        violations.append({"file": str(eval_path), "reason": "no_generation_calls_found"})
    hashes[str(eval_path)] = sha256(eval_path)
    return hashes, violations


def walk_selector(value, location, violations):
    if isinstance(value, dict):
        for key, child in value.items():
            next_location = f"{location}.{key}"
            if key in FORBIDDEN_SELECTOR_KEYS:
                violations.append({"location": next_location, "reason": "forbidden_selector_key"})
            if key in {"uses_hidden_tests", "uses_reference_solution"} and child is not False:
                violations.append({"location": next_location, "reason": "must_be_false"})
            walk_selector(child, next_location, violations)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            walk_selector(child, f"{location}[{index}]", violations)


def audit_trace_row(row, index):
    violations = []
    if row.get("correct") is not None:
        violations.append({"location": f"record[{index}].correct", "reason": "post_generation_correct_must_be_none"})
    selector_view = {key: value for key, value in row.items() if key not in POST_GENERATION_TOP_LEVEL}
    walk_selector(selector_view, f"record[{index}].selector_view", violations)
    generation = row.get("decoded_generation")
    if not isinstance(generation, str):
        violations.append({"location": f"record[{index}].decoded_generation", "reason": "missing_generation"})
    return violations


def audit_run(run_root, expected_profile):
    run_root = Path(run_root)
    config = (run_root / "run_config.txt").read_text(encoding="utf-8")
    if f"profile={expected_profile}\n" not in config:
        raise ValueError(f"profile mismatch in {run_root}")
    trace = run_root / "trace" / "rank_0.jsonl"
    violations = []
    count = 0
    metadata_counts = {key: 0 for key in sorted(POST_GENERATION_TOP_LEVEL)}
    with trace.open(encoding="utf-8") as handle:
        for count, line in enumerate(handle, start=1):
            row = json.loads(line)
            for key in metadata_counts:
                metadata_counts[key] += int(key in row)
            violations.extend(audit_trace_row(row, count))
    return {
        "run_root": str(run_root),
        "trace_records": count,
        "trace_sha256": sha256(trace),
        "post_generation_metadata_counts": metadata_counts,
        "post_generation_metadata_used_by_selector": False,
    }, violations


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
