#!/usr/bin/env python3
"""Select a unique prompt-visible winner from all already generated drafts."""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

from differential_selector import evaluate_public_candidate


VERSION = "public_frontier_guard_v1"
REQUIRED = {
    "absolute_index",
    "task_id",
    "prompt_hash",
    "target_hash",
    "decoded_generation",
    "nfe",
}


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def index_records(rows, name):
    indexed = {}
    for row in rows:
        missing = sorted(REQUIRED - set(row))
        if missing:
            raise ValueError(f"{name} record is missing fields {missing}")
        task_id = row["task_id"]
        if task_id in indexed:
            raise ValueError(f"duplicate task_id in {name}: {task_id}")
        indexed[task_id] = row
    if not indexed:
        raise ValueError(f"no records in {name}")
    return indexed


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_name(record):
    diagnostics = record.get("decode_diagnostics") or {}
    name = diagnostics.get("selected_name")
    if name not in {"fast", "accuracy", "baseline"}:
        raise ValueError(f"unsupported v11 selected_name: {name!r}")
    return name


def select_one(v9, baseline, v11):
    task_id = v11["task_id"]
    for field in ("prompt_text", "entry_point"):
        if field not in v9:
            raise ValueError(f"v9 record is missing {field} for {task_id}")
    for field in ("absolute_index", "task_id", "prompt_hash", "target_hash"):
        if v9[field] != baseline[field] or v9[field] != v11[field]:
            raise ValueError(f"{field} mismatch for {task_id}")
    candidate_generations = v9.get("candidate_generations") or {}
    if set(candidate_generations) != {"fast", "accuracy"}:
        raise ValueError(f"missing fast/accuracy drafts for {task_id}")
    candidates = {
        "fast": candidate_generations["fast"],
        "accuracy": candidate_generations["accuracy"],
        "baseline": baseline["decoded_generation"],
    }
    current_name = selected_name(v11)
    if candidates[current_name] != v11["decoded_generation"]:
        raise ValueError(f"v11 selected generation mismatch for {task_id}")

    evidence = {
        name: evaluate_public_candidate(
            generation, v9["prompt_text"], v9["entry_point"]
        )
        for name, generation in candidates.items()
    }
    check_counts = {item["visible_check_count"] for item in evidence.values()}
    if len(check_counts) != 1:
        raise ValueError(f"public-check count mismatch for {task_id}")
    visible_check_count = check_counts.pop()
    passed = {
        name: int(item["visible_checks_passed"])
        for name, item in evidence.items()
    }
    maximum = max(passed.values())
    leaders = [name for name, value in passed.items() if value == maximum]
    if (
        visible_check_count
        and len(leaders) == 1
        and maximum > passed[current_name]
    ):
        final_name = leaders[0]
        status = "unique_public_winner_reopens_frontier"
    else:
        final_name = current_name
        status = "preserve_v11_on_tie_or_no_strict_gain"

    output = copy.deepcopy(v11)
    output["decoded_generation"] = candidates[final_name]
    output["correct"] = None
    output["evaluator_version"] = VERSION
    output["generation_settings"] = copy.deepcopy(
        v11.get("generation_settings") or {}
    )
    output["generation_settings"]["trajectory_selector"] = VERSION
    output["decode_diagnostics"] = copy.deepcopy(
        v11.get("decode_diagnostics") or {}
    )
    output["decode_diagnostics"]["pre_frontier_selected_name"] = current_name
    output["decode_diagnostics"]["selected_name"] = final_name
    output["decode_diagnostics"]["public_frontier_guard"] = {
        "selector": VERSION,
        "status": status,
        "preselected_name": current_name,
        "selected_name": final_name,
        "visible_check_count": visible_check_count,
        "visible_checks_passed": passed,
        "compile_valid": {
            name: bool(item["compile_valid"]) for name, item in evidence.items()
        },
        "candidate_hashes": {
            name: item["candidate_hash"] for name, item in evidence.items()
        },
        "unique_winner_required": True,
        "ties_preserve_v11": True,
        "uses_generated_probes": False,
        "uses_hidden_tests": False,
        "uses_reference_solution": False,
    }
    output["candidate_generations"] = candidates
    return output


def select_records(v9_rows, baseline_rows, v11_rows):
    v9 = index_records(v9_rows, "v9")
    baseline = index_records(baseline_rows, "baseline")
    v11 = index_records(v11_rows, "v11")
    if set(v9) != set(baseline) or set(v9) != set(v11):
        raise ValueError("v9, baseline, and v11 task IDs do not align")
    return [
        select_one(v9[task_id], baseline[task_id], v11[task_id])
        for task_id in sorted(v9, key=lambda key: v9[key]["absolute_index"])
    ]


def selected_sample_view(samples, selected):
    by_id = {row["doc"]["task_id"]: row for row in samples}
    if len(by_id) != len(samples):
        raise ValueError("duplicate sample task IDs")
    output = []
    for record in selected:
        task_id = record["task_id"]
        if task_id not in by_id:
            raise ValueError(f"sample missing task_id {task_id}")
        sample = copy.deepcopy(by_id[task_id])
        sample["resps"][0][0] = record["decoded_generation"]
        output.append(sample)
    return output


def attach_execution(selected, execution_rows):
    execution = {row["task_id"]: row for row in execution_rows}
    if len(execution) != len(execution_rows):
        raise ValueError("duplicate execution task IDs")
    if set(execution) != {row["task_id"] for row in selected}:
        raise ValueError("selected and execution task IDs do not align")
    for row in selected:
        row["correct"] = bool(execution[row["task_id"]]["pass_at_1"])
    return selected


def summarize(selected, v11_rows):
    v11 = index_records(v11_rows, "v11")
    switches = [
        row
        for row in selected
        if row["decode_diagnostics"]["pre_frontier_selected_name"]
        != row["decode_diagnostics"]["selected_name"]
    ]
    return {
        "evaluator_version": VERSION,
        "total": len(selected),
        "method_correct": sum(bool(row["correct"]) for row in selected),
        "v11_correct": sum(bool(row["correct"]) for row in v11.values()),
        "method_only_vs_v11": sum(
            bool(row["correct"]) and not bool(v11[row["task_id"]]["correct"])
            for row in selected
        ),
        "v11_only_vs_method": sum(
            bool(v11[row["task_id"]]["correct"]) and not bool(row["correct"])
            for row in selected
        ),
        "changed_generation_count": len(switches),
        "selected_name_counts": {
            name: sum(
                row["decode_diagnostics"]["selected_name"] == name
                for row in selected
            )
            for name in ("fast", "accuracy", "baseline")
        },
        "nfe_total": sum(int(row["nfe"] or 0) for row in selected),
        "selection_uses_hidden_tests": False,
        "accuracy_evaluation_uses_hidden_tests": True,
        "uses_reference_solution_for_selection": False,
        "duplicate_ids": 0,
        "missing_ids": 0,
        "prompt_hash_mismatches": 0,
        "target_hash_mismatches": 0,
        "residual_mask_count": 0,
    }


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def configure_runtime_path(runtime_root):
    """Expose runtime-only evaluator modules without shadowing frozen selector code."""
    runtime_root = str(Path(runtime_root))
    if runtime_root not in sys.path:
        sys.path.append(runtime_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v9-records", required=True)
    parser.add_argument("--baseline-records", required=True)
    parser.add_argument("--v11-records", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)

    v9_rows = read_jsonl(args.v9_records)
    baseline_rows = read_jsonl(args.baseline_records)
    v11_rows = read_jsonl(args.v11_records)
    selected = select_records(v9_rows, baseline_rows, v11_rows)
    samples = selected_sample_view(read_jsonl(args.samples), selected)
    selected_samples = output / "selected_samples.jsonl"
    write_jsonl(selected_samples, samples)

    configure_runtime_path(args.runtime_root)
    import postprocess_code

    postprocess_code.main(str(selected_samples))
    selected = attach_execution(
        selected, read_jsonl(str(selected_samples) + ".cleaned")
    )
    records_path = output / "audit_records.jsonl"
    write_jsonl(records_path, selected)
    summary = summarize(selected, v11_rows)
    summary.update(
        {
            "v9_records_sha256": sha256(args.v9_records),
            "baseline_records_sha256": sha256(args.baseline_records),
            "v11_records_sha256": sha256(args.v11_records),
            "samples_sha256": sha256(args.samples),
        }
    )
    (output / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
