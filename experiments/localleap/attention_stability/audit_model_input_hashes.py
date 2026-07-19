#!/usr/bin/env python3
"""Freeze the exact model input and evaluation view for one lm-eval sample file.

The LocalLeap wrapper always applies the local tokenizer chat template for the
Instruct checkpoint. Historical records only hashed the pre-template task
prompt. This auditor reconstructs the actual text and token ids used by the
wrapper, records both hashes, and can strictly compare two arms by stable id.
It never reads reference answers outside the sample artifact.
"""

import argparse
import hashlib
import json
from pathlib import Path


VERSION = "localleap_model_input_lineage_v1"


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_text(value):
    return sha256_bytes(value.encode("utf-8"))


def canonical_hash(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return sha256_text(payload)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prompt_from_sample(sample):
    arguments = sample.get("arguments", {})
    for value in arguments.values():
        if isinstance(value, dict) and "arg_0" in value:
            return str(value["arg_0"])
    doc = sample.get("doc", {})
    for key in ("prompt", "problem", "text"):
        if doc.get(key) is not None:
            return str(doc[key])
    raise ValueError("sample has no generation prompt")


def stable_id_from_sample(sample, index):
    doc = sample.get("doc", {})
    for key in ("task_id", "id", "problem_id", "unique_id"):
        if doc.get(key) is not None:
            return str(doc[key])
    return f"index_{sample.get('doc_id', index)}"


def tokenizer_file_hashes(model_path):
    names = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    )
    result = {}
    root = Path(model_path)
    for name in names:
        path = root / name
        if path.is_file():
            result[name] = sha256_bytes(path.read_bytes())
    if not result:
        raise ValueError(f"no tokenizer/config files found under {root}")
    return result


def model_weight_hashes(model_path):
    root = Path(model_path)
    paths = sorted(root.glob("*.safetensors"))
    if not paths:
        raise ValueError(f"no safetensors found under {root}")
    return {path.name: sha256_bytes(path.read_bytes()) for path in paths}


def read_weight_manifest(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        if len(digest) != 64:
            raise ValueError(f"invalid sha256 line: {line}")
        result[Path(name.strip()).name] = digest
    if not result:
        raise ValueError(f"empty weight manifest: {path}")
    return result


def build_records(samples, tokenizer, is_instruct=True):
    records = []
    seen = set()
    for index, sample in enumerate(samples):
        stable_id = stable_id_from_sample(sample, index)
        if stable_id in seen:
            raise ValueError(f"duplicate stable id: {stable_id}")
        seen.add(stable_id)
        raw_prompt = prompt_from_sample(sample)
        if is_instruct:
            model_input = tokenizer.apply_chat_template(
                [{"role": "user", "content": raw_prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            model_input = raw_prompt
        token_ids = tokenizer(model_input)["input_ids"]
        doc = sample.get("doc", {})
        target = sample.get("target")
        records.append(
            {
                "stable_task_id": stable_id,
                "absolute_index": index,
                "raw_prompt_hash": sha256_text(raw_prompt),
                "model_input_text_hash": sha256_text(model_input),
                "model_input_token_ids_hash": canonical_hash(token_ids),
                "model_input_token_count": len(token_ids),
                "document_hash": canonical_hash(doc),
                "target_hash": canonical_hash(target),
            }
        )
    return records


def index_records(rows):
    indexed = {}
    for row in rows:
        stable_id = row["stable_task_id"]
        if stable_id in indexed:
            raise ValueError(f"duplicate stable id: {stable_id}")
        indexed[stable_id] = row
    return indexed


def compare_records(reference, candidate):
    left = index_records(reference)
    right = index_records(candidate)
    missing = sorted(set(left) - set(right))
    extra = sorted(set(right) - set(left))
    fields = (
        "raw_prompt_hash",
        "model_input_text_hash",
        "model_input_token_ids_hash",
        "model_input_token_count",
        "document_hash",
        "target_hash",
    )
    mismatches = []
    for stable_id in sorted(set(left) & set(right)):
        changed = [field for field in fields if left[stable_id][field] != right[stable_id][field]]
        if changed:
            mismatches.append({"stable_task_id": stable_id, "fields": changed})
    return {
        "evaluator_version": VERSION,
        "reference_count": len(left),
        "candidate_count": len(right),
        "missing_ids": missing,
        "extra_ids": extra,
        "mismatches": mismatches,
        "all_equal": not missing and not extra and not mismatches,
    }


def write_audit(args):
    from transformers import AutoTokenizer

    samples = read_jsonl(args.samples)
    if len(samples) != args.expected_records:
        raise SystemExit(
            f"expected {args.expected_records} samples, found {len(samples)}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    records = build_records(samples, tokenizer, is_instruct=not args.base_model)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "model_input_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    sorted_rows = sorted(records, key=lambda row: row["stable_task_id"])
    summary = {
        "evaluator_version": VERSION,
        "sample_file": str(Path(args.samples).resolve()),
        "sample_file_sha256": sha256_bytes(Path(args.samples).read_bytes()),
        "record_count": len(records),
        "duplicate_ids": 0,
        "dataset_view_hash": canonical_hash(
            [
                {
                    "stable_task_id": row["stable_task_id"],
                    "document_hash": row["document_hash"],
                    "target_hash": row["target_hash"],
                }
                for row in sorted_rows
            ]
        ),
        "raw_prompt_set_hash": canonical_hash(
            [(row["stable_task_id"], row["raw_prompt_hash"]) for row in sorted_rows]
        ),
        "model_input_text_set_hash": canonical_hash(
            [(row["stable_task_id"], row["model_input_text_hash"]) for row in sorted_rows]
        ),
        "model_input_token_set_hash": canonical_hash(
            [
                (row["stable_task_id"], row["model_input_token_ids_hash"])
                for row in sorted_rows
            ]
        ),
        "tokenizer_file_hashes": tokenizer_file_hashes(args.model_path),
        "model_weight_hashes": (
            read_weight_manifest(args.model_weight_manifest)
            if args.model_weight_manifest
            else model_weight_hashes(args.model_path)
        ),
    }
    (output / "model_input_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


def write_comparison(args):
    summary = compare_records(read_jsonl(args.reference), read_jsonl(args.candidate))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    raise SystemExit(0 if summary["all_equal"] else 2)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--samples", required=True)
    audit.add_argument("--model-path", required=True)
    audit.add_argument("--expected-records", type=int, required=True)
    audit.add_argument("--output-dir", required=True)
    audit.add_argument("--base-model", action="store_true")
    audit.add_argument("--model-weight-manifest")
    audit.set_defaults(func=write_audit)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--reference", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--output", required=True)
    compare.set_defaults(func=write_comparison)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
