#!/usr/bin/env python3
"""Build and reverify a compact offline evaluation artifact manifest."""

import argparse
import hashlib
import json
from pathlib import Path


VERSION = "localleap_offline_eval_protocol_manifest_v1"
INCLUDED_SUFFIXES = {
    ".arrow",
    ".json",
    ".jsonl",
    ".jinja",
    ".md",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}
MODEL_METADATA = {
    "config.json",
    "configuration.json",
    "configuration_llada.py",
    "generation_config.json",
    "model.safetensors.index.json",
    "modeling_llada.py",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
}


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def selected_files(roots, model_path):
    chosen = {}
    model_root = Path(model_path).resolve()
    for root_value in roots:
        root = Path(root_value).resolve()
        if not root.exists():
            raise ValueError(f"missing offline artifact root: {root}")
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            resolved = path.resolve()
            if model_root == resolved.parent:
                include = resolved.name in MODEL_METADATA
            else:
                include = resolved.suffix.lower() in INCLUDED_SUFFIXES
            if include:
                chosen[str(resolved)] = resolved
    for name in MODEL_METADATA:
        path = model_root / name
        if path.is_file():
            chosen[str(path.resolve())] = path.resolve()
    if not chosen:
        raise ValueError("offline manifest selected no files")
    return [chosen[key] for key in sorted(chosen)]


def build(args):
    paths = selected_files(args.root, args.model_path)
    rows = []
    arrow_count = 0
    for path in paths:
        if path.suffix.lower() == ".arrow":
            arrow_count += 1
        rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )
    if arrow_count < args.minimum_arrow_files:
        raise ValueError(
            f"expected at least {args.minimum_arrow_files} Arrow files, found {arrow_count}"
        )
    summary = {
        "schema": VERSION,
        "offline_only": True,
        "file_count": len(rows),
        "arrow_file_count": arrow_count,
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in summary if key != "files"}))


def verify(args):
    summary = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    failures = []
    for row in summary["files"]:
        path = Path(row["path"])
        if not path.is_file():
            failures.append({"path": str(path), "reason": "missing"})
            continue
        if path.stat().st_size != row["bytes"]:
            failures.append({"path": str(path), "reason": "size"})
            continue
        actual = digest(path)
        if actual != row["sha256"]:
            failures.append({"path": str(path), "reason": "sha256"})
    result = {
        "schema": VERSION,
        "manifest": str(Path(args.manifest).resolve()),
        "verified_files": len(summary["files"]) - len(failures),
        "failures": failures,
        "pass": not failures,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 2)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--root", action="append", required=True)
    build_parser.add_argument("--model-path", required=True)
    build_parser.add_argument("--minimum-arrow-files", type=int, default=4)
    build_parser.add_argument("--output", required=True)
    build_parser.set_defaults(func=build)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("manifest")
    verify_parser.add_argument("--output")
    verify_parser.set_defaults(func=verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
