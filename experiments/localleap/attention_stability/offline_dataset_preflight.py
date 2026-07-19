#!/usr/bin/env python3
"""Prove that all benchmark views and tokenizer metadata load with no socket use."""

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path


VERSION = "localleap_offline_dataset_preflight_v1"


def canonical_hash(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def forbid_network():
    def blocked(*args, **kwargs):
        raise RuntimeError("network access is forbidden by offline preflight")

    socket.create_connection = blocked
    original_socket = socket.socket

    class OfflineSocket(original_socket):
        def connect(self, *args, **kwargs):
            blocked(*args, **kwargs)

        def connect_ex(self, *args, **kwargs):
            blocked(*args, **kwargs)

    socket.socket = OfflineSocket


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    required_environment = {
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    for key, expected in required_environment.items():
        if os.environ.get(key) != expected:
            raise SystemExit(f"{key} must be {expected}")

    forbid_network()
    from datasets import load_dataset
    from transformers import AutoConfig, AutoTokenizer

    cases = (
        ("humaneval", "openai/openai_humaneval", None, "test", 164),
        ("math500", "HuggingFaceH4/MATH-500", None, "test", 500),
        ("gsm8k", "openai/gsm8k", "main", "test", 1319),
        ("mbpp", "google-research-datasets/mbpp", "full", "test", 500),
    )
    datasets = {}
    for name, repository, config, split, expected in cases:
        view = load_dataset(repository, config, split=split)
        if len(view) != expected:
            raise ValueError(f"{name}: expected {expected}, found {len(view)}")
        rows = [dict(row) for row in view]
        datasets[name] = {
            "repository": repository,
            "config": config,
            "split": split,
            "records": len(rows),
            "columns": list(view.column_names),
            "dataset_view_hash": canonical_hash(rows),
        }
    config = AutoConfig.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    tokenizer_probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "offline protocol probe"}],
        add_generation_prompt=True,
        tokenize=False,
    )
    token_ids = tokenizer(tokenizer_probe)["input_ids"]
    result = {
        "schema": VERSION,
        "pass": True,
        "network_socket_calls_allowed": False,
        "environment": required_environment,
        "datasets": datasets,
        "model_config_class": type(config).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_probe_text_hash": hashlib.sha256(tokenizer_probe.encode()).hexdigest(),
        "tokenizer_probe_token_ids_hash": canonical_hash(token_ids),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
