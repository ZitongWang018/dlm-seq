#!/usr/bin/env python3
"""Download evaluation datasets to /root/autodl-tmp/dataset/."""
import os
from pathlib import Path

# Use HF mirror for AutoDL environment
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/.cache/huggingface")

from datasets import load_dataset

DATASET_ROOT = Path("/root/autodl-tmp/dataset")
DATASET_ROOT.mkdir(parents=True, exist_ok=True)


def main():
    print("Downloading GSM8K...")
    gsm = load_dataset("openai/gsm8k", "main")
    gsm.save_to_disk(str(DATASET_ROOT / "gsm8k"))

    print("Downloading MBPP...")
    mbpp = load_dataset("google-research-datasets/mbpp", "full")
    mbpp.save_to_disk(str(DATASET_ROOT / "mbpp"))

    print(f"Datasets saved to {DATASET_ROOT}")


if __name__ == "__main__":
    main()
