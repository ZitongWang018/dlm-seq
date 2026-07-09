"""Download MATH (algebra+number_theory level1-3) and HumanEval via hf-mirror."""
from __future__ import annotations
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset, concatenate_datasets, DatasetDict
from datasets import Dataset
import json

DATASET_ROOT = Path("/root/autodl-tmp/dataset")

MATH_SUBSETS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]


def download_math():
    """Download all MATH subsets, keep level 1-3 as dev split, level 4-5 as test."""
    print("Downloading MATH (all subsets)...")
    all_test = []
    for sub in MATH_SUBSETS:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", sub, split="test")
            for item in ds:
                all_test.append({
                    "problem": item["problem"],
                    "solution": item["solution"],
                    "level": item["level"],        # "Level 1" ... "Level 5"
                    "type": sub,
                })
            print(f"  {sub}: {len(ds)} test items")
        except Exception as e:
            print(f"  {sub}: SKIP ({e})")

    # Dev = level 1-3, Test = level 4-5
    dev = [x for x in all_test if int(x["level"].split()[-1]) <= 3]
    test = [x for x in all_test if int(x["level"].split()[-1]) > 3]
    print(f"MATH total: {len(all_test)}, dev (L1-3): {len(dev)}, test (L4-5): {len(test)}")

    out = DATASET_ROOT / "math"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "dev.json", "w") as f:
        json.dump(dev, f, indent=2)
    with open(out / "test.json", "w") as f:
        json.dump(test, f, indent=2)
    print(f"Saved MATH to {out}/dev.json ({len(dev)}) and test.json ({len(test)})")


def download_humaneval():
    """Download HumanEval (164 problems)."""
    print("Downloading HumanEval...")
    ds = load_dataset("openai/openai_humaneval", split="test")
    out = DATASET_ROOT / "humaneval"
    out.mkdir(parents=True, exist_ok=True)
    items = [dict(row) for row in ds]
    with open(out / "test.json", "w") as f:
        json.dump(items, f, indent=2)
    print(f"HumanEval: {len(items)} problems → {out}/test.json")


if __name__ == "__main__":
    download_math()
    download_humaneval()
    print("All done.")
