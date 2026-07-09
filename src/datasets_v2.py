"""Extended dataset loaders: MATH (level 1-3 dev) and HumanEval."""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


# ── MATH ──────────────────────────────────────────────────────────────────────

def load_math(dataset_root: str, split: str = "dev", limit: int | None = None) -> list[dict]:
    """Load MATH dataset from local JSON (downloaded by download_math_humaneval.py).

    split: 'dev'  → level 1-3 problems (accessible, useful for iteration)
           'test' → level 4-5 problems (harder, final held-out)
    """
    path = Path(dataset_root) / "math" / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"MATH dataset not found at {path}. Run scripts/download_math_humaneval.py first.")
    with open(path) as f:
        items = json.load(f)
    if limit:
        items = items[:limit]
    return items


def extract_math_answer(solution: str) -> str | None:
    """Extract the final boxed answer from MATH solution text."""
    # Look for \boxed{...} at the end
    matches = re.findall(r"\\boxed\{([^}]+)\}", solution)
    if matches:
        return matches[-1].strip()
    # Fallback: last number
    nums = re.findall(r"-?\d+\.?\d*", solution)
    return nums[-1] if nums else None


def math_prompt(item: dict) -> str:
    return (
        f"Solve the following math problem step by step. "
        f"End your answer with \\boxed{{answer}}.\n\n"
        f"Problem: {item['problem']}"
    )


def check_math_answer(pred_text: str, gold_solution: str) -> bool:
    """Check if predicted answer matches gold boxed answer."""
    gold = extract_math_answer(gold_solution)
    if gold is None:
        return False
    # Try boxed in pred
    pred_boxes = re.findall(r"\\boxed\{([^}]+)\}", pred_text)
    if pred_boxes:
        pred = pred_boxes[-1].strip()
        return _normalize_math(pred) == _normalize_math(gold)
    # Fallback: numeric match
    pred_nums = re.findall(r"-?\d+\.?\d*", pred_text)
    if not pred_nums:
        return False
    gold_nums = re.findall(r"-?\d+\.?\d*", gold)
    if not gold_nums:
        return False
    try:
        return abs(float(pred_nums[-1]) - float(gold_nums[-1])) < 1e-6
    except ValueError:
        return pred_nums[-1] == gold_nums[-1]


def _normalize_math(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace(",", "")
    return s


# ── HumanEval ─────────────────────────────────────────────────────────────────

def load_humaneval(dataset_root: str, limit: int | None = None) -> list[dict]:
    path = Path(dataset_root) / "humaneval" / "test.json"
    if not path.exists():
        raise FileNotFoundError(f"HumanEval not found at {path}. Run scripts/download_math_humaneval.py first.")
    with open(path) as f:
        items = json.load(f)
    if limit:
        items = items[:limit]
    return items


def humaneval_prompt(item: dict) -> str:
    """Prompt for HumanEval: ask for complete function implementation."""
    return (
        f"Complete the following Python function. Return only the full function definition.\n\n"
        f"{item['prompt']}"
    )


def run_humaneval_tests(code: str, item: dict, timeout: int = 10) -> bool:
    """Execute the generated code against HumanEval test cases."""
    full_code = f"{code}\n\n{item['test']}\n\ncheck({item['entry_point']})"
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            fname = f.name
        result = subprocess.run(
            ["python", fname], capture_output=True, timeout=timeout
        )
        return result.returncode == 0
    except Exception:
        return False
    finally:
        Path(fname).unlink(missing_ok=True)
