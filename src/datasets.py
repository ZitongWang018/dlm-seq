"""Dataset loading and answer extraction helpers."""
from __future__ import annotations

import re
from pathlib import Path

from datasets import load_dataset, load_from_disk


def load_gsm8k(dataset_root: str, split: str = "test", limit: int | None = None, start_index: int = 0):
    path = Path(dataset_root) / "gsm8k"
    ds = load_from_disk(str(path))[split] if path.exists() else load_dataset("openai/gsm8k", "main", split=split)
    if start_index < 0 or start_index >= len(ds):
        raise ValueError(f"start_index must be in [0, {len(ds) - 1}], got {start_index}")
    if limit:
        ds = ds.select(range(start_index, min(start_index + limit, len(ds))))
    items = []
    for row in ds:
        answer = re.sub(r"[^0-9.\-]", "", row["answer"].split("####")[-1].strip())
        items.append({"question": row["question"], "answer": answer})
    return items


def load_mbpp(dataset_root: str, split: str = "test", limit: int | None = None):
    path = Path(dataset_root) / "mbpp"
    ds = load_from_disk(str(path))[split] if path.exists() else load_dataset("google-research-datasets/mbpp", "full", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [{"task_id": row.get("task_id", i), "prompt": row["text"], "test_list": row["test_list"], "test_setup_code": row.get("test_setup_code", ""), "reference": row["code"]} for i, row in enumerate(ds)]


def extract_number(text: str) -> str | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def answer_evidence(text: str, pred: str | None) -> dict:
    """Small decoded-text checks for whether the numerical answer settled at the end."""
    normalized = text.replace(",", "")
    tail = normalized[-400:]
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    last_line = lines[-1] if lines else ""
    numbers = re.findall(r"-?\d+\.?\d*", tail)
    pred_pattern = re.escape(pred) + r"(?![0-9.])" if pred is not None else r"(?!)"
    return {
        "answer_pred_count_tail": len(re.findall(pred_pattern, tail)),
        "answer_pred_in_last_line": bool(re.search(pred_pattern, last_line)),
        "answer_marker_in_tail": bool(re.search(r"(?i)\b(answer|therefore|thus|so)\b", tail)),
        "answer_distinct_numbers_tail": len(set(numbers)),
        "answer_text_tail": text[-600:],
    }


def mbpp_prompt(item: dict) -> str:
    fn = extract_fn_name(item["test_list"])
    fn_name = fn or "solution"
    return f"Implement the Python function `{fn_name}`.\nTask: {item['prompt']}\nExample test: {item['test_list'][0]}\nReturn only valid Python code defining `{fn_name}`."


def extract_fn_name(test_list: list[str]) -> str | None:
    match = re.search(r"assert\s+(\w+)\(", test_list[0])
    return match.group(1) if match else None


def extract_code_block(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    lines = text.strip().splitlines()
    code_lines = []
    for line in lines:
        if line.startswith(("def ", "from ", "import ")) or code_lines:
            code_lines.append(line)
    return "\n".join(code_lines).strip() if code_lines else text.strip()


def run_mbpp_tests(code: str, test_list: list[str], setup: str = "") -> bool:
    namespace: dict = {}
    try:
        if setup:
            exec(setup, namespace)
        exec(code, namespace)
        for test in test_list:
            exec(test, namespace)
        return True
    except Exception:
        return False
