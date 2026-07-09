"""Dataset loading and answer extraction."""
from __future__ import annotations

import re
from pathlib import Path

from datasets import load_dataset, load_from_disk


def load_gsm8k(dataset_root: str, split: str = "test", limit: int | None = None):
    path = Path(dataset_root) / "gsm8k"
    if path.exists():
        ds = load_from_disk(str(path))[split]
    else:
        ds = load_dataset("openai/gsm8k", "main", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    items = []
    for row in ds:
        answer = row["answer"].split("####")[-1].strip()
        answer = re.sub(r"[^0-9.\-]", "", answer)
        items.append({"question": row["question"], "answer": answer})
    return items


def load_mbpp(dataset_root: str, split: str = "test", limit: int | None = None):
    path = Path(dataset_root) / "mbpp"
    if path.exists():
        ds = load_from_disk(str(path))[split]
    else:
        ds = load_dataset("google-research-datasets/mbpp", "full", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    items = []
    for row in ds:
        items.append({
            "task_id": row.get("task_id", len(items)),
            "prompt": row["text"],
            "test_list": row["test_list"],
            "test_setup_code": row.get("test_setup_code", ""),
            "reference": row["code"],
        })
    return items


def extract_number(text: str) -> str | None:
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    if not nums:
        return None
    return nums[-1]


def mbpp_prompt(item: dict) -> str:
    fn = extract_fn_name(item["test_list"])
    fn_name = fn or "solution"
    return (
        f"Implement the Python function `{fn_name}`.\n"
        f"Task: {item['prompt']}\n"
        f"Example test: {item['test_list'][0]}\n"
        f"Return only valid Python code defining `{fn_name}`."
    )


def extract_fn_name(test_list: list[str]) -> str | None:
    m = re.search(r"assert\s+(\w+)\(", test_list[0])
    return m.group(1) if m else None


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
