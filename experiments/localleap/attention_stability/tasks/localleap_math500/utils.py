"""MATH-500 task helpers using lm-eval's established MATH normalization."""

from typing import Dict, List

import datasets
from lm_eval.tasks.hendrycks_math.utils import (
    is_equiv,
    last_boxed_only_string,
    remove_boxed,
)


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Attach the stable dataset identity used by traces and paired audits."""

    def _process_doc(doc: dict) -> dict:
        return {
            **doc,
            "task_id": doc["unique_id"],
        }

    return dataset.map(_process_doc)


def extract_answer(text: str) -> str:
    """Extract a final math answer without treating absent answers as correct."""

    boxed = last_boxed_only_string(text)
    if boxed is not None:
        return remove_boxed(boxed)

    dollar_positions = [index for index, char in enumerate(text) if char == "$"]
    if len(dollar_positions) >= 2:
        return text[dollar_positions[-2] + 1 : dollar_positions[-1]]

    lowered = text.lower()
    for marker in ("the final answer is", "the answer is", "answer:"):
        marker_index = lowered.rfind(marker)
        if marker_index >= 0:
            return text[marker_index + len(marker) :].strip().rstrip(".")
    return text.strip().rstrip(".")


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    prediction = extract_answer(results[0])
    return {"exact_match": int(is_equiv(prediction, doc["answer"]))}

