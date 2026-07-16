"""Prism-aligned MATH-500 prompt and answer evaluator.

The extraction and normalization logic follows the official Prism evaluator
(Apache-2.0) while exposing a per-record lm-eval metric for independent audits.
"""

import math
import re
from typing import Dict, List

import datasets


EVALUATOR_VERSION = "localleap_math500_prism_aligned_v2"


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        return {**doc, "task_id": doc["unique_id"]}

    return dataset.map(_process_doc)


def math500_prompt(doc):
    system_prompt = (
        "You are a math expert. You will be given a question to solve. "
        "Solve it step by step. Wrap the final answer in a \\boxed{}. \n"
        "Respond in the following format:\n"
        "<reasoning>\n"
        "Your reasoning here\n"
        "</reasoning>\n"
        "<answer>\n"
        "\\boxed{...}\n"
        "</answer>"
    )
    return f"{system_prompt}\n\n{doc['problem']}\n\n"


def extract_answer(text):
    if not text:
        return "", False
    text = text.replace("<|role_end|>", "").replace("<|endoftext|>", "").strip()
    boxed_pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    boxes = re.findall(boxed_pattern, text)
    if boxes:
        return boxes[-1], True
    tag_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if tag_match:
        return tag_match.group(1).strip(), True
    marker = "the answer is"
    if marker in text.lower():
        position = text.lower().rfind(marker)
        answer = text[position + len(marker) :].strip()
        answer = re.sub(r"^[:\s]+", "", answer)
        return answer.split("\n")[0].split("$")[0].strip(), True
    tail = text[-50:].strip()
    numbers = re.findall(r"(-?\d+[\./\d]*|\\sqrt\{\d+\}|\(-?\d+.*?\))", tail)
    if numbers:
        return numbers[-1], False
    return "", False


def normalize_math(value):
    if not value:
        return ""
    value = str(value).lower().strip()
    value = value.replace("</reasoning>", "").replace("</answer>", "").replace("<answer>", "")
    value = value.replace("...", "").replace("cannot be determined", "")
    value = re.sub(r"([a-z]+|\\theta|\\alpha|\\pi)\s*=\s*", "", value)
    value = re.sub(r"\\text\{([^}]*)\}", r"\1", value)
    value = re.sub(r"\\(mathbf|mathrm|bold|unit|mbox|operatorname)\{([^}]*)\}", r"\2", value)
    value = re.sub(r"\\(d|t)?frac\{([^{}]*)\}\{([^{}]*)\}", r"\2/\3", value)
    value = value.replace("\\!", "").replace("\\ ", "").replace("{", "").replace("}", "")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "")
    units = r"(units?|cm\^2|cm|inches|inch|square|degrees?|radians?|miles?|per|hour|cents?)"
    value = re.sub(units, "", value)
    value = value.replace("^{\\circ}", "").replace("^\\circ", "").replace("°", "").replace("\\degree", "")
    value = value.replace("\\pi", "pi")
    value = re.sub(r"(\d),(\d{3})", r"\1\2", value)
    value = value.rstrip(".:,; ").replace(" ", "")
    if "=" in value:
        value = value.split("=")[-1]
    return value


def is_equiv(prediction, gold):
    if not prediction:
        return False
    prediction_norm = normalize_math(prediction)
    gold_norm = normalize_math(gold)
    if prediction_norm == gold_norm:
        return True
    try:
        def to_float(value):
            if "/" in value and value.count("/") == 1:
                numerator, denominator = value.split("/")
                return float(numerator) / float(denominator)
            if "_" in value:
                value = value.split("_")[0]
            return float(value)

        return math.isclose(
            to_float(prediction_norm), to_float(gold_norm), rel_tol=1e-4
        )
    except (TypeError, ValueError, ZeroDivisionError):
        prediction_fuzzy = re.sub(r"[^a-z0-9/,\-]", "", prediction_norm)
        gold_fuzzy = re.sub(r"[^a-z0-9/,\-]", "", gold_norm)
        return bool(prediction_fuzzy) and prediction_fuzzy == gold_fuzzy


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    prediction, _ = extract_answer(results[0])
    return {"exact_match": int(is_equiv(prediction, doc["answer"]))}

