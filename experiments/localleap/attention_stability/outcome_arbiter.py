"""Leakage-free outcome arbitration for complete reasoning trajectories."""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from tasks.localleap_math500.utils import extract_answer, normalize_math


_GSM_STRICT = re.compile(r"####\s*(\-?[0-9\.\,]+)")
_GSM_FLEXIBLE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def infer_outcome_family(doc: Dict[str, object]) -> Optional[str]:
    if doc.get("unique_id") is not None and doc.get("problem") is not None:
        return "math"
    answer = doc.get("answer")
    if doc.get("question") is not None and isinstance(answer, str) and "####" in answer:
        return "gsm"
    return None


def normalize_outcome(text: str, family: str) -> str:
    if family == "math":
        value, _ = extract_answer(text)
        return normalize_math(value)
    if family != "gsm":
        raise ValueError(f"unsupported outcome family: {family}")
    strict = _GSM_STRICT.findall(text or "")
    if strict:
        value = strict[-1]
    else:
        flexible = _GSM_FLEXIBLE.findall(text or "")
        if not flexible:
            return ""
        pair = flexible[-1]
        value = pair[0] or pair[1]
    return value.replace("$", "").replace(",", "").rstrip(".").strip()


def build_outcome_arbiter_prompt(question: str, outcomes: Dict[str, str]) -> str:
    unique = sorted({value for value in outcomes.values() if value})
    if len(unique) < 2:
        raise ValueError("outcome arbitration requires at least two unique answers")
    candidates = "\n".join(
        f"Candidate answer {index}: {value}"
        for index, value in enumerate(unique, start=1)
    )
    return (
        "Solve the original problem independently. Several complete reasoning "
        "trajectories proposed the candidate final answers below; any of them "
        "may be wrong. Check the problem rather than voting by frequency. "
        "Return exactly one final answer wrapped in \\boxed{}.\n\n"
        f"Original problem:\n{question}\n\n"
        f"Candidate final answers:\n{candidates}\n"
    )


def select_arbitrated_candidate(
    parent_name: str,
    candidate_outcomes: Dict[str, str],
    arbiter_outcome: str,
) -> Tuple[str, Dict[str, object]]:
    matches = sorted(
        name
        for name, outcome in candidate_outcomes.items()
        if arbiter_outcome and outcome == arbiter_outcome
    )
    selected = parent_name
    status = "unmatched_or_empty_arbiter"
    if parent_name in matches:
        status = "arbiter_confirms_parent"
    elif matches:
        attention_matches = [name for name in ("fast", "accuracy") if name in matches]
        selected = attention_matches[0] if attention_matches else matches[0]
        status = "arbiter_selects_existing_alternative"
    return selected, {
        "selector": "explicit_outcome_set_arbiter_v1",
        "parent_name": parent_name,
        "selected_name": selected,
        "arbiter_outcome": arbiter_outcome,
        "candidate_outcomes": dict(candidate_outcomes),
        "matching_candidates": matches,
        "status": status,
        "creates_novel_selected_answer": False,
        "uses_hidden_tests": False,
        "uses_reference_solution": False,
    }
