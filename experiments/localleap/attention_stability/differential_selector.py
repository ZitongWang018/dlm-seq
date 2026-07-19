"""Leakage-free differential selection for multiple generated Python drafts.

The selector may inspect the user-visible prompt, but never benchmark reference
solutions or hidden tests.  It compiles every candidate, evaluates prompt-visible
doctest examples when present, and compares deterministic behavior on small
type-derived probes.  Selection is model-free and deterministic.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple


_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_EXAMPLE_RE = re.compile(r">>>\s*(.+?)\s*\n\s*([^\n]+)")
_ASSERT_RE = re.compile(r"(?m)^\s*assert\s+(.+?)\s*$")


def extract_python_code(text: str) -> str:
    fenced = _FENCE_RE.findall(text or "")
    if fenced:
        return max(fenced, key=len).strip()
    match = re.search(r"(?m)^(?:async\s+def|def|class)\s+", text or "")
    return (text[match.start() :] if match else (text or "")).strip()


def _function(tree: ast.AST, entry_point: Optional[str]) -> Optional[ast.FunctionDef]:
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    if entry_point:
        for node in functions:
            if node.name == entry_point:
                return node
    return functions[0] if functions else None


def _annotation_family(annotation: Optional[ast.AST]) -> str:
    if annotation is None:
        return "int"
    name = ast.unparse(annotation).lower()
    if "bool" in name:
        return "bool"
    if "str" in name:
        return "str"
    if "float" in name:
        return "float"
    if "list" in name or "sequence" in name or "tuple" in name:
        if "str" in name:
            return "list_str"
        return "list_int"
    return "int"


_VALUES = {
    "int": [0, 1, -1, 2, 5, 10],
    "float": [0.0, 1.0, -1.0, 2.5, 10.0, 0.5],
    "bool": [False, True, False, True, False, True],
    "str": ["", "a", "ab", "aba", "01", "hello"],
    "list_int": [[], [0], [1, 2], [-1, 0, 1], [2, 2], [5, 1, 3]],
    "list_str": [[], ["a"], ["a", "b"], ["", "a"], ["aa", "b"], ["0", "1"]],
}


def build_type_probes(code: str, entry_point: Optional[str], limit: int = 6) -> List[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    function = _function(tree, entry_point)
    if function is None:
        return []
    name = entry_point or function.name
    arguments = list(function.args.posonlyargs) + list(function.args.args)
    if arguments and arguments[0].arg in {"self", "cls"}:
        return []
    required = max(0, len(arguments) - len(function.args.defaults))
    arguments = arguments[: max(required, 1)] if arguments else []
    if not arguments:
        return [f"{name}()"]
    families = [_annotation_family(argument.annotation) for argument in arguments]
    probes = []
    for index in range(limit):
        values = [_VALUES[family][index % len(_VALUES[family])] for family in families]
        probes.append(f"{name}({', '.join(repr(value) for value in values)})")
    return probes


def prompt_examples(prompt: str, entry_point: Optional[str]) -> List[Tuple[str, str]]:
    examples = []
    for expression, expected in _EXAMPLE_RE.findall(prompt or ""):
        expression = expression.strip()
        expected = expected.strip()
        if entry_point and not re.search(rf"\b{re.escape(entry_point)}\s*\(", expression):
            continue
        if expected.startswith(">>>"):
            continue
        examples.append((expression, expected))
    return examples


def prompt_assertions(prompt: str, entry_point: Optional[str]) -> List[str]:
    """Return only assertions in the current prompt segment.

    MBPP includes solved few-shot programs before the current task.  The last
    ``[DONE]`` delimiter separates those demonstrations from the public tests
    for the current request, so earlier assertions must never vote.
    """
    current = (prompt or "").rsplit("[DONE]", 1)[-1]
    assertions = [expression.strip() for expression in _ASSERT_RE.findall(current)]
    if entry_point:
        assertions = [
            expression
            for expression in assertions
            if re.search(rf"\b{re.escape(entry_point)}\s*\(", expression)
        ]
    return assertions


def has_public_checks(prompt: str, entry_point: Optional[str]) -> bool:
    return bool(
        prompt_examples(prompt, entry_point)
        or prompt_assertions(prompt, entry_point)
    )


_DRIVER = r"""
import json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
try:
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
except Exception:
    pass
payload = json.loads(sys.stdin.read())
namespace = {}
try:
    exec(payload["code"], namespace, namespace)
except BaseException as error:
    print(json.dumps({"compile": False, "error": type(error).__name__, "outputs": []}))
    raise SystemExit(0)
outputs = []
for expression in payload["expressions"]:
    try:
        value = eval(expression, namespace, namespace)
        outputs.append({"ok": True, "value": repr(value)})
    except BaseException as error:
        outputs.append({"ok": False, "value": type(error).__name__})
print(json.dumps({"compile": True, "error": None, "outputs": outputs}))
"""


def execute_expressions(code: str, expressions: Sequence[str], timeout: float = 1.5) -> Dict[str, Any]:
    payload = json.dumps({"code": code, "expressions": list(expressions)})
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", _DRIVER],
            input=payload,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        line = result.stdout.strip().splitlines()[-1]
        return json.loads(line)
    except (subprocess.TimeoutExpired, IndexError, json.JSONDecodeError):
        return {"compile": False, "error": "timeout_or_protocol", "outputs": []}


def _normalize_expected(text: str) -> str:
    try:
        return repr(ast.literal_eval(text))
    except (ValueError, SyntaxError):
        return text.strip()


def select_differential_candidate(
    candidates: Sequence[str],
    prompt: str,
    entry_point: Optional[str] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Return the selected candidate index and independently auditable signals."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    codes = [extract_python_code(candidate) for candidate in candidates]
    examples = prompt_examples(prompt, entry_point)
    base = next((code for code in codes if build_type_probes(code, entry_point)), codes[0])
    probes = build_type_probes(base, entry_point)
    expressions = [expression for expression, _ in examples] + probes
    executions = [execute_expressions(code, expressions) for code in codes]

    signatures = []
    visible_passes = []
    successes = []
    for execution in executions:
        outputs = execution.get("outputs", [])
        visible = 0
        for index, (_, expected) in enumerate(examples):
            if index < len(outputs) and outputs[index].get("ok"):
                visible += outputs[index].get("value") == _normalize_expected(expected)
        probe_outputs = outputs[len(examples) :]
        signature = tuple(
            (item.get("ok", False), item.get("value")) for item in probe_outputs
        )
        signatures.append(signature if execution.get("compile") else ((False, "compile"),))
        visible_passes.append(int(visible))
        successes.append(sum(bool(item.get("ok")) for item in probe_outputs))

    cluster_sizes = Counter(signatures)
    preferred = len(candidates) - 1  # repaired draft wins only exact signal ties
    selected = max(
        range(len(candidates)),
        key=lambda index: (
            visible_passes[index],
            int(bool(executions[index].get("compile"))),
            cluster_sizes[signatures[index]],
            successes[index],
            int(index == preferred),
            -index,
        ),
    )
    diagnostics = {
        "selector": "prompt_safe_differential_execution_v1",
        "candidate_count": len(candidates),
        "selected_index": selected,
        "entry_point": entry_point,
        "visible_example_count": len(examples),
        "generated_probe_count": len(probes),
        "compile_valid": [bool(item.get("compile")) for item in executions],
        "visible_examples_passed": visible_passes,
        "probe_successes": successes,
        "behavior_cluster_sizes": [cluster_sizes[signature] for signature in signatures],
        "candidate_hashes": [hashlib.sha256(code.encode("utf-8")).hexdigest() for code in codes],
        "uses_hidden_tests": False,
        "uses_reference_solution": False,
    }
    return selected, diagnostics


def evaluate_public_candidate(
    candidate: str,
    prompt: str,
    entry_point: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate one draft using only checks already visible in the prompt."""
    examples = prompt_examples(prompt, entry_point)
    assertions = prompt_assertions(prompt, entry_point)
    expressions = [expression for expression, _ in examples] + assertions
    code = extract_python_code(candidate)
    execution = execute_expressions(code, expressions)
    outputs = execution.get("outputs", [])
    passed = 0
    for index, (_, expected) in enumerate(examples):
        if index < len(outputs) and outputs[index].get("ok"):
            passed += outputs[index].get("value") == _normalize_expected(expected)
    for index in range(len(examples), len(expressions)):
        if index < len(outputs) and outputs[index].get("ok"):
            passed += outputs[index].get("value") == "True"
    return {
        "visible_example_count": len(examples),
        "visible_assertion_count": len(assertions),
        "visible_check_count": len(expressions),
        "visible_checks_passed": int(passed),
        "compile_valid": bool(execution.get("compile")),
        "candidate_hash": hashlib.sha256(code.encode("utf-8")).hexdigest(),
    }


def decide_prompt_visible_repair_retention(
    repair_evidence: Dict[str, Any],
    parent_evidence: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Reject a local repair only when prompt-visible evidence regresses."""
    if int(repair_evidence["visible_check_count"]) != int(
        parent_evidence["visible_check_count"]
    ):
        raise ValueError("repair and parent public-check counts must match")
    repair_rank = (
        int(repair_evidence["visible_checks_passed"]),
        int(bool(repair_evidence["compile_valid"])),
    )
    parent_rank = (
        int(parent_evidence["visible_checks_passed"]),
        int(bool(parent_evidence["compile_valid"])),
    )
    retained_repair = repair_rank >= parent_rank
    return ("repair" if retained_repair else "parent"), {
        "selector": "prompt_visible_nonregression_v1",
        "selected_name": "repair" if retained_repair else "parent",
        "retained_repair": retained_repair,
        "repair_evidence": repair_evidence,
        "parent_evidence": parent_evidence,
        "uses_generated_probes": False,
        "uses_hidden_tests": False,
        "uses_reference_solution": False,
    }


def decide_public_example_guard(
    baseline_evidence: Optional[Dict[str, Any]],
    parent_evidence: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Apply the strict guard, allowing an exact baseline-pruning shortcut."""
    visible_check_count = int(parent_evidence["visible_check_count"])
    parent_passed = int(parent_evidence["visible_checks_passed"])
    baseline_generated = baseline_evidence is not None
    if baseline_evidence is None:
        if not visible_check_count or parent_passed != visible_check_count:
            raise ValueError(
                "baseline may be pruned only when the parent passes every public check"
            )
        selected_name = "parent"
        baseline_passed = None
        baseline_compile = None
        status = "baseline_pruned_parent_exhausted_public_checks"
    else:
        if int(baseline_evidence["visible_check_count"]) != visible_check_count:
            raise ValueError("baseline and parent public-check counts must match")
        baseline_passed = int(baseline_evidence["visible_checks_passed"])
        baseline_compile = bool(baseline_evidence["compile_valid"])
        selected_name = (
            "baseline"
            if visible_check_count and baseline_passed > parent_passed
            else "parent"
        )
        status = "baseline_evaluated"
    return selected_name, {
        "selector": (
            "strict_public_example_guard_v3_lazy_exact"
            if not baseline_generated
            else "strict_public_example_guard_v2"
        ),
        "selected_name": selected_name,
        "status": status,
        "baseline_generated": baseline_generated,
        "exact_eager_selection": True,
        "visible_example_count": int(parent_evidence["visible_example_count"]),
        "visible_assertion_count": int(parent_evidence["visible_assertion_count"]),
        "visible_check_count": visible_check_count,
        "visible_examples_passed": {
            "baseline": baseline_passed,
            "parent": parent_passed,
        },
        "compile_valid": {
            "baseline": baseline_compile,
            "parent": bool(parent_evidence["compile_valid"]),
        },
        "candidate_hashes": {
            "baseline": (
                baseline_evidence["candidate_hash"]
                if baseline_evidence is not None
                else None
            ),
            "parent": parent_evidence["candidate_hash"],
        },
        "uses_generated_probes": False,
        "uses_hidden_tests": False,
        "uses_reference_solution": False,
    }


def select_public_example_guard(
    baseline_candidate: str,
    parent_candidate: str,
    prompt: str,
    entry_point: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Let baseline replace the parent only on strict public-example evidence.

    This is intentionally narrower than differential candidate selection: it
    executes only examples already visible in the model prompt, creates no
    synthetic probes, and keeps the parent on every tie or missing-example
    case.  Reference solutions and hidden tests are not accepted as inputs.
    """
    baseline_evidence = evaluate_public_candidate(
        baseline_candidate, prompt, entry_point
    )
    parent_evidence = evaluate_public_candidate(parent_candidate, prompt, entry_point)
    return decide_public_example_guard(baseline_evidence, parent_evidence)
