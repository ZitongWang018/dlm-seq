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
