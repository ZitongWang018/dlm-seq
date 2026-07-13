import evaluate as hf_evaluate

from sanitize import sanitize


REFERENCE = """
def check(candidate):
    assert candidate(-1.5) == 1.25
check(solve)
"""

METRIC = hf_evaluate.load("code_eval")


def main():
    fenced = """```python
def solve(x):
    return (x + 1.0) * (x - 1.0)
```"""
    sanitized = sanitize(fenced, "solve")
    assert "def solve" in sanitized

    repeated = """
def solve(x):
    return 999

def helper(x):
    return (x + 1.0) * (x - 1.0)

def solve(x):
    return helper(x)
"""
    missing = sanitize("The answer is unavailable.", "solve")
    candidates = [
        sanitized,
        sanitize(repeated, "solve"),
        missing,
        "def solve(x):\n    raise RuntimeError('synthetic')",
        "def solve(x):\n    while True:\n        pass",
    ]
    result = METRIC.compute(
        references=[REFERENCE] * len(candidates),
        predictions=[[candidate] for candidate in candidates],
        k=[1],
    )[0]["pass@1"]
    assert result == 0.4, result
    print("synthetic evaluator cases passed")


if __name__ == "__main__":
    main()
