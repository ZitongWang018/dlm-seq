from humaneval_execution import check_correctness
from sanitize import sanitize


REFERENCE = """
def check(candidate):
    assert candidate(-1.5) == 1.25
check(solve)
"""

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
    # Match the formal HumanEval postprocessor by invoking the official
    # sandbox checker from the main thread.  Its metric wrapper starts a
    # multiprocessing.Manager from a ThreadPool and can deadlock after torch
    # initializes runtime threads on Python 3.12.
    outcomes = [
        float(
            check_correctness(
                candidate + "\n" + REFERENCE,
                3.0,
                task_id=index,
                completion_id=0,
            )["passed"]
        )
        for index, candidate in enumerate(candidates)
    ]
    assert outcomes == [1.0, 1.0, 0.0, 0.0, 0.0], outcomes
    assert sum(outcomes) / len(outcomes) == 0.4
    print("synthetic evaluator cases passed")


if __name__ == "__main__":
    main()
