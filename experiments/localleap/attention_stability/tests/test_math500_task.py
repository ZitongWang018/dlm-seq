import importlib.util
from pathlib import Path
import unittest


UTILS = (
    Path(__file__).resolve().parents[1]
    / "tasks"
    / "localleap_math500"
    / "utils.py"
)
SPEC = importlib.util.spec_from_file_location("localleap_math500_utils", UTILS)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Math500TaskTest(unittest.TestCase):
    def test_nested_boxed_answer(self):
        doc = {"answer": r"\frac{1}{2}"}
        result = MODULE.process_results(doc, [r"Thus $\boxed{\frac{1}{2}}$."])
        self.assertEqual(result["exact_match"], 1)

    def test_variable_assignment_before_tuple(self):
        doc = {"answer": r"\left(3, \frac{\pi}{2}\right)"}
        response = r"<answer>\boxed{(r,\theta)=(3,\frac{\pi}{2})}</answer>"
        self.assertEqual(MODULE.process_results(doc, [response])["exact_match"], 1)

    def test_negative_decimal_answer(self):
        doc = {"answer": "-2.5"}
        result = MODULE.process_results(doc, ["The final answer is -2.5."])
        self.assertEqual(result["exact_match"], 1)

    def test_terminal_punctuation_is_removed(self):
        self.assertEqual(MODULE.normalize_math("7."), "7")

    def test_missing_answer_does_not_match(self):
        doc = {"answer": "4"}
        result = MODULE.process_results(doc, ["I cannot determine this."])
        self.assertEqual(result["exact_match"], 0)


if __name__ == "__main__":
    unittest.main()
