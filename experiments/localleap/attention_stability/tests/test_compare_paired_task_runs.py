import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "compare_paired_task_runs.py"
SPEC = importlib.util.spec_from_file_location("compare_paired_task_runs", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PairedAuditTest(unittest.TestCase):
    def test_exact_mcnemar_balanced(self):
        self.assertEqual(MODULE.exact_mcnemar_p(0, 0), 1.0)
        self.assertEqual(MODULE.exact_mcnemar_p(1, 1), 1.0)

    def test_exact_mcnemar_one_sided(self):
        self.assertAlmostEqual(MODULE.exact_mcnemar_p(0, 5), 0.0625)

    def test_accepts_humaneval_task_id(self):
        rows = MODULE.index_records([{"task_id": "HumanEval/0"}])
        self.assertEqual(list(rows), ["HumanEval/0"])

    def test_accepts_math_stable_task_id(self):
        rows = MODULE.index_records([{"stable_task_id": "test/algebra/1.json"}])
        self.assertEqual(list(rows), ["test/algebra/1.json"])

    def test_rejects_duplicate_or_conflicting_identity(self):
        with self.assertRaisesRegex(ValueError, "duplicate stable identity"):
            MODULE.index_records([{"task_id": "x"}, {"task_id": "x"}])
        with self.assertRaisesRegex(ValueError, "conflicting stable identities"):
            MODULE.record_stable_id({"stable_task_id": "x", "task_id": "y"})


if __name__ == "__main__":
    unittest.main()
