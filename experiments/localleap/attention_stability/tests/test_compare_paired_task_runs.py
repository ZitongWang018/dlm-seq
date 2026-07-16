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


if __name__ == "__main__":
    unittest.main()

