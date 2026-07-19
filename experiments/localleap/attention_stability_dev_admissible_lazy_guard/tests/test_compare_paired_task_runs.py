import importlib.util
from pathlib import Path
import tempfile
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

    def test_source_hashes_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.txt"
            method = Path(directory) / "method.txt"
            common_generate = "a" * 64
            common_eval = "b" * 64
            baseline.write_text(
                f"{common_generate}  generate.py\n{common_eval}  eval_llada.py\n"
            )
            method.write_text(
                f"{common_generate}  generate.py\n{common_eval}  eval_llada.py\n"
            )
            self.assertEqual(
                MODULE.verify_matching_source_hashes(baseline, method),
                ["eval_llada.py", "generate.py"],
            )
            method.write_text(
                f"{'c' * 64}  generate.py\n{common_eval}  eval_llada.py\n"
            )
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                MODULE.verify_matching_source_hashes(baseline, method)

    def test_source_comparison_reports_cross_version_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.txt"
            method = Path(directory) / "method.txt"
            baseline.write_text(
                f"{'a' * 64}  generate.py\n{'b' * 64}  eval_llada.py\n"
            )
            method.write_text(
                f"{'c' * 64}  generate.py\n{'b' * 64}  eval_llada.py\n"
            )
            comparison = MODULE.compare_source_hashes(baseline, method)
            self.assertEqual(comparison["mismatches"], ["generate.py"])
            self.assertEqual(
                comparison["common"], ["eval_llada.py", "generate.py"]
            )


if __name__ == "__main__":
    unittest.main()
