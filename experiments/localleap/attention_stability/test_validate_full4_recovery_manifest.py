import csv
import json
import tempfile
import unittest
from pathlib import Path

from validate_full4_recovery_manifest import REQUIRED_GENERATION_STAGES, validate


class RecoveryManifestTest(unittest.TestCase):
    def make_queue(self, extra_rows=(), report_pass=False):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        (root / "leakage").mkdir()
        (root / "FAILED").touch()
        rows = [(stage, "DONE") for stage in sorted(REQUIRED_GENERATION_STAGES)]
        rows.extend(extra_rows)
        with (root / "formal_manifest.tsv").open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["stage", "status", "start", "finish", "note"])
            for stage, status in rows:
                writer.writerow([stage, status, "x", "", ""])
        (root / "leakage" / "humaneval.json").write_text(
            json.dumps({"pass": report_pass})
        )
        return temp, root

    def test_accepts_interrupted_old_auditor_with_failed_report(self):
        temp, root = self.make_queue([("leakage_humaneval", "STARTED")])
        self.addCleanup(temp.cleanup)
        result = validate(root / "formal_manifest.tsv", root)
        self.assertEqual(result["generation_stages_complete"], 7)
        self.assertIn(
            "interrupted_report:leakage_humaneval:humaneval.json",
            result["recoverable_evaluator_failure_evidence"],
        )

    def test_accepts_explicit_leakage_failure(self):
        temp, root = self.make_queue([("leakage_humaneval", "FAILED")])
        self.addCleanup(temp.cleanup)
        result = validate(root / "formal_manifest.tsv", root)
        self.assertIn(
            "manifest:leakage_humaneval",
            result["recoverable_evaluator_failure_evidence"],
        )

    def test_rejects_generation_failure(self):
        temp, root = self.make_queue([("he_method_full164", "FAILED")])
        self.addCleanup(temp.cleanup)
        with self.assertRaisesRegex(ValueError, "generation_failed"):
            validate(root / "formal_manifest.tsv", root)

    def test_rejects_unexplained_controller_failure(self):
        temp, root = self.make_queue(
            [("leakage_humaneval", "STARTED")], report_pass=True
        )
        self.addCleanup(temp.cleanup)
        with self.assertRaisesRegex(ValueError, "no_evaluator_failure_evidence"):
            validate(root / "formal_manifest.tsv", root)


if __name__ == "__main__":
    unittest.main()
