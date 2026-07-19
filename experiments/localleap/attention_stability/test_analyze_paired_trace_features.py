import json
import tempfile
import unittest
from pathlib import Path

from analyze_paired_trace_features import build_report


class PairedTraceFeatureAuditTest(unittest.TestCase):
    def test_joins_by_identity_and_summarizes_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair_path = root / "pairs.jsonl"
            trace_path = root / "trace.jsonl"
            pair_path.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in [
                        {
                            "stable_task_id": "a",
                            "prompt_hash": "ha",
                            "baseline_correct": False,
                            "method_correct": True,
                            "method_nfe": 140,
                        },
                        {
                            "stable_task_id": "b",
                            "prompt_hash": "hb",
                            "baseline_correct": True,
                            "method_correct": False,
                            "method_nfe": 150,
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            trace_path.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in [
                        {
                            "task_id": "b",
                            "prompt_hash": "hb",
                            "decode_diagnostics": {
                                "selected_name": "fast",
                                "accuracy_early_abort": {"triggered": True},
                                "public_example_guard": {
                                    "status": "not_available",
                                    "baseline_generated": False,
                                },
                            },
                        },
                        {
                            "task_id": "a",
                            "prompt_hash": "ha",
                            "decode_diagnostics": {
                                "selected_name": "accuracy",
                                "accuracy_early_abort": {"triggered": False},
                                "public_example_guard": {
                                    "status": "not_available",
                                    "baseline_generated": False,
                                },
                            },
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            report = build_report(pair_path, trace_path)
            self.assertEqual(report["duplicate_or_missing_ids"], 0)
            self.assertEqual(report["groups"]["method_only"]["count"], 1)
            self.assertEqual(report["groups"]["baseline_only"]["count"], 1)
            self.assertEqual(
                report["groups"]["baseline_only"]["selected_name_counts"],
                {"fast": 1},
            )
            self.assertEqual(
                report["outcome_cross_tabs"]["selected_name"]["fast"][
                    "paired_balance"
                ],
                -1,
            )

    def test_rejects_prompt_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = root / "pairs.jsonl"
            traces = root / "trace.jsonl"
            pairs.write_text(
                json.dumps(
                    {
                        "stable_task_id": "a",
                        "prompt_hash": "one",
                        "baseline_correct": True,
                        "method_correct": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            traces.write_text(
                json.dumps({"task_id": "a", "prompt_hash": "two"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "prompt hash mismatches"):
                build_report(pairs, traces)


if __name__ == "__main__":
    unittest.main()
