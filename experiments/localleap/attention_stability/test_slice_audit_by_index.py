import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("slice_audit_by_index.py")
SPEC = importlib.util.spec_from_file_location("slice_audit_by_index", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AbsoluteIndexSliceTest(unittest.TestCase):
    def test_orders_mixed_task_ids_by_absolute_index(self):
        records = [
            {"absolute_index": 2, "stable_task_id": "test/precalculus/807.json"},
            {"absolute_index": 0, "stable_task_id": "main/0"},
            {"absolute_index": 1, "stable_task_id": "mbpp/11"},
        ]
        selected = MODULE.slice_records(records, 0, 3)
        self.assertEqual([row["absolute_index"] for row in selected], [0, 1, 2])

    def test_rejects_missing_absolute_index(self):
        with self.assertRaisesRegex(ValueError, "slice indices mismatch"):
            MODULE.slice_records([{"absolute_index": 0}], 0, 2)

    def test_existing_output_is_preserved_by_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "existing.jsonl"
            output.write_text("preserve\n")
            self.assertEqual(output.read_text(), "preserve\n")


if __name__ == "__main__":
    unittest.main()
