import json
import tempfile
import unittest
from pathlib import Path

from audit_run_config_fairness import compare, parse_model_args
from audit_runtime_model_inputs import index as index_runtime_inputs
from audit_runtime_model_inputs import set_hash
from finalize_unified_offline_protocol import finalize
from freeze_offline_eval_protocol import selected_files


class StrictUnifiedOfflineProtocolTests(unittest.TestCase):
    def test_registered_candidate_selection_order_is_implemented(self):
        controller = (
            Path(__file__).parent
            / "scripts"
            / "run_strict_unified_offline_three_arm_queue.sh"
        ).read_text()
        v19_check = 'if [[ -e "${v19_queue}/ACCEPTED" ]]'
        v18_check = 'elif [[ -e "${v18_queue}/ACCEPTED" ]]'
        v20_check = 'elif [[ -e "${v20_queue}/ACCEPTED" ]]'
        self.assertIn(v19_check, controller)
        self.assertIn(v18_check, controller)
        self.assertIn(v20_check, controller)
        self.assertIn("HF_EVALUATE_OFFLINE=1", controller)
        repair_wait = 'waiting_for_repair_chain_terminal=${v19_queue}'
        self.assertIn(repair_wait, controller)
        self.assertLess(controller.index(repair_wait), controller.index(v19_check))
        self.assertLess(controller.index(v19_check), controller.index(v18_check))
        self.assertLess(controller.index(v18_check), controller.index(v20_check))
        self.assertNotIn("no_new_repair_passed_retain_v15_v11_family", controller)

    def test_runtime_capture_hashes_exact_token_ids_and_chat_text(self):
        from eval_llada import build_runtime_input_record

        first = build_runtime_input_record(
            0, {"task_id": "t0", "answer": "hidden"}, "question", "<chat>question", [1, 2, 3]
        )
        token_changed = build_runtime_input_record(
            0, {"task_id": "t0", "answer": "hidden"}, "question", "<chat>question", [1, 2, 4]
        )
        chat_changed = build_runtime_input_record(
            0, {"task_id": "t0", "answer": "hidden"}, "question", "question", [1, 2, 3]
        )
        self.assertNotEqual(
            first["model_input_token_ids_hash"], token_changed["model_input_token_ids_hash"]
        )
        self.assertNotEqual(
            first["model_input_text_hash"], chat_changed["model_input_text_hash"]
        )
        self.assertEqual(first["stable_task_id"], "t0")

    def test_config_comparison_allows_only_dependency_arguments(self):
        core = {
            "task": "gsm8k",
            "steps": "128",
            "gen_length": "256",
            "baseline_budget_per_step": "2",
            "expected_records": "1319",
            "num_fewshot": "0",
            "seeds": "0,1234,1234,1234",
            "profile": "baseline",
            "model_args": (
                "model_path=/model,gen_length=256,steps=128,block_length=32,"
                "remasking=low_confidence,early_stop=False,show_speed=True,"
                "integrate_speed=False,runtime_input_trace_dir=/baseline"
            ),
        }
        candidate = dict(core)
        candidate["profile"] = "trajectory_early_sparse_context_repair"
        candidate["model_args"] = (
            core["model_args"].replace("/baseline", "/candidate")
            + ",dependency_threshold=0.004,dependency_mode=symmetric"
        )
        result = compare(core, candidate)
        self.assertTrue(result["all_equal_core"])
        candidate["steps"] = "256"
        self.assertFalse(compare(core, candidate)["all_equal_core"])

    def test_runtime_input_index_rejects_duplicates(self):
        row = {
            "schema": "localleap_runtime_model_input_v1",
            "stable_task_id": "x",
            "absolute_index": 0,
            "raw_prompt_hash": "a",
            "model_input_text_hash": "b",
            "model_input_token_ids_hash": "c",
            "model_input_token_count": 3,
            "implicit_attention_mask_hash": "d",
            "document_hash": "e",
            "tokenizer_call": {},
        }
        indexed = index_runtime_inputs([row])
        self.assertEqual(len(set_hash(indexed)), 64)
        with self.assertRaises(ValueError):
            index_runtime_inputs([row, row])

    def test_offline_manifest_excludes_model_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            data = root / "data"
            model.mkdir()
            data.mkdir()
            (model / "config.json").write_text("{}")
            (model / "model-1.safetensors").write_bytes(b"weights")
            (data / "test.arrow").write_bytes(b"arrow")
            paths = selected_files([data, model], model)
            names = {path.name for path in paths}
            self.assertIn("config.json", names)
            self.assertIn("test.arrow", names)
            self.assertNotIn("model-1.safetensors", names)

    def test_finalizer_requires_one_profile_and_runtime_input_equality(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            offline = root / "offline.json"
            offline.write_text(json.dumps({"pass": True}))
            tasks = {}
            for task, total in {
                "humaneval": 164,
                "math500": 500,
                "gsm8k": 1319,
                "mbpp": 500,
            }.items():
                pair = {
                    "total": total,
                    "prompt_hash_mismatches": 0,
                    "target_hash_mismatches": 0,
                    "duplicate_or_missing_ids": 0,
                    "source_hash_mismatches": 0,
                    "evaluator_version_mismatches": 0,
                    "baseline_correct": 1,
                    "method_correct": 2,
                    "baseline_total_nfe": total * 128,
                    "method_total_nfe": total * 127,
                    "baseline_wall_seconds": 10.0,
                    "method_wall_seconds": 9.0,
                }
                paths = {}
                for name, payload in {
                    "candidate_pair": pair,
                    "fast_pair": pair,
                    "candidate_input_compare": {"all_equal": True},
                    "fast_input_compare": {"all_equal": True},
                    "candidate_config_compare": {
                        "all_equal_core": True,
                        "candidate_profile": "one_profile",
                    },
                    "fast_config_compare": {
                        "all_equal_core": True,
                        "candidate_profile": "symmetric_fast",
                    },
                    "candidate_leakage": {
                        "pass": True,
                        "evaluator_version": "generation_information_leakage_audit_v2",
                    },
                }.items():
                    path = root / f"{task}_{name}.json"
                    path.write_text(json.dumps(payload))
                    paths[name] = str(path)
                tasks[task] = paths
            result = finalize(
                {
                    "candidate_profile": "one_profile",
                    "candidate_family": "one_family",
                    "selection_reason": "test",
                    "offline_manifest_verification": str(offline),
                    "tasks": tasks,
                }
            )
            self.assertTrue(result["single_algorithm"])
            self.assertFalse(result["task_specific_routing"])


if __name__ == "__main__":
    unittest.main()
