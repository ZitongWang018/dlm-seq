#!/usr/bin/env python3
import tempfile
from pathlib import Path

from audit_model_input_hashes import build_records, compare_records


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        assert add_generation_prompt and not tokenize
        return "USER:" + messages[0]["content"] + "\nASSISTANT:"

    def __call__(self, text):
        return {"input_ids": [ord(char) for char in text]}


def sample(task_id, prompt, target="gold"):
    return {
        "doc_id": 0,
        "doc": {"task_id": task_id, "prompt": prompt},
        "target": target,
        "arguments": {"gen_args_0": {"arg_0": prompt}},
    }


def main():
    tokenizer = FakeTokenizer()
    left = build_records([sample("a", "solve me")], tokenizer)
    right = build_records([sample("a", "solve me")], tokenizer)
    assert compare_records(left, right)["all_equal"]
    changed = build_records([sample("a", "different")], tokenizer)
    result = compare_records(left, changed)
    assert not result["all_equal"]
    assert "model_input_token_ids_hash" in result["mismatches"][0]["fields"]
    try:
        build_records([sample("a", "x"), sample("a", "x")], tokenizer)
    except ValueError as exc:
        assert "duplicate stable id" in str(exc)
    else:
        raise AssertionError("duplicate id was accepted")
    print("model input lineage tests passed")


if __name__ == "__main__":
    main()
