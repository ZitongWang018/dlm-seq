import torch

from eval_llada import decode_candidate_generations


class DummyTokenizer:
    def decode(self, values, skip_special_tokens=True):
        assert skip_special_tokens is True
        return "".join(chr(int(value)) for value in values)


def candidate_ids(text, prompt="P"):
    return torch.tensor([[ord(char) for char in prompt + text]])


def test_humaneval_candidate_trace_preserves_full_selectable_response():
    text = "def solve():\n    return 1\n\n\nextra"
    decoded = decode_candidate_generations(
        DummyTokenizer(),
        {"fast": candidate_ids(text)},
        prompt_length=1,
        stop_tokens=["\n\n\n"],
        preserve_full_humaneval_response=True,
    )
    assert decoded["fast"] == text


def test_non_humaneval_candidate_trace_keeps_generic_stop_policy():
    decoded = decode_candidate_generations(
        DummyTokenizer(),
        {"fast": candidate_ids("answer<STOP>ignored")},
        prompt_length=1,
        stop_tokens=["<STOP>"],
        preserve_full_humaneval_response=False,
    )
    assert decoded["fast"] == "answer"


if __name__ == "__main__":
    test_humaneval_candidate_trace_preserves_full_selectable_response()
    test_non_humaneval_candidate_trace_keeps_generic_stop_policy()
    print("2 candidate-generation trace tests passed")
