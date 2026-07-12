import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.distribution import _topk_l1, kl_divergence
from src.datasets import extract_number
from src.samplers import _topk_path_efficiency, generate_with_sampler, response_budget_transfer_count


def test_topk_distance_uses_token_indices_not_probability_values():
    p = torch.tensor([10.0, 0.0, 0.0, -5.0])
    q = torch.tensor([0.0, 10.0, 0.0, -5.0])

    assert _topk_l1(p, q, k=1) > 1.9
    assert kl_divergence(p, q, top_k=1) > 5.0


def test_extract_number_ignores_sentence_terminal_period():
    assert extract_number("The answer is 694.") == "694"
    assert extract_number("The answer is 3.14.") == "3.14"


def test_response_budget_reduces_only_intermediate_batches():
    assert response_budget_transfer_count(128, 64, 0.4, 0.35, 0.5) == 1
    assert response_budget_transfer_count(128, 64, 0.2, 0.35, 0.5) == 2
    assert response_budget_transfer_count(7, 1, 1.0, 0.35, 0.5) == 7


def test_path_efficiency_prefers_continuing_distribution_change():
    ids = torch.tensor([1, 2])
    previous = (ids, torch.tensor([0.9, 0.1]))
    current = (ids, torch.tensor([0.6, 0.4]))
    continued = (ids, torch.tensor([0.3, 0.7]))
    reversed_state = (ids, torch.tensor([0.85, 0.15]))
    continued_score, _ = _topk_path_efficiency(previous, current, continued)
    reversed_score, _ = _topk_path_efficiency(previous, current, reversed_state)
    assert continued_score > reversed_score


def test_multiblock_sampler_finishes_each_block():
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, tokens, attention_mask=None):
            logits = torch.zeros((*tokens.shape, 6), device=tokens.device)
            for pos in range(tokens.shape[1]):
                logits[:, pos, 1 + pos % 4] = 5.0
            return type("Output", (), {"logits": logits})()

    output = generate_with_sampler(
        FakeModel(), torch.tensor([[1]]), steps=4, gen_length=8,
        block_length=4, mask_id=5, sampler="lcr",
    )
    assert output["nfe"] == 4
    assert not bool((output["tokens"] == 5).any())

    for sampler in ("lcr_wavefront", "lcr_response_wavefront"):
        wavefront_output = generate_with_sampler(
            FakeModel(), torch.tensor([[1]]), steps=4, gen_length=8,
            block_length=8, mask_id=5, sampler=sampler,
            wavefront_size=4, wavefront_radius=2,
        )
        assert wavefront_output["nfe"] == 4
        assert not bool((wavefront_output["tokens"] == 5).any())
        if sampler == "lcr_response_wavefront":
            assert wavefront_output["diagnostics"]["wavefront_expansion_steps"] > 0


if __name__ == "__main__":
    test_topk_distance_uses_token_indices_not_probability_values()
    test_extract_number_ignores_sentence_terminal_period()
    test_response_budget_reduces_only_intermediate_batches()
    test_path_efficiency_prefers_continuing_distribution_change()
    test_multiblock_sampler_finishes_each_block()
    print("distribution distance test passed")
