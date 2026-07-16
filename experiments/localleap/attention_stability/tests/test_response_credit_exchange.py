import torch

import generate


def test_disagreement_repair_preserves_agreement_and_fills_masks():
    original_forward = generate._forward_with_block_attention

    def fake_forward(model, x, block_start, block_end):
        vocab = 6
        logits = torch.zeros((*x.shape, vocab), dtype=torch.float32)
        for position in range(block_start, block_end):
            logits[:, position, position % (vocab - 1)] = 8.0 - position * 0.1
        dependency = torch.zeros(
            (x.shape[0], block_end - block_start, block_end - block_start),
            dtype=torch.float32,
        )
        return logits, dependency, dependency

    generate._forward_with_block_attention = fake_forward
    try:
        draft = torch.tensor([[1, 2, 3, 2, 1]], dtype=torch.long)
        disagreement = torch.tensor([[False, False, True, False, True]])
        repaired, nfe, summary = generate.repair_draft_disagreements(
            model=object(),
            draft=draft,
            disagreement_mask=disagreement,
            prompt_length=1,
            dependency_threshold=0.004,
            steps=4,
            gen_length=4,
            block_length=4,
            mask_id=5,
        )
    finally:
        generate._forward_with_block_attention = original_forward

    assert repaired[0, 1].item() == 2
    assert repaired[0, 3].item() == 2
    assert repaired[0, 2].item() == 2
    assert repaired[0, 4].item() == 4
    assert nfe == 2
    assert summary["disagreement_positions"] == 2
    assert summary["residual_mask_count"] == 0


def test_exchange_masks_only_cross_draft_disagreements():
    original_generate = generate.generate_attention_stability
    original_repair = generate.repair_draft_disagreements
    prompt = torch.tensor([[9]], dtype=torch.long)
    anchor = torch.tensor([[9, 1, 2, 3, 4]], dtype=torch.long)
    explorer = torch.tensor([[9, 1, 5, 3, 6]], dtype=torch.long)
    repaired = torch.tensor([[9, 1, 7, 3, 8]], dtype=torch.long)

    def fake_generate_attention_stability(*, temporal_mode, **kwargs):
        if temporal_mode == "top1":
            return anchor.clone(), 4, {"temporal_mode": temporal_mode}
        return explorer.clone(), 5, {"temporal_mode": temporal_mode}

    def fake_repair(**kwargs):
        assert kwargs["disagreement_mask"].tolist() == [
            [False, False, True, False, True]
        ]
        return repaired.clone(), 2, {"residual_mask_count": 0}

    generate.generate_attention_stability = fake_generate_attention_stability
    generate.repair_draft_disagreements = fake_repair
    try:
        output, nfe, summary = generate.generate_response_credit_exchange(
            model=object(),
            prompt=prompt,
            dependency_threshold=0.004,
            steps=4,
            gen_length=4,
            block_length=4,
        )
    finally:
        generate.generate_attention_stability = original_generate
        generate.repair_draft_disagreements = original_repair

    assert torch.equal(output, repaired)
    assert nfe == 11
    assert summary["draft_disagreement_positions"] == 2
    candidates = summary["_draft_candidate_token_ids"]
    assert set(candidates) == {"anchor", "explorer", "repaired"}


if __name__ == "__main__":
    test_disagreement_repair_preserves_agreement_and_fills_masks()
    test_exchange_masks_only_cross_draft_disagreements()
    print("2 response-credit exchange tests passed")
