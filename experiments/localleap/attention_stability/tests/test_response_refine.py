import torch
from types import SimpleNamespace

import generate


def make_position_risk(prompt_length=1, gen_length=4, block_length=4):
    sequence_length = prompt_length + gen_length
    return {
        "response_invalidations": torch.zeros(sequence_length, dtype=torch.int32),
        "response_validations": torch.zeros(sequence_length, dtype=torch.int32),
        "commit_confidence": torch.full((sequence_length,), 0.9),
        "commit_maturity": torch.ones(sequence_length, dtype=torch.bool),
        "commit_forced": torch.zeros(sequence_length, dtype=torch.bool),
        "commit_revision_margin": torch.zeros(sequence_length),
        "final_directional_attention": torch.zeros(
            (gen_length // block_length, block_length, block_length)
        ),
    }


def test_frontier_prefers_forced_conditioned_risk_without_score_weight():
    risk = make_position_risk()
    risk["commit_forced"][3] = True
    risk["response_invalidations"][2] = 4
    risk["commit_confidence"][3] = 0.99
    remask, summary = generate.build_response_refine_mask(
        position_risk=risk,
        prompt_length=1,
        gen_length=4,
        block_length=4,
        dependency_threshold=0.004,
        repair_steps=1,
    )
    assert torch.where(remask[0])[0].tolist() == [3]
    assert summary["remask_per_block"] == 1
    assert summary["selected_forced_commits"] == 1


def test_risk_gated_frontier_underfills_instead_of_rewriting_stable_tokens():
    risk = make_position_risk()
    risk["response_invalidations"][2] = 1
    remask, summary = generate.build_response_refine_mask(
        position_risk=risk,
        prompt_length=1,
        gen_length=4,
        block_length=4,
        dependency_threshold=0.004,
        repair_steps=2,
        risk_gated=True,
    )
    assert torch.where(remask[0])[0].tolist() == [2]
    assert summary["risk_gated"] is True
    assert summary["remasked_positions"] == 1


def test_causal_risk_requires_invalidation_and_commit_risk():
    risk = make_position_risk()
    risk["response_invalidations"][1] = 2
    risk["commit_forced"][2] = True
    risk["response_invalidations"][3] = 1
    risk["commit_maturity"][3] = False
    remask, summary = generate.build_response_refine_mask(
        position_risk=risk,
        prompt_length=1,
        gen_length=4,
        block_length=4,
        dependency_threshold=0.004,
        repair_steps=4,
        risk_gated=True,
        require_commit_risk=True,
    )
    assert torch.where(remask[0])[0].tolist() == [3]
    assert summary["require_commit_risk"] is True


def test_repair_decodes_live_directed_source_before_dependent():
    original_forward = generate._forward_with_block_attention

    def fake_forward(model, x, block_start, block_end):
        vocab = 8
        logits = torch.zeros((*x.shape, vocab), dtype=torch.float32)
        for position in range(block_start, block_end):
            logits[:, position, (position + 1) % vocab] = 8.0
        directional = torch.zeros((1, 4, 4), dtype=torch.float32)
        # A[target, source]: local position 1 is read by three positions,
        # while local position 0 mostly reads others.
        directional[0, 0, 1] = 0.10
        directional[0, 2, 1] = 0.10
        directional[0, 3, 1] = 0.10
        symmetric = 0.5 * (directional + directional.transpose(-2, -1))
        return logits, directional, symmetric

    generate._forward_with_block_attention = fake_forward
    try:
        draft = torch.tensor([[6, 1, 2, 3, 4]], dtype=torch.long)
        remask = torch.tensor([[False, True, True, False, False]])
        repaired, nfe, summary = generate.repair_response_refine_positions(
            model=object(),
            draft=draft,
            remask=remask,
            prompt_length=1,
            dependency_threshold=0.004,
            repair_steps=2,
            gen_length=4,
            block_length=4,
            mask_id=7,
        )
    finally:
        generate._forward_with_block_attention = original_forward

    assert nfe == 2
    assert summary["source_first"][0]["global_position"] == 2
    assert summary["selection_order_global"][0] == 2
    assert summary["residual_mask_count"] == 0
    assert repaired[0, 1].item() == 2
    assert repaired[0, 2].item() == 3


def test_matched_mode_preserves_total_forward_budget():
    original_generate = generate.generate_attention_stability
    original_repair = generate.repair_response_refine_positions
    prompt = torch.tensor([[9]], dtype=torch.long)
    draft = torch.tensor([[9, 1, 2, 3, 4]], dtype=torch.long)
    risk = make_position_risk()

    def fake_generate(**kwargs):
        assert kwargs["steps"] == 2
        assert kwargs["collect_position_risk"] is True
        return draft.clone(), 2, {"_position_risk_state": risk, "decoder": "fill"}

    def fake_repair(**kwargs):
        assert int(kwargs["remask"].sum().item()) == 2
        return draft.clone(), 2, {"residual_mask_count": 0}

    generate.generate_attention_stability = fake_generate
    generate.repair_response_refine_positions = fake_repair
    try:
        output, nfe, summary = generate.generate_response_refine(
            model=object(),
            prompt=prompt,
            dependency_threshold=0.004,
            steps=4,
            gen_length=4,
            block_length=4,
            budget_mode="matched",
        )
    finally:
        generate.generate_attention_stability = original_generate
        generate.repair_response_refine_positions = original_repair

    assert torch.equal(output, draft)
    assert nfe == 4
    assert summary["fill_nfe"] == 2
    assert summary["repair_nfe"] == 2


def test_shared_mask_retention_accepts_or_rejects_whole_blocks():
    class FakeModel:
        def __call__(self, x):
            logits = torch.zeros((*x.shape, 10), dtype=torch.float32)
            logits[0, 1, 2] = 5.0  # repaired token wins in block 0
            logits[0, 1, 1] = 1.0
            logits[0, 3, 3] = 5.0  # original token wins in block 1
            logits[0, 3, 4] = 1.0
            return SimpleNamespace(logits=logits)

    draft = torch.tensor([[9, 1, 2, 3, 4]], dtype=torch.long)
    repaired = torch.tensor([[9, 2, 2, 4, 4]], dtype=torch.long)
    remask = torch.tensor([[False, True, False, True, False]])
    retained, nfe, summary = generate.retain_response_refine_blocks(
        model=FakeModel(),
        draft=draft,
        repaired=repaired,
        remask=remask,
        prompt_length=1,
        gen_length=4,
        block_length=2,
        mask_id=7,
    )
    assert nfe == 1
    assert retained.tolist() == [[9, 2, 2, 3, 4]]
    assert summary["accepted_blocks"] == 1
    assert summary["rejected_blocks"] == 1


def test_pareto_retention_rejects_block_with_one_regressing_token():
    class FakeModel:
        def __call__(self, x):
            logits = torch.zeros((*x.shape, 10), dtype=torch.float32)
            logits[0, 1, 2] = 8.0
            logits[0, 1, 1] = 1.0
            logits[0, 2, 2] = 3.0
            logits[0, 2, 3] = 2.0
            return SimpleNamespace(logits=logits)

    draft = torch.tensor([[9, 1, 2]], dtype=torch.long)
    repaired = torch.tensor([[9, 2, 3]], dtype=torch.long)
    remask = torch.tensor([[False, True, True]])
    retained, nfe, summary = generate.retain_response_refine_blocks(
        model=FakeModel(),
        draft=draft,
        repaired=repaired,
        remask=remask,
        prompt_length=1,
        gen_length=2,
        block_length=2,
        mask_id=7,
        require_pareto=True,
    )
    assert nfe == 1
    assert torch.equal(retained, draft)
    assert summary["accepted_blocks"] == 0
    assert summary["rejected_blocks"] == 1
    assert summary["blocks"][0]["score_margin"] > 0
    assert summary["blocks"][0]["minimum_token_margin"] < 0


if __name__ == "__main__":
    test_frontier_prefers_forced_conditioned_risk_without_score_weight()
    test_risk_gated_frontier_underfills_instead_of_rewriting_stable_tokens()
    test_causal_risk_requires_invalidation_and_commit_risk()
    test_repair_decodes_live_directed_source_before_dependent()
    test_matched_mode_preserves_total_forward_budget()
    test_shared_mask_retention_accepts_or_rejects_whole_blocks()
    test_pareto_retention_rejects_block_with_one_regressing_token()
    print("7 response-refine tests passed")
