import math

import torch

import generate as generate_module
from generate import _jsd_from_probabilities, select_candidate_memory_tokens


MASK = 99


def _selector(
    logits,
    x,
    previous_memory=None,
    previous_selected=None,
    fallback="confidence",
    delta=0.0,
    exact_jsd=False,
):
    mask_index = x == MASK
    directional = torch.zeros((1, x.shape[1], x.shape[1]), dtype=torch.float32)
    if previous_selected is not None and previous_selected.numel():
        # Deliberately asymmetric: remaining position 1 reads selected 0 weakly,
        # while position 2 reads it strongly.  Reverse directions are distinct.
        directional[0, 1, previous_selected[0]] = 0.01
        directional[0, 2, previous_selected[0]] = 0.90
        directional[0, previous_selected[0], 1] = 0.20
        directional[0, previous_selected[0], 2] = 0.30
    return select_candidate_memory_tokens(
        logits=logits,
        temperature=0.0,
        remasking="low_confidence",
        mask_index=mask_index,
        x=x,
        budget=1,
        directional_attention=directional,
        block_start=0,
        candidate_topk=2,
        confidence_threshold=delta,
        fallback_mode=fallback,
        collect_exact_jsd=exact_jsd,
        previous_memory=previous_memory,
        previous_selected=previous_selected,
    )


def test_jsd_boundaries_and_symmetry():
    first = torch.tensor([[1.0, 0.0], [0.2, 0.8]])
    same = _jsd_from_probabilities(first, first)
    assert torch.allclose(same, torch.zeros_like(same), atol=1e-7)
    left = torch.tensor([[1.0, 0.0]])
    right = torch.tensor([[0.0, 1.0]])
    divergence = _jsd_from_probabilities(left, right)
    assert abs(float(divergence.item()) - math.log(2)) < 1e-6
    assert torch.allclose(divergence, _jsd_from_probabilities(right, left), atol=1e-7)


def test_bootstrap_matches_baseline_and_memory_deletes_selected():
    x = torch.full((1, 3), MASK, dtype=torch.long)
    logits = torch.tensor(
        [[
            [4.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0],
        ]]
    )
    x0, transfer, diagnostics, memory = _selector(logits, x)
    selected = torch.where(transfer[0])[0]
    assert selected.tolist() == [0]
    assert x0[0, 0].item() == 0
    assert set(memory[0]) == {1, 2}
    state = diagnostics["candidate_state"][0]
    assert not state["has_history"].any()
    assert state["selection_reason"][0] == "bootstrap"


def test_bootstrap_matches_full_sequence_topk_with_ties():
    x = torch.tensor([[7, 8, 9, MASK, MASK, MASK]], dtype=torch.long)
    logits = torch.zeros((1, 6, 5), dtype=torch.float32)
    logits[0, 3:, 0] = 100.0
    mask_index = x == MASK
    directional = torch.zeros((1, 3, 3), dtype=torch.float32)
    _, transfer, _, _ = select_candidate_memory_tokens(
        logits=logits,
        temperature=0.0,
        remasking="low_confidence",
        mask_index=mask_index,
        x=x,
        budget=1,
        directional_attention=directional,
        block_start=3,
        candidate_topk=2,
        confidence_threshold=0.0,
        fallback_mode="confidence",
        collect_exact_jsd=False,
    )

    baseline_top1 = torch.argmax(logits, dim=-1)
    baseline_probabilities = torch.softmax(logits.to(torch.float64), dim=-1)
    baseline_confidence = baseline_probabilities.gather(
        -1, baseline_top1.unsqueeze(-1)
    ).squeeze(-1)
    baseline_confidence = torch.where(
        mask_index, baseline_confidence, torch.tensor(-torch.inf, dtype=torch.float64)
    )
    expected = torch.topk(baseline_confidence[0], k=1).indices
    assert torch.equal(torch.where(transfer[0])[0], expected)


def test_stable_candidate_precedes_more_confident_changed_candidate():
    x = torch.full((1, 3), MASK, dtype=torch.long)
    first_logits = torch.tensor(
        [[
            [4.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0],
        ]]
    )
    first_x0, first_transfer, _, memory = _selector(first_logits, x)
    selected = torch.where(first_transfer[0])[0]
    x[first_transfer] = first_x0[first_transfer]

    second_logits = torch.tensor(
        [[
            [4.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.2, 0.0, 0.0, 0.0],  # stable token 1, lower confidence
            [0.0, 0.0, 0.0, 5.0, 0.0],  # changed token 3, very confident
        ]]
    )
    _, transfer, diagnostics, next_memory = _selector(
        second_logits, x, memory, selected, fallback="confidence"
    )
    assert torch.where(transfer[0])[0].tolist() == [1]
    assert set(next_memory[0]) == {2}
    state = diagnostics["candidate_state"][0]
    assert state["top1_stable"].tolist() == [True, False]
    assert torch.allclose(state["attention_arrival"], torch.tensor([0.01, 0.90]))
    assert torch.allclose(
        state["directional_attention_asymmetry_to_previous"],
        torch.tensor([[0.19], [0.60]]),
    )
    assert not state["full_jsd_available"].any()
    assert torch.equal(
        state["decision_jsd_nats"], state["sparse_previous_partition_jsd_nats"]
    )
    assert diagnostics["runtime_full_probability_elements"] == 0


def test_optional_exact_jsd_does_not_change_decision_jsd():
    x = torch.full((1, 3), MASK, dtype=torch.long)
    first_logits = torch.tensor(
        [[[4.0, 0.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0, 0.0]]]
    )
    first_x0, first_transfer, _, memory = _selector(
        first_logits, x, exact_jsd=True
    )
    selected = torch.where(first_transfer[0])[0]
    x[first_transfer] = first_x0[first_transfer]
    second_logits = first_logits.clone()
    second_logits[0, 2] = torch.tensor([0.0, 0.0, 0.0, 3.0, 0.0])
    _, _, diagnostics, _ = _selector(
        second_logits, x, memory, selected, exact_jsd=True
    )
    state = diagnostics["candidate_state"][0]
    available = state["full_jsd_available"]
    assert available.all()
    assert torch.all(
        state["sparse_previous_partition_jsd_nats"][available]
        <= state["full_jsd_nats"][available] + 1e-6
    )
    assert torch.equal(
        state["decision_jsd_nats"], state["sparse_previous_partition_jsd_nats"]
    )


def test_argmax_is_preserved_when_topk_tie_break_differs():
    x = torch.full((1, 3), MASK, dtype=torch.long)
    logits = torch.zeros((1, 3, 8), dtype=torch.float32)
    logits[:, :, 2] = 5.0
    logits[:, :, 6] = 5.0
    actual_argmax = torch.argmax(logits, dim=-1)
    _, transfer, diagnostics, memory = _selector(logits, x)
    state = diagnostics["candidate_state"][0]
    assert torch.equal(state["current_top1_token_ids"], actual_argmax[0].to(torch.int32))
    assert torch.equal(
        state["current_topk_token_ids"][:, 0], state["current_top1_token_ids"]
    )
    for position, item in memory[0].items():
        assert int(item["top1_id"]) == int(actual_argmax[0, position])
        assert int(item["topk_ids"][0]) == int(actual_argmax[0, position])
    assert transfer.sum().item() == 1


def test_no_eligible_fallback_modes_and_frontier():
    x = torch.full((1, 3), MASK, dtype=torch.long)
    first_logits = torch.tensor(
        [[
            [4.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0],
        ]]
    )
    first_x0, first_transfer, _, memory = _selector(first_logits, x)
    selected = torch.where(first_transfer[0])[0]
    x[first_transfer] = first_x0[first_transfer]
    changed_logits = torch.tensor(
        [[
            [4.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.5, 0.0],
            [0.0, 0.0, 0.0, 5.0, 0.0],
        ]]
    )
    _, confidence_transfer, confidence_diag, _ = _selector(
        changed_logits, x.clone(), memory, selected, fallback="confidence"
    )
    assert torch.where(confidence_transfer[0])[0].tolist() == [2]
    assert confidence_diag["fallback_reason"] == "no_eligible"
    assert confidence_diag["forced_commits"] == 1

    _, impact_transfer, _, _ = _selector(
        changed_logits, x.clone(), memory, selected, fallback="impact"
    )
    assert torch.where(impact_transfer[0])[0].tolist() == [1]

    # The frontier rule only allows the two highest-confidence candidates to
    # replace each other when b=1.  With two remaining positions it agrees with
    # ordinary stability-first and remains fixed-budget.
    stable_logits = changed_logits.clone()
    stable_logits[0, 1] = torch.tensor([0.0, 1.1, 0.0, 0.0, 0.0])
    _, frontier_transfer, frontier_diag, _ = _selector(
        stable_logits, x.clone(), memory, selected, fallback="frontier"
    )
    assert torch.where(frontier_transfer[0])[0].tolist() == [1]
    state = frontier_diag["candidate_state"][0]
    assert state["in_confidence_frontier"].all()


def test_frontier_excludes_deep_stable_candidate():
    x = torch.full((1, 4), MASK, dtype=torch.long)
    first_logits = torch.tensor(
        [[
            [6.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 3.0, 0.0],
        ]]
    )
    directional = torch.zeros((1, 4, 4), dtype=torch.float32)
    first_x0, first_transfer, _, memory = select_candidate_memory_tokens(
        logits=first_logits,
        temperature=0.0,
        remasking="low_confidence",
        mask_index=x == MASK,
        x=x,
        budget=1,
        directional_attention=directional,
        block_start=0,
        candidate_topk=2,
        confidence_threshold=0.0,
        fallback_mode="confidence",
    )
    selected = torch.where(first_transfer[0])[0]
    x[first_transfer] = first_x0[first_transfer]
    second_logits = torch.tensor(
        [[
            [6.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.0, 0.0, 0.0],  # deep but stable
            [0.0, 0.0, 0.0, 0.0, 2.5],  # frontier, changed
            [0.0, 0.0, 0.0, 0.0, 4.0],  # baseline top-1, changed
        ]]
    )
    _, ordinary_transfer, _, _ = select_candidate_memory_tokens(
        logits=second_logits,
        temperature=0.0,
        remasking="low_confidence",
        mask_index=x == MASK,
        x=x,
        budget=1,
        directional_attention=directional,
        block_start=0,
        candidate_topk=2,
        confidence_threshold=0.0,
        fallback_mode="confidence",
        previous_memory=memory,
        previous_selected=selected,
    )
    _, frontier_transfer, diagnostics, _ = select_candidate_memory_tokens(
        logits=second_logits,
        temperature=0.0,
        remasking="low_confidence",
        mask_index=x == MASK,
        x=x,
        budget=1,
        directional_attention=directional,
        block_start=0,
        candidate_topk=2,
        confidence_threshold=0.0,
        fallback_mode="frontier",
        previous_memory=memory,
        previous_selected=selected,
    )
    assert torch.where(ordinary_transfer[0])[0].tolist() == [1]
    assert torch.where(frontier_transfer[0])[0].tolist() == [3]
    state = diagnostics["candidate_state"][0]
    deep_index = state["masked_positions_global"].tolist().index(1)
    assert state["top1_stable"][deep_index]
    assert not state["in_confidence_frontier"][deep_index]


def test_confidence_threshold_forces_fixed_budget_fallback():
    x = torch.full((1, 3), MASK, dtype=torch.long)
    first_logits = torch.tensor(
        [[
            [4.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]]
    )
    first_x0, first_transfer, _, memory = _selector(first_logits, x)
    selected = torch.where(first_transfer[0])[0]
    x[first_transfer] = first_x0[first_transfer]
    _, transfer, diagnostics, _ = _selector(
        first_logits, x, memory, selected, fallback="confidence", delta=0.99
    )
    assert transfer.sum().item() == 1
    assert diagnostics["forced_commits"] == 1
    assert diagnostics["fallback_used"]


def test_generate_candidate_memory_fixed_budget_and_sparse_state():
    class FakeModel:
        device = torch.device("cpu")

    def fake_forward(_model, x, block_start, block_end):
        logits = torch.zeros((1, x.shape[1], 5), dtype=torch.float32)
        logits[0, block_start, 1] = 4.0
        logits[0, block_start + 1, 2] = 3.0
        directional = torch.zeros((1, block_end - block_start, block_end - block_start))
        directional[0, 1, 0] = 0.25
        symmetric = 0.5 * (directional + directional.transpose(-2, -1))
        return logits, directional, symmetric

    original = generate_module._forward_with_block_attention
    generate_module._forward_with_block_attention = fake_forward
    try:
        prompt = torch.tensor([[0]], dtype=torch.long)
        output, nfe, summary = generate_module.generate_candidate_memory(
            FakeModel(),
            prompt,
            candidate_topk=2,
            confidence_threshold=0.0,
            fallback_mode="confidence",
            steps=2,
            gen_length=2,
            block_length=2,
            mask_id=MASK,
            collect_step_diagnostics=True,
            collect_exact_jsd=False,
        )
    finally:
        generate_module._forward_with_block_attention = original
    assert nfe == 2
    assert not (output == MASK).any()
    assert summary["decoder"] == "candidate_memory_stability_v2"
    assert summary["peak_runtime_full_probability_elements"] == 0
    assert len(summary["_step_records"]) == 2
    assert all(step["budget"] == 1 for step in summary["_step_records"])
