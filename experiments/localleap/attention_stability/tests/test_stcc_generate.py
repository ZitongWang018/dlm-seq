import types

import torch

from stcc_generate import generate_stcc, select_stcc_tokens


MASK = 99


def _memory(top1_id, top1_probability, second_id, second_probability, streak=0):
    return {
        "top1_id": torch.tensor(top1_id, dtype=torch.long),
        "top1_confidence": torch.tensor(top1_probability, dtype=torch.float32),
        "topk_ids": torch.tensor([top1_id, second_id], dtype=torch.long),
        "topk_probs": torch.tensor(
            [top1_probability, second_probability], dtype=torch.float32
        ),
        "other_mass": torch.tensor(
            1.0 - top1_probability - second_probability, dtype=torch.float32
        ),
        "stability_streak": torch.tensor(streak, dtype=torch.int16),
    }


def _run(
    logits,
    previous_memory=None,
    budget=1,
    mode="none",
    directional=None,
    jsd_threshold=0.01,
    attention_threshold=0.006,
    extra_multiplier=1,
):
    x = torch.full((1, logits.shape[1]), MASK, dtype=torch.long)
    if directional is None and mode != "none":
        directional = torch.zeros((1, logits.shape[1], logits.shape[1]))
    return select_stcc_tokens(
        logits=logits,
        mask_index=x == MASK,
        x=x,
        base_budget=budget,
        block_start=0,
        candidate_topk=2,
        jsd_threshold=jsd_threshold,
        horizontal_mode=mode,
        directional_attention=directional,
        attention_threshold=attention_threshold,
        extra_multiplier=extra_multiplier,
        extra_jsd_threshold=jsd_threshold,
        min_topk_overlap=2,
        min_stability_streak=1,
        previous_memory=previous_memory,
    )


def test_distribution_response_is_not_top1_only():
    # Position 0 keeps top-1 but its distribution moves substantially.
    # Position 1 flips a near-tie with tiny distribution movement.  A response-
    # based rule should select the low-JSD position despite its top-1 flip.
    previous = [{
        0: _memory(0, 0.90, 1, 0.09),
        1: _memory(2, 0.51, 3, 0.49),
    }]
    logits = torch.log(torch.tensor([[
        [0.51, 0.48, 0.005, 0.005],
        [0.001, 0.001, 0.489, 0.509],
    ]], dtype=torch.float64))
    _, transfer, diagnostics, _ = _run(
        logits, previous_memory=previous, jsd_threshold=0.01
    )
    assert torch.where(transfer[0])[0].tolist() == [1]
    state = diagnostics["candidate_state"][0]
    assert state["top1_stable"].tolist() == [True, False]
    assert state["low_distribution_response"].tolist() == [False, True]
    assert state["response_class"].tolist() == [2, 1]


def test_directed_threshold_detects_pair_hidden_by_symmetric_average():
    logits = torch.tensor([[
        [5.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 3.0],
    ]])
    directional = torch.zeros((1, 3, 3), dtype=torch.float32)
    directional[0, 0, 1] = 0.009
    directional[0, 1, 0] = 0.001
    _, symmetric_transfer, _, _ = _run(
        logits,
        budget=2,
        mode="symmetric",
        directional=directional,
        attention_threshold=0.006,
    )
    _, directed_transfer, diagnostics, _ = _run(
        logits,
        budget=2,
        mode="directed",
        directional=directional,
        attention_threshold=0.006,
    )
    assert torch.where(symmetric_transfer[0])[0].tolist() == [0, 1]
    assert torch.where(directed_transfer[0])[0].tolist() == [0, 2]
    assert diagnostics["horizontal_active_directed_edges"] == 1
    assert diagnostics["horizontal_rejections"] == 1


def test_acceleration_adds_only_verified_low_response_candidates():
    previous = [{
        0: _memory(0, 0.80, 1, 0.19),
        1: _memory(1, 0.80, 0, 0.19),
        2: _memory(2, 0.80, 1, 0.19),
    }]
    logits = torch.log(torch.tensor([[
        [0.80, 0.19, 0.01],
        [0.19, 0.80, 0.01],
        [0.01, 0.19, 0.80],
    ]], dtype=torch.float64))
    _, transfer, diagnostics, _ = _run(
        logits,
        previous_memory=previous,
        budget=1,
        jsd_threshold=0.01,
        extra_multiplier=3,
    )
    assert transfer.sum().item() == 3
    assert diagnostics["base_commits"] == 1
    assert diagnostics["extra_commits"] == 2


def test_low_response_prunes_dense_directed_read():
    previous = [{
        0: _memory(0, 0.80, 1, 0.19),
        1: _memory(1, 0.80, 0, 0.19),
    }]
    logits = torch.log(torch.tensor([[
        [0.80, 0.19],
        [0.19, 0.80],
    ]], dtype=torch.float64))
    directional = torch.zeros((1, 2, 2), dtype=torch.float32)
    directional[0, 0, 1] = 0.90
    _, transfer, diagnostics, _ = _run(
        logits,
        previous_memory=previous,
        budget=2,
        mode="directed",
        directional=directional,
        jsd_threshold=0.01,
        attention_threshold=0.006,
    )
    assert transfer.sum().item() == 2
    assert diagnostics["horizontal_raw_directed_edges"] == 1
    assert diagnostics["horizontal_active_directed_edges"] == 0
    assert diagnostics["horizontal_pruned_low_response_edges"] == 1


def test_generate_acceleration_reduces_nfe_without_underfill():
    class FakeModel:
        device = torch.device("cpu")

        def __call__(self, x):
            logits = torch.zeros((1, x.shape[1], 6), dtype=torch.float32)
            for position in range(x.shape[1]):
                logits[0, position, position % 5] = 5.0
            return types.SimpleNamespace(logits=logits)

    prompt = torch.tensor([[5]], dtype=torch.long)
    quality, quality_nfe, quality_summary = generate_stcc(
        FakeModel(),
        prompt,
        steps=4,
        gen_length=4,
        block_length=4,
        mask_id=MASK,
        candidate_topk=2,
        jsd_threshold=0.01,
        horizontal_mode="none",
        extra_multiplier=1,
        collect_step_diagnostics=True,
    )
    accelerated, accelerated_nfe, accelerated_summary = generate_stcc(
        FakeModel(),
        prompt,
        steps=4,
        gen_length=4,
        block_length=4,
        mask_id=MASK,
        candidate_topk=2,
        jsd_threshold=0.01,
        horizontal_mode="none",
        extra_multiplier=3,
        extra_jsd_threshold=0.01,
        min_topk_overlap=2,
        min_stability_streak=1,
        collect_step_diagnostics=True,
    )
    assert quality_nfe == 4
    assert accelerated_nfe == 2
    assert torch.equal(quality, accelerated)
    assert not (accelerated == MASK).any()
    assert quality_summary["extra_commits"] == 0
    assert accelerated_summary["extra_commits"] == 2
    assert accelerated_summary["residual_mask_count"] == 0
