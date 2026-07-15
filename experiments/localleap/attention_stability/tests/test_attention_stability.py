import torch

from generate import select_attention_stability_tokens


def make_logits(top_ids, confidences, vocab_size=8):
    logits = torch.full((1, len(top_ids), vocab_size), -20.0)
    for position, (token_id, confidence) in enumerate(zip(top_ids, confidences)):
        logits[0, position, token_id] = torch.logit(torch.tensor(confidence))
        logits[0, position, (token_id + 1) % vocab_size] = 0.0
    return logits


def select(
    logits,
    dependency,
    budget,
    tau,
    previous_top1=None,
    previous_selected=None,
    previous_topk_ids=None,
    temporal_mode="top1",
    temporal_topk=4,
    prune_stable_conflicts=False,
    fill_budget=False,
):
    x = torch.full(logits.shape[:2], 7, dtype=torch.long)
    mask = torch.ones_like(x, dtype=torch.bool)
    return select_attention_stability_tokens(
        logits, 0.0, "low_confidence", mask, x, budget, dependency, tau, 0,
        previous_top1, previous_selected, previous_topk_ids, temporal_mode,
        temporal_topk, prune_stable_conflicts, fill_budget,
    )


def test_high_dependency_positions_are_not_committed_together():
    logits = make_logits([1, 2, 3], [0.9, 0.8, 0.7])
    dependency = torch.zeros((1, 3, 3))
    dependency[0, 0, 1] = dependency[0, 1, 0] = 0.8
    _, transfer, diagnostics, _ = select(logits, dependency, budget=2, tau=0.5)
    assert transfer.nonzero(as_tuple=True)[1].tolist() == [0, 2]
    assert diagnostics["rejected_pairs"] == 1


def test_changed_dependent_candidate_is_ranked_after_mature_candidate():
    logits = make_logits([1, 2, 3], [0.95, 0.8, 0.7])
    dependency = torch.zeros((1, 3, 3))
    dependency[0, 0, 2] = dependency[0, 2, 0] = 0.8
    previous_top1 = torch.tensor([[4, 2, 5]])
    _, transfer, diagnostics, _ = select(
        logits, dependency, budget=1, tau=0.5,
        previous_top1=previous_top1, previous_selected=torch.tensor([2]),
    )
    assert transfer.nonzero(as_tuple=True)[1].tolist() == [1]
    assert diagnostics["unstable_candidates"] == 1


def test_all_immature_candidates_fall_back_to_confidence():
    logits = make_logits([1, 2], [0.9, 0.8])
    dependency = torch.ones((1, 2, 2))
    previous_top1 = torch.tensor([[4, 5]])
    _, transfer, _, _ = select(
        logits, dependency, budget=1, tau=0.5,
        previous_top1=previous_top1, previous_selected=torch.tensor([0, 1]),
    )
    assert transfer.nonzero(as_tuple=True)[1].tolist() == [0]


def test_inactive_constraint_matches_baseline_topk():
    logits = make_logits([1, 2, 3], [0.6, 0.9, 0.7])
    dependency = torch.zeros((1, 3, 3))
    _, transfer, _, _ = select(logits, dependency, budget=2, tau=1.0)
    expected = torch.topk(torch.softmax(logits.double(), dim=-1).max(dim=-1).values[0], k=2).indices.sort().values
    actual = transfer.nonzero(as_tuple=True)[1].sort().values
    assert torch.equal(actual, expected)


def test_candidate_diagnostics_capture_temporal_state():
    logits = make_logits([1, 2, 3], [0.95, 0.8, 0.7])
    dependency = torch.zeros((1, 3, 3))
    dependency[0, 0, 2] = dependency[0, 2, 0] = 0.8
    previous_top1 = torch.tensor([[4, 2, 5]])
    _, _, diagnostics, _ = select(
        logits, dependency, budget=1, tau=0.5,
        previous_top1=previous_top1, previous_selected=torch.tensor([2]),
    )
    state = diagnostics["candidate_state"][0]
    assert state["masked_positions_global"].tolist() == [0, 1, 2]
    assert state["top1_token_ids"].tolist() == [1, 2, 3]
    assert state["previous_top1_token_ids"].tolist() == [4, 2, 5]
    assert state["candidate_changed"].tolist() == [True, False, True]
    assert state["maturity"].tolist() == [False, True, True]
    assert torch.allclose(state["max_dependency_to_previous"], torch.tensor([0.8, 0.0, 0.0]))


def test_directed_read_dependency_is_not_implicitly_symmetrized():
    logits = make_logits([1, 2, 3], [0.9, 0.8, 0.7])
    dependency = torch.zeros((1, 3, 3))
    dependency[0, 0, 1] = 0.9
    _, transfer, diagnostics, _ = select(logits, dependency, budget=2, tau=0.5)
    assert transfer.nonzero(as_tuple=True)[1].tolist() == [0, 1]
    assert diagnostics["rejected_pairs"] == 0


def test_stable_dense_conflict_can_be_pruned_without_new_threshold():
    logits = make_logits([1, 2, 3], [0.9, 0.8, 0.7])
    dependency = torch.zeros((1, 3, 3))
    dependency[0, 1, 0] = 0.9
    previous_top1 = torch.tensor([[1, 2, 3]])
    _, transfer, diagnostics, _ = select(
        logits,
        dependency,
        budget=2,
        tau=0.5,
        previous_top1=previous_top1,
        previous_selected=torch.tensor([2]),
        prune_stable_conflicts=True,
    )
    assert transfer.nonzero(as_tuple=True)[1].tolist() == [0, 1]
    assert diagnostics["stable_conflicts_pruned"] == 1


def test_fixed_budget_fill_guarantees_parallel_commit_count():
    logits = make_logits([1, 2, 3], [0.9, 0.8, 0.7])
    dependency = torch.zeros((1, 3, 3))
    dependency[0, 1, 0] = dependency[0, 2, 0] = 0.9
    previous_top1 = torch.tensor([[4, 5, 6]])
    _, transfer, diagnostics, _ = select(
        logits,
        dependency,
        budget=2,
        tau=0.5,
        previous_top1=previous_top1,
        previous_selected=torch.tensor([2]),
        fill_budget=True,
    )
    assert int(transfer.sum().item()) == 2
    assert diagnostics["forced_budget_fills"] == 1
    assert diagnostics["underfilled"] is False


def test_topk_overlap_creates_intermediate_temporal_tier():
    logits = make_logits([1, 2, 3], [0.70, 0.65, 0.95])
    dependency = torch.ones((1, 3, 3))
    previous_top1 = torch.tensor([[4, 5, 6]])
    previous_topk = torch.tensor([[[4, 1], [5, 7], [6, 7]]])
    _, transfer, diagnostics, _ = select(
        logits,
        dependency,
        budget=1,
        tau=0.5,
        previous_top1=previous_top1,
        previous_selected=torch.tensor([0]),
        previous_topk_ids=previous_topk,
        temporal_mode="topk_overlap",
        temporal_topk=2,
    )
    state = diagnostics["candidate_state"][0]
    assert state["temporal_tier"].tolist() == [1, 0, 0]
    assert transfer.nonzero(as_tuple=True)[1].tolist() == [0]
    assert diagnostics["intermediate_candidates"] == 1


def test_topk_overlap_rejects_invalid_k():
    logits = make_logits([1], [0.9])
    dependency = torch.zeros((1, 1, 1))
    try:
        select(logits, dependency, budget=1, tau=0.5, temporal_topk=99)
    except ValueError as error:
        assert "temporal_topk" in str(error)
    else:
        raise AssertionError("invalid temporal_topk was accepted")


if __name__ == "__main__":
    test_high_dependency_positions_are_not_committed_together()
    test_changed_dependent_candidate_is_ranked_after_mature_candidate()
    test_all_immature_candidates_fall_back_to_confidence()
    test_inactive_constraint_matches_baseline_topk()
    test_candidate_diagnostics_capture_temporal_state()
    test_directed_read_dependency_is_not_implicitly_symmetrized()
    test_stable_dense_conflict_can_be_pruned_without_new_threshold()
    test_fixed_budget_fill_guarantees_parallel_commit_count()
    test_topk_overlap_creates_intermediate_temporal_tier()
    test_topk_overlap_rejects_invalid_k()
    print("10 selector tests passed")
