import torch

from generate import select_attention_stability_tokens


def make_logits(top_ids, confidences, vocab_size=8):
    logits = torch.full((1, len(top_ids), vocab_size), -20.0)
    for position, (token_id, confidence) in enumerate(zip(top_ids, confidences)):
        logits[0, position, token_id] = torch.logit(torch.tensor(confidence))
        logits[0, position, (token_id + 1) % vocab_size] = 0.0
    return logits


def select(logits, dependency, budget, tau, previous_top1=None, previous_selected=None):
    x = torch.full(logits.shape[:2], 7, dtype=torch.long)
    mask = torch.ones_like(x, dtype=torch.bool)
    return select_attention_stability_tokens(
        logits, 0.0, "low_confidence", mask, x, budget, dependency, tau, 0,
        previous_top1, previous_selected,
    )


def test_high_dependency_positions_are_not_committed_together():
    logits = make_logits([1, 2, 3], [0.9, 0.8, 0.7])
    dependency = torch.zeros((1, 3, 3))
    dependency[0, 0, 1] = dependency[0, 1, 0] = 0.8
    _, transfer, diagnostics = select(logits, dependency, budget=2, tau=0.5)
    assert transfer.nonzero(as_tuple=True)[1].tolist() == [0, 2]
    assert diagnostics["rejected_pairs"] == 1


def test_changed_dependent_candidate_is_ranked_after_mature_candidate():
    logits = make_logits([1, 2, 3], [0.95, 0.8, 0.7])
    dependency = torch.zeros((1, 3, 3))
    dependency[0, 0, 2] = dependency[0, 2, 0] = 0.8
    previous_top1 = torch.tensor([[4, 2, 5]])
    _, transfer, diagnostics = select(
        logits, dependency, budget=1, tau=0.5,
        previous_top1=previous_top1, previous_selected=torch.tensor([2]),
    )
    assert transfer.nonzero(as_tuple=True)[1].tolist() == [1]
    assert diagnostics["unstable_candidates"] == 1


def test_all_immature_candidates_fall_back_to_confidence():
    logits = make_logits([1, 2], [0.9, 0.8])
    dependency = torch.ones((1, 2, 2))
    previous_top1 = torch.tensor([[4, 5]])
    _, transfer, _ = select(
        logits, dependency, budget=1, tau=0.5,
        previous_top1=previous_top1, previous_selected=torch.tensor([0, 1]),
    )
    assert transfer.nonzero(as_tuple=True)[1].tolist() == [0]


def test_inactive_constraint_matches_baseline_topk():
    logits = make_logits([1, 2, 3], [0.6, 0.9, 0.7])
    dependency = torch.zeros((1, 3, 3))
    _, transfer, _ = select(logits, dependency, budget=2, tau=1.0)
    expected = torch.topk(torch.softmax(logits.double(), dim=-1).max(dim=-1).values[0], k=2).indices.sort().values
    actual = transfer.nonzero(as_tuple=True)[1].sort().values
    assert torch.equal(actual, expected)


if __name__ == "__main__":
    test_high_dependency_positions_are_not_committed_together()
    test_changed_dependent_candidate_is_ranked_after_mature_candidate()
    test_all_immature_candidates_fall_back_to_confidence()
    test_inactive_constraint_matches_baseline_topk()
    print("4 selector tests passed")
