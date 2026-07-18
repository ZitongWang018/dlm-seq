import torch
import generate as generate_module

from generate import (
    generate_trajectory_likelihood_selection,
    score_committed_tokens,
    score_disagreement_evidence,
    score_baseline_consensus,
    score_shared_skeleton_candidates,
    score_bidirectional_block_candidates,
    select_attention_stability_tokens,
    select_likelihood_trajectory,
)


class SharedSkeletonModel:
    def __init__(self):
        self.inputs = []

    def __call__(self, input_ids):
        self.inputs.append(input_ids.clone())
        logits = torch.zeros((*input_ids.shape, 10))
        logits[0, 2, 4] = 4.0
        logits[0, 3, 5] = 4.0
        return type("Output", (), {"logits": logits})()


def test_shared_skeleton_scores_both_paths_from_identical_context():
    model = SharedSkeletonModel()
    scores, disagreements, nfe = score_shared_skeleton_candidates(
        model,
        {
            "fast": torch.tensor([[9, 1, 2, 3]]),
            "accuracy": torch.tensor([[9, 1, 4, 5]]),
        },
        prompt_length=1,
        mask_id=7,
    )
    assert model.inputs[0].tolist() == [[9, 1, 7, 7]]
    assert scores["accuracy"] > scores["fast"]
    assert disagreements == 2
    assert nfe == 1


def test_shared_skeleton_tie_preserves_fast_without_extra_forward():
    model = SharedSkeletonModel()
    scores, disagreements, nfe = score_shared_skeleton_candidates(
        model,
        {
            "fast": torch.tensor([[9, 1, 2]]),
            "accuracy": torch.tensor([[9, 1, 2]]),
        },
        prompt_length=1,
        mask_id=7,
    )
    assert scores == {"fast": 0.0, "accuracy": 0.0}
    assert disagreements == 0
    assert nfe == 0
    assert model.inputs == []


class BidirectionalBlockModel:
    def __init__(self):
        self.inputs = []

    def __call__(self, input_ids):
        self.inputs.append(input_ids.clone())
        logits = torch.zeros((*input_ids.shape, 10))
        if len(self.inputs) == 1:
            logits[:, 2, 4] = 4.0
        else:
            logits[:, 4, 6] = 4.0
        return type("Output", (), {"logits": logits})()


def test_bidirectional_block_masks_one_block_under_both_external_drafts():
    model = BidirectionalBlockModel()
    scores, disagreements, nfe, blocks = score_bidirectional_block_candidates(
        model,
        {
            "fast": torch.tensor([[9, 1, 2, 3, 4]]),
            "accuracy": torch.tensor([[9, 1, 4, 3, 6]]),
        },
        prompt_length=1,
        block_length=2,
        mask_id=7,
    )
    assert model.inputs[0].tolist() == [
        [9, 1, 7, 3, 4],
        [9, 1, 7, 3, 6],
    ]
    assert model.inputs[1].tolist() == [
        [9, 1, 2, 3, 7],
        [9, 1, 4, 3, 7],
    ]
    assert scores["accuracy"] > scores["fast"]
    assert disagreements == 2
    assert nfe == 2
    assert [row["disagreement_token_count"] for row in blocks] == [1, 1]


def test_bidirectional_block_identical_paths_need_no_selector_forward():
    model = BidirectionalBlockModel()
    scores, disagreements, nfe, blocks = score_bidirectional_block_candidates(
        model,
        {
            "fast": torch.tensor([[9, 1, 2]]),
            "accuracy": torch.tensor([[9, 1, 2]]),
        },
        prompt_length=1,
        block_length=2,
        mask_id=7,
    )
    assert scores == {"fast": 0.0, "accuracy": 0.0}
    assert disagreements == 0
    assert nfe == 0
    assert blocks == []
    assert model.inputs == []


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
    previous_response_credit=None,
):
    x = torch.full(logits.shape[:2], 7, dtype=torch.long)
    mask = torch.ones_like(x, dtype=torch.bool)
    return select_attention_stability_tokens(
        logits, 0.0, "low_confidence", mask, x, budget, dependency, tau, 0,
        previous_top1, previous_selected, previous_topk_ids, temporal_mode,
        temporal_topk, prune_stable_conflicts, fill_budget,
        previous_response_credit,
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


def test_risk_switch_fills_stable_pairs_but_waits_after_conditioned_rewrite():
    logits = make_logits([1, 2, 3], [0.95, 0.90, 0.80])
    dependency = torch.zeros((1, 3, 3))
    dependency[0, 1, 0] = dependency[0, 0, 1] = 0.9
    dependency[0, 1, 2] = 0.9
    previous_selected = torch.tensor([2])
    x = torch.tensor([[7, 7, 3]])
    mask = torch.tensor([[True, True, False]])

    # With no adjacent-step rewrite, the dense pair is low-information and
    # both candidates retain the native parallel budget.
    _, stable_transfer, stable_diagnostics, _ = select_attention_stability_tokens(
        logits=logits,
        temperature=0.0,
        remasking="low_confidence",
        mask_index=mask,
        x=x,
        budget=2,
        dependency=dependency,
        dependency_threshold=0.5,
        block_start=0,
        previous_top1=torch.tensor([[1, 2, 3]]),
        previous_selected=previous_selected,
        prune_stable_conflicts=True,
        fill_budget=False,
    )
    assert stable_transfer.nonzero(as_tuple=True)[1].tolist() == [0, 1]
    assert stable_diagnostics["stable_conflicts_pruned"] == 1
    assert not stable_diagnostics["underfilled"]

    # Once position 1 changes after the new condition arrives, the same
    # horizontal edge is informative.  The decoder commits only the parent
    # anchor and waits for another ordinary forward instead of force-filling.
    _, risky_transfer, risky_diagnostics, _ = select_attention_stability_tokens(
        logits=logits,
        temperature=0.0,
        remasking="low_confidence",
        mask_index=mask,
        x=x,
        budget=2,
        dependency=dependency,
        dependency_threshold=0.5,
        block_start=0,
        previous_top1=torch.tensor([[1, 7, 3]]),
        previous_selected=previous_selected,
        prune_stable_conflicts=True,
        fill_budget=False,
    )
    assert risky_transfer.nonzero(as_tuple=True)[1].tolist() == [0]
    assert risky_diagnostics["rejected_pairs"] == 1
    assert risky_diagnostics["underfilled"]
    assert risky_diagnostics["forced_budget_fills"] == 0


def test_likelihood_trajectory_selection_uses_vertical_path_score():
    summaries = {
        "fast": {"commit_logprob_mean": -0.20},
        "accuracy": {"commit_logprob_mean": -0.12},
    }
    assert select_likelihood_trajectory(summaries) == "accuracy"


def test_committed_token_score_uses_selected_positions_only():
    candidate_state = {
        "masked_positions_global": torch.tensor([3, 5, 8]),
        "selected_positions_global": torch.tensor([8, 3]),
        "top1_confidences": torch.tensor([0.8, 0.1, 0.5]),
    }
    score, count = score_committed_tokens(candidate_state)
    assert count == 2
    assert torch.isclose(torch.tensor(score), torch.log(torch.tensor(0.4)))


def test_likelihood_trajectory_selection_tie_preserves_fast_parent():
    summaries = {
        "fast": {"commit_logprob_mean": -0.20},
        "accuracy": {"commit_logprob_mean": -0.20},
    }
    assert select_likelihood_trajectory(summaries) == "fast"


def test_block_evidence_requires_one_nat_per_existing_block():
    weak = {
        "fast": {"commit_logprob_mean": -0.20},
        "accuracy": {"commit_logprob_mean": -0.18},
    }
    strong = {
        "fast": {"commit_logprob_mean": -0.20},
        "accuracy": {"commit_logprob_mean": -0.16},
    }
    assert select_likelihood_trajectory(
        weak, block_length=32, selection_mode="block_evidence"
    ) == "fast"
    assert select_likelihood_trajectory(
        strong, block_length=32, selection_mode="block_evidence"
    ) == "accuracy"


def test_disagreement_evidence_ignores_shared_tokens():
    token_ids = {
        "fast": torch.tensor([[10, 20, 30, 40]]),
        "accuracy": torch.tensor([[10, 21, 30, 41]]),
    }
    summaries = {
        "fast": {
            "commit_logprob_mean": -0.20,
            "_commit_logprob_by_position": torch.tensor([-9.0, -0.2, -9.0, -0.3]),
        },
        "accuracy": {
            "commit_logprob_mean": -0.30,
            "_commit_logprob_by_position": torch.tensor([-0.01, -0.1, -0.01, -0.1]),
        },
    }
    scores, disagreement_count, scored_count = score_disagreement_evidence(
        token_ids, summaries
    )
    assert disagreement_count == 2
    assert scored_count == 2
    assert scores["accuracy"] > scores["fast"]
    assert select_likelihood_trajectory(
        summaries,
        selection_mode="disagreement_evidence",
        candidate_token_ids=token_ids,
    ) == "accuracy"


def test_disagreement_evidence_tie_preserves_fast_parent():
    token_ids = {
        "fast": torch.tensor([[1, 2]]),
        "accuracy": torch.tensor([[1, 2]]),
    }
    summaries = {
        "fast": {
            "commit_logprob_mean": -0.2,
            "_commit_logprob_by_position": torch.tensor([-0.2, -0.2]),
        },
        "accuracy": {
            "commit_logprob_mean": -0.1,
            "_commit_logprob_by_position": torch.tensor([-0.1, -0.1]),
        },
    }
    assert select_likelihood_trajectory(
        summaries,
        selection_mode="disagreement_evidence",
        candidate_token_ids=token_ids,
    ) == "fast"


def test_consensus_block_requires_vertical_evidence_and_horizontal_vote():
    token_ids = {
        "fast": torch.tensor([[1, 2, 3, 4]]),
        "accuracy": torch.tensor([[1, 8, 3, 9]]),
        "baseline": torch.tensor([[1, 8, 3, 9]]),
    }
    strong = {
        "fast": {"commit_logprob_mean": -0.20},
        "accuracy": {"commit_logprob_mean": -0.15},
    }
    weak = {
        "fast": {"commit_logprob_mean": -0.20},
        "accuracy": {"commit_logprob_mean": -0.18},
    }
    consensus = score_baseline_consensus(token_ids)
    assert consensus == {
        "disagreement_token_count": 2,
        "baseline_fast_matches": 0,
        "baseline_accuracy_matches": 2,
    }
    assert select_likelihood_trajectory(
        strong,
        block_length=32,
        selection_mode="consensus_block",
        candidate_token_ids=token_ids,
    ) == "accuracy"
    assert select_likelihood_trajectory(
        weak,
        block_length=32,
        selection_mode="consensus_block",
        candidate_token_ids=token_ids,
    ) == "fast"


def test_consensus_block_preserves_fast_when_baseline_votes_fast():
    token_ids = {
        "fast": torch.tensor([[1, 2, 3, 4]]),
        "accuracy": torch.tensor([[1, 8, 3, 9]]),
        "baseline": torch.tensor([[1, 2, 3, 4]]),
    }
    summaries = {
        "fast": {"commit_logprob_mean": -0.20},
        "accuracy": {"commit_logprob_mean": -0.10},
    }
    assert select_likelihood_trajectory(
        summaries,
        block_length=32,
        selection_mode="consensus_block",
        candidate_token_ids=token_ids,
    ) == "fast"


def test_lazy_consensus_skips_baseline_until_vertical_override_is_possible():
    original_attention = generate_module.generate_attention_stability
    original_baseline = generate_module.generate
    baseline_calls = []

    def fake_attention(**kwargs):
        if kwargs["prune_stable_conflicts"]:
            return torch.tensor([[1, 2]]), 128, {
                "commit_logprob_mean": -0.20,
            }
        return torch.tensor([[1, 8]]), 144, {
            "commit_logprob_mean": -0.18,
        }

    def fake_baseline(**kwargs):
        baseline_calls.append(True)
        return torch.tensor([[1, 8]]), 128

    generate_module.generate_attention_stability = fake_attention
    generate_module.generate = fake_baseline
    try:
        output, nfe, summary = generate_trajectory_likelihood_selection(
            model=object(),
            prompt=torch.tensor([[1]]),
            dependency_threshold=0.004,
            steps=128,
            gen_length=2,
            block_length=32,
            selection_mode="lazy_consensus_block",
        )
    finally:
        generate_module.generate_attention_stability = original_attention
        generate_module.generate = original_baseline

    assert baseline_calls == []
    assert output.tolist() == [[1, 2]]
    assert nfe == 272
    assert summary["selected_name"] == "fast"
    assert summary["baseline_consensus"] is None
    assert summary["candidate_nfe"] == {"fast": 128, "accuracy": 144}


def test_lazy_consensus_matches_full_consensus_when_vertical_evidence_is_strong():
    original_attention = generate_module.generate_attention_stability
    original_baseline = generate_module.generate
    baseline_calls = []

    def fake_attention(**kwargs):
        if kwargs["prune_stable_conflicts"]:
            return torch.tensor([[1, 2]]), 128, {
                "commit_logprob_mean": -0.20,
            }
        return torch.tensor([[1, 8]]), 144, {
            "commit_logprob_mean": -0.10,
        }

    def fake_baseline(**kwargs):
        baseline_calls.append(True)
        return torch.tensor([[1, 8]]), 128

    generate_module.generate_attention_stability = fake_attention
    generate_module.generate = fake_baseline
    try:
        output, nfe, summary = generate_trajectory_likelihood_selection(
            model=object(),
            prompt=torch.tensor([[1]]),
            dependency_threshold=0.004,
            steps=128,
            gen_length=2,
            block_length=32,
            selection_mode="lazy_consensus_block",
        )
    finally:
        generate_module.generate_attention_stability = original_attention
        generate_module.generate = original_baseline

    assert baseline_calls == [True]
    assert output.tolist() == [[1, 8]]
    assert nfe == 400
    assert summary["selected_name"] == "accuracy"
    assert summary["baseline_consensus"] == {
        "disagreement_token_count": 1,
        "baseline_fast_matches": 0,
        "baseline_accuracy_matches": 1,
    }


def test_coverage_consensus_accepts_one_extra_revision_per_disagreement():
    token_ids = {
        "fast": torch.tensor([[1, 2, 3, 4]]),
        "accuracy": torch.tensor([[1, 8, 3, 9]]),
        "baseline": torch.tensor([[1, 8, 3, 9]]),
    }
    summaries = {
        "fast": {
            "commit_logprob_mean": -0.20,
            "response_invalidations": 10,
        },
        "accuracy": {
            "commit_logprob_mean": -0.19,
            "response_invalidations": 12,
        },
    }
    assert select_likelihood_trajectory(
        summaries,
        block_length=32,
        selection_mode="coverage_consensus_block",
        candidate_token_ids=token_ids,
    ) == "accuracy"


def test_coverage_consensus_preserves_fast_when_revisions_do_not_cover_differences():
    token_ids = {
        "fast": torch.tensor([[1, 2, 3, 4]]),
        "accuracy": torch.tensor([[1, 8, 3, 9]]),
    }
    summaries = {
        "fast": {
            "commit_logprob_mean": -0.20,
            "response_invalidations": 10,
        },
        "accuracy": {
            "commit_logprob_mean": -0.19,
            "response_invalidations": 11,
        },
    }
    assert select_likelihood_trajectory(
        summaries,
        block_length=32,
        selection_mode="coverage_consensus_block",
        candidate_token_ids=token_ids,
    ) == "fast"


def test_coverage_consensus_runs_horizontal_vote_for_covered_weak_margin():
    original_attention = generate_module.generate_attention_stability
    original_baseline = generate_module.generate
    baseline_calls = []

    def fake_attention(**kwargs):
        if kwargs["prune_stable_conflicts"]:
            return torch.tensor([[1, 2, 3]]), 128, {
                "commit_logprob_mean": -0.20,
                "response_invalidations": 10,
            }
        return torch.tensor([[1, 8, 9]]), 144, {
            "commit_logprob_mean": -0.19,
            "response_invalidations": 12,
        }

    def fake_baseline(**kwargs):
        baseline_calls.append(True)
        return torch.tensor([[1, 8, 9]]), 128

    generate_module.generate_attention_stability = fake_attention
    generate_module.generate = fake_baseline
    try:
        output, nfe, summary = generate_trajectory_likelihood_selection(
            model=object(),
            prompt=torch.tensor([[1]]),
            dependency_threshold=0.004,
            steps=128,
            gen_length=3,
            block_length=32,
            selection_mode="coverage_consensus_block",
        )
    finally:
        generate_module.generate_attention_stability = original_attention
        generate_module.generate = original_baseline

    assert baseline_calls == [True]
    assert output.tolist() == [[1, 8, 9]]
    assert nfe == 400
    assert summary["selected_name"] == "accuracy"
    assert summary["revision_coverage"] == {
        "disagreement_token_count": 2,
        "extra_response_invalidations": 2,
        "satisfied": True,
    }


def test_convergent_consensus_accepts_strong_evidence_with_fewer_invalidations():
    token_ids = {
        "fast": torch.tensor([[1, 2, 3, 4]]),
        "accuracy": torch.tensor([[1, 8, 3, 9]]),
        "baseline": torch.tensor([[1, 8, 3, 9]]),
    }
    summaries = {
        "fast": {
            "commit_logprob_mean": -0.20,
            "response_invalidations": 12,
        },
        "accuracy": {
            "commit_logprob_mean": -0.10,
            "response_invalidations": 10,
        },
    }
    assert select_likelihood_trajectory(
        summaries,
        block_length=32,
        selection_mode="convergent_coverage_consensus_block",
        candidate_token_ids=token_ids,
    ) == "accuracy"


def test_convergent_consensus_rejects_strong_but_uncovered_divergence():
    token_ids = {
        "fast": torch.tensor([[1, 2, 3, 4]]),
        "accuracy": torch.tensor([[1, 8, 3, 9]]),
    }
    summaries = {
        "fast": {
            "commit_logprob_mean": -0.20,
            "response_invalidations": 10,
        },
        "accuracy": {
            "commit_logprob_mean": -0.10,
            "response_invalidations": 11,
        },
    }
    assert select_likelihood_trajectory(
        summaries,
        block_length=32,
        selection_mode="convergent_coverage_consensus_block",
        candidate_token_ids=token_ids,
    ) == "fast"


def test_convergent_consensus_accepts_comprehensive_revision_coverage():
    token_ids = {
        "fast": torch.tensor([[1, 2, 3, 4]]),
        "accuracy": torch.tensor([[1, 8, 3, 9]]),
        "baseline": torch.tensor([[1, 8, 3, 9]]),
    }
    summaries = {
        "fast": {
            "commit_logprob_mean": -0.20,
            "response_invalidations": 10,
        },
        "accuracy": {
            "commit_logprob_mean": -0.19,
            "response_invalidations": 12,
        },
    }
    assert select_likelihood_trajectory(
        summaries,
        block_length=32,
        selection_mode="convergent_coverage_consensus_block",
        candidate_token_ids=token_ids,
    ) == "accuracy"


def test_convergent_consensus_skips_vote_for_uncovered_divergent_path():
    original_attention = generate_module.generate_attention_stability
    original_baseline = generate_module.generate
    baseline_calls = []

    def fake_attention(**kwargs):
        if kwargs["prune_stable_conflicts"]:
            return torch.tensor([[1, 2, 3]]), 128, {
                "commit_logprob_mean": -0.20,
                "response_invalidations": 10,
            }
        return torch.tensor([[1, 8, 9]]), 144, {
            "commit_logprob_mean": -0.10,
            "response_invalidations": 11,
        }

    def fake_baseline(**kwargs):
        baseline_calls.append(True)
        return torch.tensor([[1, 8, 9]]), 128

    generate_module.generate_attention_stability = fake_attention
    generate_module.generate = fake_baseline
    try:
        output, nfe, summary = generate_trajectory_likelihood_selection(
            model=object(),
            prompt=torch.tensor([[1]]),
            dependency_threshold=0.004,
            steps=128,
            gen_length=3,
            block_length=32,
            selection_mode="convergent_coverage_consensus_block",
        )
    finally:
        generate_module.generate_attention_stability = original_attention
        generate_module.generate = original_baseline

    assert baseline_calls == []
    assert output.tolist() == [[1, 2, 3]]
    assert nfe == 272
    assert summary["selected_name"] == "fast"
    assert summary["revision_coverage"] == {
        "disagreement_token_count": 2,
        "extra_response_invalidations": 1,
        "satisfied": False,
    }


def test_topk_overlap_creates_intermediate_temporal_tier():
    logits = make_logits([1, 2, 3], [0.70, 0.80, 0.95])
    dependency = torch.zeros((1, 3, 3))
    dependency[0, :, 2] = 1.0
    previous_top1 = torch.tensor([[4, 2, 6]])
    previous_topk = torch.tensor([[[4, 1], [2, 7], [6, 7]]])
    _, transfer, diagnostics, _ = select(
        logits,
        dependency,
        budget=2,
        tau=0.5,
        previous_top1=previous_top1,
        previous_selected=torch.tensor([2]),
        previous_topk_ids=previous_topk,
        temporal_mode="topk_overlap",
        temporal_topk=2,
    )
    state = diagnostics["candidate_state"][0]
    assert state["temporal_tier"].tolist() == [1, 2, 0]
    assert transfer.nonzero(as_tuple=True)[1].tolist() == [0, 1]
    assert diagnostics["intermediate_candidates"] == 1


def test_topk_overlap_preserves_parent_confidence_order_for_mature_candidates():
    logits = make_logits([1, 2], [0.70, 0.95])
    dependency = torch.ones((1, 2, 2))
    previous_top1 = torch.tensor([[1, 2]])
    previous_topk = torch.tensor([[[1, 2], [2, 7]]])
    _, transfer, _, _ = select(
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
    assert transfer.nonzero(as_tuple=True)[1].tolist() == [1]


def test_topk_overlap_rejects_invalid_k():
    logits = make_logits([1], [0.9])
    dependency = torch.zeros((1, 1, 1))
    try:
        select(logits, dependency, budget=1, tau=0.5, temporal_topk=99)
    except ValueError as error:
        assert "temporal_topk" in str(error)
    else:
        raise AssertionError("invalid temporal_topk was accepted")


def test_response_credit_counts_only_strong_conditioning_events():
    logits = make_logits([1, 2, 3], [0.80, 0.95, 0.70])
    dependency = torch.zeros((1, 3, 3))
    dependency[0, 0, 2] = 0.9
    dependency[0, 1, 2] = 0.9
    previous_top1 = torch.tensor([[1, 5, 3]])
    previous_credit = torch.tensor([[2, 4, 7]], dtype=torch.int16)
    _, _, diagnostics, _ = select(
        logits,
        dependency,
        budget=1,
        tau=0.5,
        previous_top1=previous_top1,
        previous_selected=torch.tensor([2]),
        temporal_mode="response_credit",
        previous_response_credit=previous_credit,
    )
    state = diagnostics["candidate_state"][0]
    assert state["response_credit"].tolist() == [3, 0, 7]
    assert diagnostics["response_validations"] == 1
    assert diagnostics["response_invalidations"] == 1


def test_response_credit_precedes_confidence_within_mature_tier():
    logits = make_logits([1, 2, 3], [0.80, 0.95, 0.70])
    dependency = torch.zeros((1, 3, 3))
    dependency[0, 0, 2] = dependency[0, 1, 2] = 0.9
    previous_top1 = torch.tensor([[1, 2, 3]])
    previous_credit = torch.tensor([[3, 0, 0]], dtype=torch.int16)
    _, transfer, diagnostics, _ = select(
        logits,
        dependency,
        budget=1,
        tau=0.5,
        previous_top1=previous_top1,
        previous_selected=torch.tensor([2]),
        temporal_mode="response_credit",
        previous_response_credit=previous_credit,
    )
    assert transfer.nonzero(as_tuple=True)[1].tolist() == [0]
    assert diagnostics["candidate_state"][0]["ordered_positions_global"].tolist()[0] == 0


def test_response_credit_saturates_without_int16_wraparound():
    logits = make_logits([1], [0.9])
    dependency = torch.ones((1, 1, 1))
    _, _, diagnostics, _ = select(
        logits,
        dependency,
        budget=1,
        tau=0.5,
        previous_top1=torch.tensor([[1]]),
        previous_selected=torch.tensor([0]),
        temporal_mode="response_credit",
        previous_response_credit=torch.tensor([[32767]], dtype=torch.int16),
    )
    assert diagnostics["candidate_state"][0]["response_credit"].tolist() == [32767]


def test_revision_margin_prioritizes_decisive_conditioned_change():
    logits = torch.full((1, 3, 8), -20.0)
    # Position 0 is less confident overall because several alternatives remain,
    # but the new token decisively displaced the previous token (log-ratio 4).
    logits[0, 0, 1] = 2.0
    logits[0, 0, 4] = -2.0
    logits[0, 0, [0, 2, 3, 5, 6, 7]] = 0.0
    # Position 1 has higher top-1 confidence but only narrowly displaced its
    # previous candidate (log-ratio 0.2).
    logits[0, 1, 2] = 3.0
    logits[0, 1, 5] = 2.8
    logits[0, 2, 3] = 3.0
    x = torch.tensor([[7, 7, 3]])
    mask = torch.tensor([[True, True, False]])
    dependency = torch.zeros((1, 3, 3))
    dependency[0, 0, 2] = dependency[0, 1, 2] = 0.9
    previous_top1 = torch.tensor([[4, 5, 3]])
    _, transfer, diagnostics, _ = select_attention_stability_tokens(
        logits=logits,
        temperature=0.0,
        remasking="low_confidence",
        mask_index=mask,
        x=x,
        budget=1,
        dependency=dependency,
        dependency_threshold=0.5,
        block_start=0,
        previous_top1=previous_top1,
        previous_selected=torch.tensor([2]),
        temporal_mode="revision_margin",
    )
    state = diagnostics["candidate_state"][0]
    assert state["top1_confidences"][1] > state["top1_confidences"][0]
    assert state["revision_margin"].tolist()[0] > state["revision_margin"].tolist()[1]
    assert transfer.nonzero(as_tuple=True)[1].tolist() == [0]


if __name__ == "__main__":
    test_bidirectional_block_masks_one_block_under_both_external_drafts()
    test_bidirectional_block_identical_paths_need_no_selector_forward()
    test_shared_skeleton_scores_both_paths_from_identical_context()
    test_shared_skeleton_tie_preserves_fast_without_extra_forward()
    test_high_dependency_positions_are_not_committed_together()
    test_changed_dependent_candidate_is_ranked_after_mature_candidate()
    test_all_immature_candidates_fall_back_to_confidence()
    test_inactive_constraint_matches_baseline_topk()
    test_candidate_diagnostics_capture_temporal_state()
    test_directed_read_dependency_is_not_implicitly_symmetrized()
    test_stable_dense_conflict_can_be_pruned_without_new_threshold()
    test_fixed_budget_fill_guarantees_parallel_commit_count()
    test_risk_switch_fills_stable_pairs_but_waits_after_conditioned_rewrite()
    test_likelihood_trajectory_selection_uses_vertical_path_score()
    test_committed_token_score_uses_selected_positions_only()
    test_likelihood_trajectory_selection_tie_preserves_fast_parent()
    test_block_evidence_requires_one_nat_per_existing_block()
    test_disagreement_evidence_ignores_shared_tokens()
    test_disagreement_evidence_tie_preserves_fast_parent()
    test_consensus_block_requires_vertical_evidence_and_horizontal_vote()
    test_consensus_block_preserves_fast_when_baseline_votes_fast()
    test_lazy_consensus_skips_baseline_until_vertical_override_is_possible()
    test_lazy_consensus_matches_full_consensus_when_vertical_evidence_is_strong()
    test_coverage_consensus_accepts_one_extra_revision_per_disagreement()
    test_coverage_consensus_preserves_fast_when_revisions_do_not_cover_differences()
    test_coverage_consensus_runs_horizontal_vote_for_covered_weak_margin()
    test_convergent_consensus_accepts_strong_evidence_with_fewer_invalidations()
    test_convergent_consensus_rejects_strong_but_uncovered_divergence()
    test_convergent_consensus_accepts_comprehensive_revision_coverage()
    test_convergent_consensus_skips_vote_for_uncovered_divergent_path()
    test_topk_overlap_creates_intermediate_temporal_tier()
    test_topk_overlap_preserves_parent_confidence_order_for_mature_candidates()
    test_topk_overlap_rejects_invalid_k()
    test_response_credit_counts_only_strong_conditioning_events()
    test_response_credit_precedes_confidence_within_mature_tier()
    test_response_credit_saturates_without_int16_wraparound()
    test_revision_margin_prioritizes_decisive_conditioned_change()
    print("37 selector tests passed")
