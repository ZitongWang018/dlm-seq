from audit_lm_eval_task import metric_value, select_filter_records


def test_selects_one_gsm8k_filter_per_task():
    samples = [
        {"doc_id": 0, "filter": "strict-match", "exact_match": 1.0},
        {"doc_id": 0, "filter": "flexible-extract", "exact_match": 1.0},
        {"doc_id": 1, "filter": "strict-match", "exact_match": 0.0},
        {"doc_id": 1, "filter": "flexible-extract", "exact_match": 1.0},
    ]
    selected = select_filter_records(samples, "strict-match")
    assert [sample["doc_id"] for sample in selected] == [0, 1]
    assert [metric_value(sample, "exact_match") for sample in selected] == [1.0, 0.0]


def test_missing_filter_is_rejected():
    try:
        select_filter_records([{"filter": "strict-match"}], "missing")
    except ValueError as error:
        assert "available" in str(error)
    else:
        raise AssertionError("missing filter was accepted")


if __name__ == "__main__":
    test_selects_one_gsm8k_filter_per_task()
    test_missing_filter_is_rejected()
    print("2 lm-eval auditor tests passed")
