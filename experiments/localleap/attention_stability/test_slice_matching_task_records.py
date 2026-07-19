from slice_matching_task_records import slice_matching


def row(identity):
    return {"stable_task_id": str(identity), "correct": False}


def test_subset_follows_reference_identity_order():
    selected = slice_matching([row(11), row(12), row(13)], [row(13), row(11)])
    assert [item["stable_task_id"] for item in selected] == ["13", "11"]


def test_missing_reference_identity_is_rejected():
    try:
        slice_matching([row(11)], [row(12)])
    except ValueError as error:
        assert "missing reference identities" in str(error)
    else:
        raise AssertionError("missing identity was accepted")


if __name__ == "__main__":
    test_subset_follows_reference_identity_order()
    test_missing_reference_identity_is_rejected()
    print("2 matching-slice tests passed")
