from validate_step_diagnostics import has_contiguous_nfe


def test_variable_length_trace_can_remain_contiguous():
    steps = [{"global_nfe": index} for index in range(1, 148)]
    assert has_contiguous_nfe(steps)


def test_nfe_gap_is_rejected():
    steps = [{"global_nfe": 1}, {"global_nfe": 3}]
    assert not has_contiguous_nfe(steps)


if __name__ == "__main__":
    test_variable_length_trace_can_remain_contiguous()
    test_nfe_gap_is_rejected()
    print("2 step-diagnostic validator tests passed")
