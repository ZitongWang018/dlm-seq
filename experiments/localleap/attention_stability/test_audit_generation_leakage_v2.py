#!/usr/bin/env python3
from audit_generation_leakage_v2 import audit_trace_row


def main():
    allowed = {
        "raw_gold": "answer",
        "normalized_gold": "answer",
        "correct": None,
        "decoded_generation": "candidate",
        "decode_diagnostics": {
            "selected_name": "accuracy",
            "public_guard": {
                "uses_hidden_tests": False,
                "uses_reference_solution": False,
            },
        },
    }
    assert audit_trace_row(allowed, 1) == []
    nested = dict(allowed)
    nested["decode_diagnostics"] = {"raw_gold": "leak"}
    assert any(x["reason"] == "forbidden_selector_key" for x in audit_trace_row(nested, 2))
    scored = dict(allowed)
    scored["correct"] = True
    assert any(x["reason"] == "post_generation_correct_must_be_none" for x in audit_trace_row(scored, 3))
    hidden = dict(allowed)
    hidden["decode_diagnostics"] = {"uses_hidden_tests": True}
    assert any(x["reason"] == "must_be_false" for x in audit_trace_row(hidden, 4))
    print("generation leakage v2 tests passed")


if __name__ == "__main__":
    main()
