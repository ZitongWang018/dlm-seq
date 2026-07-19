import json
import math
import tempfile
from pathlib import Path

from audit_partial_gsm8k_trace import (
    audit_records,
    extract_flexible_answer,
    is_correct,
    load_append_only_jsonl,
    normalize_exact_match,
    wilson_interval,
)


def row(index, generation, gold, *, nfe=128, residual=0, prompt_hash="p"):
    return {
        "absolute_index": index,
        "decoded_generation": generation,
        "raw_gold": gold,
        "prompt_hash": prompt_hash,
        "nfe": nfe,
        "residual_mask_count": residual,
        "correct": None,
    }


def test_extraction_matches_lm_eval_last_match():
    assert extract_flexible_answer("work 2 then answer $1,234.50") == "$1,234.50"
    assert extract_flexible_answer("answer is -12.") == "-12."
    assert extract_flexible_answer("no numeric answer") == "[invalid]"
    assert is_correct("we get $1,234.50.", "reasoning\n#### 1234.50")[0]
    assert is_correct("therefore -12.", "reasoning\n#### -12")[0]
    assert not is_correct("first 3 then 4", "reasoning\n#### 3")[0]
    assert normalize_exact_match("reasoning\n#### $1,000.") == "1000"


def test_health_summary_and_anomalies():
    records = [
        row(0, "answer 7", "work\n#### 7", nfe=129),
        row(1, "answer -2.5", "work\n#### -2.5", nfe=300),
        row(2, "missing", "work\n#### 9", nfe=140),
    ]
    summary = audit_records(records, expected_total=4)
    assert summary["pass"]
    assert summary["correct"] == 2
    assert summary["extraction_failures"] == 1
    assert summary["min_id"] == 0 and summary["max_id"] == 2
    assert summary["missing_prefix_ids"] == []
    assert summary["nfe_total"] == 569
    assert summary["nfe_min"] == 129 and summary["nfe_max"] == 300
    assert summary["missing_target_hash"] == 3
    assert summary["target_hash_phase"] == "post_generation_enrichment"

    broken = [
        records[0],
        records[2],
        row(2, "answer 1", "#### 1", nfe=float("nan"), residual=1, prompt_hash=""),
    ]
    broken[-1]["correct"] = True
    result = audit_records(broken, expected_total=3)
    assert not result["pass"]
    assert set(result["anomalies"]) == {
        "duplicate_absolute_ids",
        "missing_prefix_ids",
        "invalid_or_missing_nfe",
        "residual_masks",
        "missing_prompt_hash",
        "generation_trace_contains_non_null_correct",
    }


def test_jsonl_only_tolerates_incomplete_tail():
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "trace.jsonl"
        path.write_bytes((json.dumps(row(0, "7", "#### 7")) + "\n{\"absolute_index\":").encode())
        records, partial = load_append_only_jsonl(path)
        assert len(records) == 1 and partial == 1

        path.write_text("{bad}\n" + json.dumps(row(0, "7", "#### 7")) + "\n", encoding="utf-8")
        try:
            load_append_only_jsonl(path)
        except ValueError as error:
            assert "physical line 1" in str(error)
        else:
            raise AssertionError("malformed complete line was silently accepted")


def test_wilson_interval_is_finite_and_contains_rate():
    low, high = wilson_interval(453, 671)
    assert math.isfinite(low) and math.isfinite(high)
    assert low < 453 / 671 < high


if __name__ == "__main__":
    test_extraction_matches_lm_eval_last_match()
    test_health_summary_and_anomalies()
    test_jsonl_only_tolerates_incomplete_tail()
    test_wilson_interval_is_finite_and_contains_rate()
    print("4 partial GSM8K trace audit tests passed")
