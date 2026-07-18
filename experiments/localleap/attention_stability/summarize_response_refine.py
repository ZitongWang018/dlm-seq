#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--output")
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in Path(args.trace).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise SystemExit("empty trace")
    diagnostics = [record["decode_diagnostics"] for record in records]
    if any(
        item.get("decoder")
        not in {"response_refine_v1", "response_refine_v2", "response_refine_v3"}
        for item in diagnostics
    ):
        raise SystemExit("trace contains a non-response-refine decoder")

    sample_records = []
    for record, item in zip(records, diagnostics):
        frontier = item["frontier"]
        repair = item["repair"]
        fill = item["fill"]
        retention = item.get("retention") or {}
        sample_records.append(
            {
                "task_id": record["task_id"],
                "prompt_hash": record["prompt_hash"],
                "nfe": record["nfe"],
                "fill_nfe": item["fill_nfe"],
                "repair_nfe": item["repair_nfe"],
                "selector_nfe": item.get("selector_nfe", 0),
                "residual_mask_count": item["residual_mask_count"],
                "remasked_positions": frontier["remasked_positions"],
                "selected_forced_commits": frontier["selected_forced_commits"],
                "selected_immature_commits": frontier[
                    "selected_immature_commits"
                ],
                "selected_response_invalidations": frontier[
                    "selected_response_invalidations"
                ],
                "mean_selected_incident_count": frontier[
                    "mean_incident_count_per_remasked_position"
                ],
                "fill_response_invalidations": fill["response_invalidations"],
                "repair_response_invalidations": repair[
                    "response_invalidations"
                ],
                "repair_response_validations": repair["response_validations"],
                "repair_revision_margin_candidates": repair[
                    "revision_margin_candidates"
                ],
                "revised_token_count": repair["revised_token_count"],
                "source_first_overrides": repair["source_first_overrides"],
                "accepted_blocks": retention.get("accepted_blocks", 0),
                "rejected_blocks": retention.get("rejected_blocks", 0),
                "accepted_positions": retention.get("accepted_positions", 0),
                "draft_to_retained_changes": item.get(
                    "draft_to_retained_changes", item["draft_to_repaired_changes"]
                ),
            }
        )

    def values(key):
        return [item[key] for item in sample_records]

    summary = {
        "schema_version": "response_refine_trace_summary_v2",
        "records": len(records),
        "unique_task_ids": len({item["task_id"] for item in sample_records}),
        "unique_prompt_hashes": len(
            {item["prompt_hash"] for item in sample_records}
        ),
        "nfe_min": min(values("nfe")),
        "nfe_max": max(values("nfe")),
        "residual_mask_count": sum(values("residual_mask_count")),
        "mean_remasked_positions": mean(values("remasked_positions")),
        "mean_selected_forced_commits": mean(values("selected_forced_commits")),
        "mean_selected_immature_commits": mean(
            values("selected_immature_commits")
        ),
        "mean_selected_response_invalidations": mean(
            values("selected_response_invalidations")
        ),
        "mean_selected_incident_count": mean(
            values("mean_selected_incident_count")
        ),
        "mean_fill_response_invalidations": mean(
            values("fill_response_invalidations")
        ),
        "mean_repair_response_invalidations": mean(
            values("repair_response_invalidations")
        ),
        "mean_repair_response_validations": mean(
            values("repair_response_validations")
        ),
        "mean_repair_revision_margin_candidates": mean(
            values("repair_revision_margin_candidates")
        ),
        "mean_revised_token_count": mean(values("revised_token_count")),
        "mean_source_first_overrides": mean(values("source_first_overrides")),
        "mean_selector_nfe": mean(values("selector_nfe")),
        "mean_accepted_blocks": mean(values("accepted_blocks")),
        "mean_rejected_blocks": mean(values("rejected_blocks")),
        "mean_accepted_positions": mean(values("accepted_positions")),
        "mean_draft_to_retained_changes": mean(
            values("draft_to_retained_changes")
        ),
        "samples": sample_records,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
