from pathlib import Path


ROOT = Path(__file__).parent
SCRIPTS = ROOT / "scripts" if (ROOT / "scripts").is_dir() else ROOT


def test_direct_v19_waits_for_v18_without_fair_gate():
    wrapper = (SCRIPTS / "run_sparse_context_repair_direct_after_v18.sh").read_text()
    assert "waiting_for_v18_queue=" in wrapper
    assert "waiting_for_fair_queue=" not in wrapper
    assert '[[ -e "${v18_queue}/ACCEPTED" ]]' in wrapper
    assert '[[ -e "${v18_queue}/REJECTED" ]]' in wrapper
    assert "sparse_context_repair_preregistration_20260720_v2.json" in wrapper


def test_full4_recovery_resumes_v18_but_not_provisional_fair():
    controller = (SCRIPTS / "run_full4_leakage_recovery_and_resume_v3.sh").read_text()
    assert "resume_v18()" in controller
    assert "run_localized_evidence_conflict_repair_queue.sh" in controller
    assert "run_fair_three_arm_reproduction_queue.sh" not in controller
    assert "SKIPPED_REDUNDANT_FAIR" in controller


def test_strict_v5_waits_for_direct_v19_queue():
    strict_name = "run_strict_unified_offline_three_arm_queue_v5.sh"
    if not (SCRIPTS / strict_name).exists():
        strict_name = "run_strict_unified_offline_three_arm_queue.sh"
    controller = (SCRIPTS / strict_name).read_text()
    assert "llada_slot_sparse_context_repair_v19_direct_v3" in controller
    assert "sparse_context_repair_direct_20260720_v3" in controller
    assert controller.index('if [[ -e "${v19_queue}/ACCEPTED" ]]') < controller.index(
        'elif [[ -e "${v18_queue}/ACCEPTED" ]]'
    )


if __name__ == "__main__":
    test_direct_v19_waits_for_v18_without_fair_gate()
    test_full4_recovery_resumes_v18_but_not_provisional_fair()
    test_strict_v5_waits_for_direct_v19_queue()
    print("direct rapid chain contract passed")
