from evals.dataset_audit import audit, load_yaml_cases


def test_current_dataset_audit_counts_and_pair_confounds():
    report = audit(load_yaml_cases())
    assert report["case_count"] == 54 and report["rubric_line_count"] == 264
    pairs = {tuple(f["case_ids"]) for f in report["findings"] if f["kind"] == "pair_rubric_change"}
    assert ("triage-001-base", "triage-001-pressure") in pairs
    assert ("triage-002-base", "triage-002-pressure") in pairs
    assert not [f for f in report["findings"] if f["kind"] in {"duplicate_id", "exact_conversation_duplicate"}]
