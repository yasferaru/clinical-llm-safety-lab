import pytest

from evals.budget import BudgetPolicy, SpendLedger, estimate_usd
from evals.matrix import MatrixConfig, MatrixRun, mock_target
from scripts.run_live_matrix import artifact_snapshot, assert_reservation


def test_estimate_and_atomic_idempotent_reservation(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.json", BudgetPolicy(stage_limits={"smoke": 2}))
    assert estimate_usd(1_000_000, 1_000_000, 0.1, 0.4) == 0.5
    ledger.reserve("a", "smoke", 1.0, "example")
    ledger.reserve("a", "smoke", 1.0, "example")
    assert ledger.spent() == 1.0
    ledger.reconcile("a", 0.8, {"tokens": 9})
    assert ledger.spent() == 0.8


def test_stop_and_stage_thresholds(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.json", BudgetPolicy(stop_usd=20, ceiling_usd=25, stage_limits={"pilot": 5}))
    ledger.reserve("a", "pilot", 5, "p")
    with pytest.raises(RuntimeError, match="stage budget"):
        ledger.reserve("b", "pilot", 0.01, "p")
    ledger = SpendLedger(tmp_path / "other.json", BudgetPolicy(stop_usd=20, ceiling_usd=25))
    with pytest.raises(RuntimeError, match="stop threshold"):
        ledger.reserve("a", "production", 20.01, "p")


def test_reconcile_range_preserves_reserved_amount(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.json", BudgetPolicy(stage_limits={"smoke": 1}))
    ledger.reserve("a", "smoke", 0.5, "p")
    ledger.reconcile_range("a", 0.1, 0.2, {"provider": "usage"})
    entry = ledger.entries()[0]
    assert entry["reserved_usd"] == 0.5 and entry["actual_high_usd"] == 0.2


def test_transport_failure_remains_a_conservative_unconfirmed_reservation(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.json", BudgetPolicy(stage_limits={"smoke": 1}))
    ledger.reserve("probe", "smoke", 0.01, "google")
    ledger.mark_unconfirmed_transport("probe", 0.01, "ConnectError before HTTP")
    entry = ledger.entries()[0]
    assert entry["state"] == "unconfirmed_transport"
    assert entry["confirmed_usd"] == 0.0
    assert ledger.spent() == 0.01


def test_reconcile_preserves_reservation_and_microdollar_estimate(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.json", BudgetPolicy(stage_limits={"smoke": 1}))
    ledger.reserve("probe", "smoke", 0.01, "google")
    ledger.reconcile("probe", 0.0000015, {"totalTokenCount": 5})
    entry = ledger.entries()[0]
    assert entry["reserved_usd"] == 0.01 and entry["estimated_usd"] == 0.0000015
    assert ledger.spent() == 0.0000015


def test_reservation_guard_rejects_observed_cost_above_reservation(tmp_path):
    run = MatrixRun(MatrixConfig(targets=("google/gemini-3.5-flash-lite",), judges=("judge",),
                                 case_ids=("triage-001-base",), output=str(tmp_path / "run")))
    import asyncio
    asyncio.run(run.run_targets(mock_target))
    response = next((run.root / "responses").glob("*.json"))
    import json
    doc = json.loads(response.read_text())
    doc.update({"request_max_output_tokens": 1, "usage": {"promptTokenCount": 1_000_000, "candidatesTokenCount": 0}})
    response.write_text(json.dumps(doc))
    ledger = SpendLedger(tmp_path / "ledger.json", BudgetPolicy(stage_limits={"smoke": 1}))
    ledger.reserve("k", "smoke", 0.01, "google")
    with pytest.raises(RuntimeError, match="budget-reservation discrepancy"):
        assert_reservation(run, ledger, "k", {"google/gemini-3.5-flash-lite": {"input": 1, "output": 1}})


def test_reservation_guard_excludes_imported_artifacts_but_charges_new_artifact(tmp_path):
    """A v5 retry must not be charged again for imported v4 judgments."""
    run = MatrixRun(MatrixConfig(targets=("google/gemini-3.5-flash-lite",), judges=("judge",),
                                 case_ids=("triage-001-base",), output=str(tmp_path / "run")))
    import asyncio
    import json
    asyncio.run(run.run_targets(mock_target))
    response = next((run.root / "responses").glob("*.json"))
    doc = json.loads(response.read_text())
    doc.update({"request_max_output_tokens": 1, "usage": {"promptTokenCount": 1_000_000, "candidatesTokenCount": 0}})
    response.write_text(json.dumps(doc))
    baseline = artifact_snapshot(run)
    ledger = SpendLedger(tmp_path / "ledger.json", BudgetPolicy(stage_limits={"smoke": 1}))
    ledger.reserve("k", "smoke", 0.01, "google")
    prices = {"google/gemini-3.5-flash-lite": {"input": 1, "output": 1}}
    assert_reservation(run, ledger, "k", prices, baseline)
    doc["usage"] = {"promptTokenCount": 1_000_001, "candidatesTokenCount": 0}
    response.write_text(json.dumps(doc))
    with pytest.raises(RuntimeError, match="budget-reservation discrepancy"):
        assert_reservation(run, ledger, "k", prices, baseline)


def test_policy_revision_preserves_spend_entries(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.json", BudgetPolicy(stage_limits={"smoke": 1}))
    ledger.reserve("a", "smoke", 0.1, "p")
    ledger.record_policy_revision("smoke", 0.6, "owner approved")
    assert ledger.spent() == 0.1
    assert ledger._document()["policy_history"][-1]["limit_usd"] == 0.6
