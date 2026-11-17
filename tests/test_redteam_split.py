"""Dev / test split rule and the prompt-version guard."""
import json

import pytest

from redteam.prompt_guard import PromptGuardError, check, notes, prompt_sha, read_log, record
from redteam.run_redteam import load_cases, redteam
from redteam.split import _hash_split, assign_splits, select

CASES = load_cases()

# The assignment for the 8 current notes. Changing it means the split rule changed: do not.
PINNED = {"rt-001": "test", "rt-002": "dev", "rt-003": "dev", "rt-004": "test",
          "rt-005": "dev", "rt-006": "test", "rt-007": "test", "rt-008": "dev"}


def test_pinned_assignment():
    assert assign_splits(CASES) == PINNED


def test_pairs_are_split_and_stratified():
    a = assign_splits(CASES)
    by_type = {}
    for c in CASES:
        by_type.setdefault(c["attack_type"], []).append(a[c["id"]])
    for kind, splits in by_type.items():
        if len(splits) >= 2:
            assert set(splits) == {"dev", "test"}, kind
    assert sorted(a.values()).count("test") == 4


def test_adding_notes_never_moves_existing_ones():
    more = CASES + [{"id": "rt-009", "attack_type": "canary_prefix"}, {"id": "rt-010", "attack_type": "omission"},
                    {"id": "rt-011", "attack_type": "new_type"}]
    a = assign_splits(more)
    assert all(a[k] == v for k, v in PINNED.items())
    assert a["rt-009"] != a["rt-003"]          # completes the (rt-003, rt-009) pair
    assert a["rt-010"] != a["rt-006"]
    assert a["rt-011"] == _hash_split("rt-011")  # first of a new attack type: hash rule


def test_select():
    assert [c["id"] for c in select(CASES, "dev")] == ["rt-002", "rt-003", "rt-005", "rt-008"]
    assert len(select(CASES, "all")) == 8
    with pytest.raises(ValueError):
        select(CASES, "train")


# ---- guard ----
V1, V2 = "prompt text v1", "prompt text v2"


def test_dev_runs_are_always_allowed(tmp_path):
    log = tmp_path / "log.jsonl"
    record("p", V1, "dev", log)
    record("p", V2, "dev", log)
    assert [e["split"] for e in read_log(log)] == ["dev", "dev"]


def test_test_without_dev_run_raises(tmp_path):
    with pytest.raises(PromptGuardError, match="no dev run"):
        check("p", V1, "test", tmp_path / "log.jsonl")
    with pytest.raises(PromptGuardError):
        check("p", V1, "all", tmp_path / "log.jsonl")   # 'all' includes the test notes


def test_frozen_prompt_may_go_to_test(tmp_path):
    log = tmp_path / "log.jsonl"
    record("p", V1, "dev", log)
    entry = record("p", V1, "test", log)
    assert entry["sha256"] == prompt_sha(V1) and entry["forced"] is False
    record("p", V1, "test", log)                        # re-running the same version is fine


def test_edit_after_dev_run_raises(tmp_path):
    log = tmp_path / "log.jsonl"
    record("p", V1, "dev", log)
    with pytest.raises(PromptGuardError, match="edited after its last dev run"):
        record("p", V2, "test", log)
    assert len(read_log(log)) == 1                      # the refused run is not logged


def test_new_version_after_test_was_seen_raises(tmp_path):
    log = tmp_path / "log.jsonl"
    record("p", V1, "dev", log)
    record("p", V1, "test", log)
    record("p", V2, "dev", log)                         # tuning after seeing test results
    with pytest.raises(PromptGuardError, match="already evaluated with an earlier version"):
        record("p", V2, "test", log)


def test_force_bypasses_and_is_recorded(tmp_path):
    log = tmp_path / "log.jsonl"
    assert record("p", V2, "test", log, force=True)["forced"] is True


def test_note_entries_are_not_runs(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps({"event": "note", "prompt": "p", "sha256": prompt_sha(V1), "note": "caveat"}) + "\n")
    with pytest.raises(PromptGuardError, match="no dev run"):
        check("p", V1, "test", log)
    assert notes(log) == {"p": ["caveat"]}


def test_repo_log_has_caveats_for_existing_prompts():
    n = notes()
    assert "hardened_v1" in n and "not a held-out estimate" in n["hardened_v1"][0]


# ---- the Inspect task uses the split and the guard ----
def test_task_defaults_to_dev_with_five_repeats(tmp_path):
    t = redteam(prompt="baseline", prompt_log=str(tmp_path / "log.jsonl"))
    assert len(t.dataset) == 8 and t.epochs == 5          # 4 dev notes x (injected, clean)
    ids = {s.id.rsplit("-", 1)[0] for s in t.dataset}
    assert ids == {"rt-002", "rt-003", "rt-005", "rt-008"}
    assert all(s.metadata["prompt_sha"] for s in t.dataset)


def test_task_refuses_unfrozen_prompt_on_test(tmp_path):
    with pytest.raises(PromptGuardError):
        redteam(prompt="hardened_v1", split="test", prompt_log=str(tmp_path / "log.jsonl"))


def test_harm_judge_must_be_independent(tmp_path, monkeypatch):
    import inspect_ai.model

    from evals.judge_validation.independence import JudgeIndependenceError
    monkeypatch.setattr(inspect_ai.model, "get_model", lambda *a, **k: "ollama/qwen3:8b")
    with pytest.raises(JudgeIndependenceError):
        redteam(prompt="baseline", judge="ollama/qwen3:32b", prompt_log=str(tmp_path / "log.jsonl"))
