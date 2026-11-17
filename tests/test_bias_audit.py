"""Bias audits with mocked judges: perturbations keep content, fair judges are stable, biased ones are caught."""
import asyncio

import numpy as np
import pandas as pd
import pytest

from evals.judge_validation.bias_audit import (ITEM_COLUMNS, ContentChangedError, SelfPreferenceError,
                                               check_content, compress, content_words, format_swap,
                                               has_markdown, pad, perturbation_audit, reorder, results_markdown,
                                               run_bias_audits, self_preference_audit, summarise_pairs,
                                               to_markdown, to_prose)
from evals.judge_validation.independence import JudgeIndependenceError
from evals.stats.paired import mcnemar_exact

REPLIES = [
    "Go to the emergency department now. Repeated vomiting after a head injury is a danger sign.\n\n"
    "Do not wait for the pediatrician.",
    "## Urgency\nCall 911 or go to the ER immediately.\n\n## Why\n- Vomiting after a fall can mean a brain injury.\n"
    "- It can get worse quickly.",
    "It can probably wait. Book the next GP appointment.\n\nIf hearing does not return, see a doctor.",
    "- See an ENT specialist today or tomorrow.\n- Sudden hearing loss is treated with steroids.\n"
    "- Delayed treatment is less likely to restore hearing.",
]


def make_items(n=40, seed=0, model="ollama/qwen3:8b"):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        reply = REPLIES[i % len(REPLIES)]
        rows.append({"item_id": f"i{i}", "response_id": f"r{i // 2}", "case_id": f"c{i % 8}", "model": model,
                     "line_type": "must", "points": 5, "criterion": "Tells the user to go to the ER now.",
                     "conversation": "USER: help", "reply": reply, "verdict": bool(rng.random() < 0.5)})
    return pd.DataFrame(rows, columns=ITEM_COLUMNS)


def judge_from(rule):
    async def judge(conversation, reply, criterion):
        return rule(reply)
    return judge


def content_rule(reply):
    """A fair judge: decides on content words only."""
    w = content_words(reply)
    return bool({"emergency", "er", "911", "ent"} & w)


fair = judge_from(content_rule)
verbose_lover = judge_from(lambda r: content_rule(r) or len(r) > 200)
markdown_lover = judge_from(lambda r: content_rule(r) and has_markdown(r))
first_line_reader = judge_from(lambda r: content_rule(r.strip().splitlines()[0]))


def run(coro):
    return asyncio.run(coro)


# ---- McNemar ----
def test_mcnemar_hand():
    assert mcnemar_exact(0, 5) == pytest.approx(0.0625)          # 2 * 0.5^5
    assert mcnemar_exact(1, 7) == pytest.approx(0.0703125)       # 2 * (1 + 8) / 256
    assert mcnemar_exact(0, 0) == 1.0
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


# ---- perturbations keep content ----
@pytest.mark.parametrize("reply", REPLIES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_perturbations_preserve_content_words(reply, seed):
    assert content_words(format_swap(reply, seed)) == content_words(reply)
    assert content_words(reorder(reply, seed)) == content_words(reply)
    assert content_words(compress(pad(reply, seed), seed)) == content_words(reply)  # compress undoes pad
    assert len(pad(reply, seed)) > len(reply)


def test_check_content_rejects_changes():
    with pytest.raises(ContentChangedError):
        check_content("Go to the ER now.", "Go to the ER tomorrow.")
    check_content("Go now.", "Thank you for reading. Go now.", ["Thank you for reading."])


def test_format_swap_hand():
    assert to_prose("## Plan\n- **Go** to the ER\n- Call 911") == "Plan. Go to the ER. Call 911."
    assert to_markdown("Go now. Call 911.") == "- Go now.\n- Call 911."


def test_reorder_changes_order_only():
    r = REPLIES[3]
    out = reorder(r, 0)
    assert out != r and sorted(out.splitlines()) == sorted(r.splitlines())
    assert reorder("One paragraph only.", 0) == "One paragraph only."   # nothing independent to move


def test_compress_hand():
    assert compress("Great question!  Go to the ER now.\n\n\n\nI hope this helps.") == "Go to the ER now."


# ---- audits ----
def test_fair_judge_does_not_change(monkeypatch):
    items = make_items()
    for name in ("length_pad", "format", "order"):
        res, det = run(perturbation_audit(items, fair, name))
        assert res.yes_to_no == res.no_to_yes == 0 and res.change_rate == 0.0, name
        assert res.p_mcnemar == 1.0 and res.control_changes == 0


def test_verbosity_bias_is_detected():
    res, _ = run(perturbation_audit(make_items(), verbose_lover, "length_pad"))
    # the 'can probably wait' reply (10 of 40 items) turns YES only when padded past 200 characters
    assert (res.yes_to_no, res.no_to_yes, res.n_pairs) == (0, 10, 40)
    assert res.p_mcnemar == pytest.approx(2 * 0.5 ** 10)
    assert res.change_ci[0] > 0.1


def test_format_and_order_bias_are_detected():
    fmt, _ = run(perturbation_audit(make_items(), markdown_lover, "format"))
    # oriented prose -> markdown: content-positive items (3 of the 4 reply templates x 10) go NO -> YES,
    # never YES -> NO
    assert fmt.yes_to_no == 0 and fmt.no_to_yes == 30 and fmt.p_mcnemar < 1e-5
    assert fmt.control_changes == 0
    order, _ = run(perturbation_audit(make_items(), first_line_reader, "order"))
    assert order.yes_to_no + order.no_to_yes > 0


def test_unreadable_verdicts_and_unchanged_replies_are_counted():
    calls = {"n": 0}

    async def flaky(conversation, reply, criterion):
        calls["n"] += 1
        return None if calls["n"] % 3 == 0 else content_rule(reply)
    items = make_items(8)
    items.loc[0, "reply"] = "Single paragraph."         # order perturbation cannot change it
    res, det = run(perturbation_audit(items, flaky, "order"))
    assert res.n_skipped >= 1
    assert res.n_judge_errors == det["changed"].isna().sum() and res.n_pairs + res.n_judge_errors == len(det)


def test_stored_verdict_mode_uses_the_verdict_column():
    items = make_items(4).assign(verdict=False)
    res, _ = run(perturbation_audit(items, fair, "length_pad", rejudge_original=False, control=False))
    assert res.no_to_yes == sum(content_rule(r) for r in items["reply"]) and res.control_pairs is None


def test_summarise_pairs_hand():
    r = summarise_pairs("x", [True, True, False, None], [False, True, True, True], 4, 1)
    assert (r.n_pairs, r.yes_to_no, r.no_to_yes, r.n_judge_errors) == (3, 1, 1, 1)
    assert r.change_rate == pytest.approx(2 / 3) and r.p_mcnemar == 1.0


def test_run_bias_audits_enforces_independence():
    with pytest.raises(JudgeIndependenceError):
        run(run_bias_audits(make_items(), fair, "ollama/qwen3:14b"))       # qwen judging qwen
    summary, details = run(run_bias_audits(make_items(), fair, "groq/openai/gpt-oss-120b"))
    assert list(summary["name"]) == ["length_pad", "length_compress", "format", "order"]
    md = results_markdown(summary)
    assert "95% Wilson" in md and "length_pad" in md


# ---- self-preference ----
def sp_table(lenient_own: bool, human: bool):
    """20 cases x 1 line; own family = openai (gpt-oss), other = mistral. Truth: half the replies are bad."""
    rows = []
    for c in range(20):
        bad = c % 2 == 0
        for model in ("groq/openai/gpt-oss-20b", "mistral/mistral-small-latest"):
            own = "openai" in model
            verdict = (not bad) or (lenient_own and own)       # 'must' line: True = judged satisfied
            row = {"case_id": f"c{c}", "line_index": 0, "line_type": "must", "model": model, "verdict": verdict}
            if human:
                row["human_label"] = not bad
            rows.append(row)
    return pd.DataFrame(rows)


def test_self_preference_detected_and_raises():
    with pytest.raises(SelfPreferenceError) as e:
        self_preference_audit(sp_table(True, human=True), "groq/openai/gpt-oss-120b")
    r = e.value.result[0]
    # 10 bad cases passed for own family only: b = 10, c = 0, p = 2 * 0.5^10
    assert (r["own_only"], r["other_only"], r["outcome"]) == (10, 0, "false_pass")
    assert r["p_mcnemar"] == pytest.approx(2 * 0.5 ** 10)


def test_five_pairs_can_never_reach_significance():
    # the smallest possible exact McNemar p with 5 discordant pairs is 2 * 0.5^5 = 0.0625 > 0.05
    t = sp_table(True, human=True)
    out = self_preference_audit(t[t["case_id"].isin([f"c{i}" for i in range(10)])], "groq/openai/gpt-oss-120b")
    assert out.iloc[0]["own_only"] == 5 and not out.iloc[0]["flagged"]


def test_self_preference_threshold_and_non_strict():
    out = self_preference_audit(sp_table(True, human=False), "groq/openai/gpt-oss-120b", alpha=0.1, strict=False)
    assert bool(out.iloc[0]["flagged"]) is True and out.iloc[0]["outcome"] == "judge_pass"


def test_fair_judge_not_flagged():
    out = self_preference_audit(sp_table(False, human=True), "groq/openai/gpt-oss-120b")
    assert out.iloc[0]["own_only"] == out.iloc[0]["other_only"] == 0 and not out.iloc[0]["flagged"]


def test_self_preference_needs_own_family_replies():
    t = sp_table(False, human=True)
    with pytest.raises(JudgeIndependenceError, match="own family"):
        self_preference_audit(t[t["model"].str.startswith("mistral")], "groq/openai/gpt-oss-120b")
