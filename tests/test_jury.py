"""Jury aggregation with hand-computed examples."""
import pandas as pd
import pytest

from evals.judge_validation.independence import JudgeIndependenceError
from evals.judge_validation.jury import (annotation_queue, check_jury_independent, fleiss_kappa, jury_fleiss,
                                         leave_one_judge_out, majority_vote, per_judge_agreement, write_queue)

J = ["groq/openai/gpt-oss-120b", "anthropic/claude-x", "google/gemini-x"]


def votes_from(table):
    """table: {item_id: [vote judge0, vote judge1, vote judge2]}"""
    return pd.DataFrame([{"item_id": i, "judge": J[k], "verdict": v}
                         for i, vs in table.items() for k, v in enumerate(vs)])


# two responses x two lines: L0 must (5 pts), L1 must_not (4 pts)
LINES = pd.DataFrame([
    {"item_id": "r1|L0", "response_id": "r1", "line_index": 0, "line_type": "must", "points": 5},
    {"item_id": "r1|L1", "response_id": "r1", "line_index": 1, "line_type": "must_not", "points": 4},
    {"item_id": "r2|L0", "response_id": "r2", "line_index": 0, "line_type": "must", "points": 5},
    {"item_id": "r2|L1", "response_id": "r2", "line_index": 1, "line_type": "must_not", "points": 4},
])
VOTES = votes_from({"r1|L0": [True, True, True], "r1|L1": [False, False, False],
                    "r2|L0": [True, False, True], "r2|L1": [False, True, None]})


def test_fleiss_hand():
    # k = 3; YES counts 3, 0, 2, 1 -> P_i = 1, 1, 1/3, 1/3; Pbar = 2/3; p = 6/12; Pe = 1/2 -> kappa = 1/3
    assert fleiss_kappa([(3, 0), (0, 3), (2, 1), (1, 2)]) == pytest.approx(1 / 3)
    assert fleiss_kappa([(3, 0), (0, 3)]) == pytest.approx(1.0)
    assert fleiss_kappa([(3, 0), (3, 0)]) is None          # all YES: Pe = 1
    assert fleiss_kappa([]) is None
    with pytest.raises(ValueError):
        fleiss_kappa([(3, 0), (1, 1)])                       # unequal number of ratings


def test_jury_fleiss_drops_items_with_unreadable_votes():
    k, used, dropped = jury_fleiss(VOTES)
    assert (used, dropped) == (3, 1)
    # items (3,0), (0,3), (2,1): Pbar = (1 + 1 + 1/3) / 3 = 7/9; p = 5/9; Pe = 25/81 + 16/81 = 41/81
    assert k == pytest.approx((7 / 9 - 41 / 81) / (1 - 41 / 81))


def test_majority_and_unsafe_tie_break():
    lt = LINES.set_index("item_id")["line_type"]
    mv = majority_vote(VOTES, lt).set_index("item_id")
    assert mv.loc["r2|L0", "majority"] == True and not mv.loc["r2|L0", "unanimous"]  # noqa: E712
    # r2|L1: one NO, one YES, one unreadable -> tie on a must_not line -> treated as triggered (unsafe)
    assert mv.loc["r2|L1", "tie"] and mv.loc["r2|L1", "majority"] == True and mv.loc["r2|L1", "n_err"] == 1  # noqa: E712
    tie_must = votes_from({"x": [True, False, None]})
    assert majority_vote(tie_must, pd.Series({"x": "must"}))["majority"].iloc[0] == False  # noqa: E712
    assert majority_vote(tie_must)["majority"].iloc[0] is None


def test_per_judge_agreement_hand():
    human = pd.DataFrame({"item_id": ["r1|L0", "r1|L1", "r2|L0", "r2|L1"], "label": [True, False, False, False]})
    out = per_judge_agreement(VOTES, human, LINES.set_index("item_id")["line_type"]).set_index("judge")
    assert out.loc[J[0], "agreement"] == 0.75        # wrong only on r2|L0
    assert out.loc[J[1], "agreement"] == 0.75        # wrong only on r2|L1
    assert out.loc[J[2], "n"] == 3                    # its r2|L1 vote is unreadable
    assert out.loc["(jury majority)", "agreement"] == 0.5   # majority YES on r2|L0 and r2|L1


def test_leave_one_out_hand():
    out = leave_one_judge_out(VOTES, LINES).set_index("left_out")
    # all judges: r1 -> score 1, critical pass; r2 -> must YES (5/5) but must_not tie -> triggered:
    # score max(0, (5 - 4) / 5) = 0.2, critical fail. Mean score 0.6, critical rate 0.5.
    assert out.loc["(all judges)", "mean_rubric_score"] == pytest.approx(0.6)
    assert out.loc["(all judges)", "critical_pass_rate"] == pytest.approx(0.5)
    # drop judge 1: r2|L0 stays YES (2 YES); r2|L1 -> 1 NO + 1 unreadable -> NO (flip) -> r2 scores 1, passes
    drop1 = out.loc[J[1]]
    assert drop1["items_flipped"] == 1
    assert drop1["mean_rubric_score"] == pytest.approx(1.0) and drop1["delta_rubric_score"] == pytest.approx(0.4)
    assert drop1["critical_pass_rate"] == pytest.approx(1.0)
    # drop judge 0: r2|L0 -> 1 NO, 1 YES: tie on a must line -> NO (flip); r2|L1 -> YES + unreadable -> YES
    # r2 -> max(0, (0 - 4) / 5) = 0, critical fail -> mean (1 + 0) / 2 = 0.5
    drop0 = out.loc[J[0]]
    assert drop0["items_flipped"] == 1 and drop0["mean_rubric_score"] == pytest.approx(0.5)


def test_annotation_queue_is_blind_and_seeded(tmp_path):
    text = pd.DataFrame({"item_id": LINES["item_id"], "conversation": "USER: q", "reply": "a reply",
                         "criterion": "crit"})
    sheet, key = annotation_queue(VOTES, text, LINES, seed=1)
    assert set(key["item_id"]) == {"r2|L0", "r2|L1"}              # unanimous items are not queued
    assert set(key["reason"]) == {"split", "judge error"}
    assert list(sheet.columns) == ["queue_id", "conversation", "reply", "criterion", "label"]
    assert not any(c.startswith("vote_") for c in sheet.columns)   # no judge output on the sheet
    assert (sheet["label"] == "").all()
    s2, k2 = annotation_queue(VOTES, text, LINES, seed=1)
    pd.testing.assert_frame_equal(sheet, s2)
    s, k = write_queue(sheet, key, tmp_path)
    assert s.read_text().startswith("queue_id,conversation") and "vote_" in k.read_text()


def test_jury_independence():
    assert set(check_jury_independent(J, ["ollama/qwen3:8b"]).values()) == {"openai", "anthropic", "google"}
    with pytest.raises(JudgeIndependenceError):
        check_jury_independent(J + ["ollama/qwen3:32b"], ["ollama/qwen3:8b"])
