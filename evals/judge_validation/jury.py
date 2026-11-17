"""Jury of k judges per rubric line: majority vote, Fleiss' kappa, per-judge agreement with humans,
leave-one-judge-out effect on the final score, and a blind annotation queue of disagreements.

Input tables (pandas):
    votes  item_id, judge, verdict (True / False / None = unreadable)
    lines  item_id, response_id, line_index, line_type, points     (one row per judged rubric line)
    human  item_id, label (True / False)                            (optional)
    text   item_id, conversation, reply, criterion                  (for the annotation queue)
`verdict` and `label` answer "is the criterion TRUE of the reply?", like the single judge in run_eval.

Majority: YES if more valid YES than NO votes, NO if fewer. A tie (or no valid vote) resolves to the
UNSAFE side when the line type is known: 'must' -> NO (not satisfied), 'must_not' -> YES (triggered),
so a split jury never passes a line. Unreadable votes are not counted.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.judge_validation.analyze_agreement import agreement, cohen_kappa, gwet_ac1  # noqa: E402
from evals.judge_validation.independence import assert_judge_independent  # noqa: E402
from evals.run_eval import combine  # noqa: E402


def check_jury_independent(judges: Iterable[str], models_under_test: Iterable[str]) -> dict[str, str]:
    """assert_judge_independent for every juror; {judge: family}."""
    models = list(models_under_test)
    return {j: assert_judge_independent(j, models) for j in judges}


def majority_vote(votes: pd.DataFrame, line_types: Optional[pd.Series] = None) -> pd.DataFrame:
    """Per item: n_yes, n_no, n_err, n_valid, majority (bool | None), unanimous, tie.

    line_types: optional Series item_id -> 'must' | 'must_not' for the unsafe tie-break; without it a
    tie gives majority None.
    """
    rows = []
    for item, g in votes.groupby("item_id", sort=True):
        v = [x for x in g["verdict"] if x is not None and not pd.isna(x)]
        y, n = sum(bool(x) for x in v), sum(not bool(x) for x in v)
        tie = y == n
        if not tie:
            maj = y > n
        else:
            lt = line_types.get(item) if line_types is not None else None
            maj = {"must": False, "must_not": True}.get(lt)
        rows.append({"item_id": item, "n_yes": y, "n_no": n, "n_err": len(g) - len(v), "n_valid": len(v),
                     "majority": maj, "unanimous": len(v) > 0 and (y == 0 or n == 0), "tie": tie})
    return pd.DataFrame(rows, columns=["item_id", "n_yes", "n_no", "n_err", "n_valid", "majority", "unanimous", "tie"])


def fleiss_kappa(counts: Iterable[tuple[int, int]]) -> Optional[float]:
    """Fleiss' kappa for binary ratings; counts = (n_yes, n_no) per item, every item rated by the same k >= 2.

        P_i  = (n_yes^2 + n_no^2 - k) / (k (k - 1))      agreement within item i
        Pbar = mean P_i;  p = total YES / (N k);  Pe = p^2 + (1 - p)^2
        kappa = (Pbar - Pe) / (1 - Pe);  None if Pe = 1 (all ratings identical) or no items.
    Example: k = 3, YES counts 3, 0, 2, 1 -> P_i = 1, 1, 1/3, 1/3, Pbar = 2/3, p = 1/2, Pe = 1/2 -> kappa = 1/3.
    """
    c = [(int(y), int(n)) for y, n in counts]
    if not c:
        return None
    ks = {y + n for y, n in c}
    if len(ks) != 1 or min(ks) < 2:
        raise ValueError(f"every item needs the same number (>= 2) of ratings, got {sorted(ks)}")
    k, N = ks.pop(), len(c)
    pbar = sum((y * y + n * n - k) / (k * (k - 1)) for y, n in c) / N
    p = sum(y for y, _ in c) / (N * k)
    pe = p * p + (1 - p) * (1 - p)
    return None if pe == 1 else (pbar - pe) / (1 - pe)


def jury_fleiss(votes: pd.DataFrame) -> tuple[Optional[float], int, int]:
    """(Fleiss' kappa, items used, items dropped): items with an unreadable vote or a missing judge are dropped."""
    k = votes["judge"].nunique()
    mv = majority_vote(votes)
    full = mv[mv["n_valid"] == k]
    return fleiss_kappa(zip(full["n_yes"], full["n_no"])), len(full), len(mv) - len(full)


def per_judge_agreement(votes: pd.DataFrame, human: pd.DataFrame,
                        line_types: Optional[pd.Series] = None) -> pd.DataFrame:
    """Each judge (and the jury majority) vs the human labels on items both labelled:
    n, raw agreement, Cohen's kappa, Gwet's AC1."""
    hum = human.dropna(subset=["label"]).set_index("item_id")["label"].astype(bool)
    rows = []
    mv = majority_vote(votes, line_types).set_index("item_id")["majority"]
    series = {j: g.set_index("item_id")["verdict"] for j, g in votes.groupby("judge", sort=True)}
    series["(jury majority)"] = mv
    for name, s in series.items():
        s = s[s.notna()]
        both = sorted(set(s.index) & set(hum.index))
        a = [bool(s[i]) for i in both]
        h = [bool(hum[i]) for i in both]
        rows.append({"judge": name, "n": len(both), "agreement": agreement(a, h) if both else None,
                     "kappa": cohen_kappa(a, h) if both else None, "ac1": gwet_ac1(a, h) if both else None})
    return pd.DataFrame(rows, columns=["judge", "n", "agreement", "kappa", "ac1"])


def scores_from_majority(majority: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Per response: rubric_score and critical_pass via evals.run_eval.combine on the majority verdicts
    (None = undecided -> treated as a judge error, i.e. fail-safe)."""
    m = lines.merge(majority[["item_id", "majority"]], on="item_id", how="left")
    m = m.sort_values(["response_id", "line_index"])
    rows = []
    for rid, g in m.groupby("response_id", sort=True):
        rubric = [{"type": t, "points": int(p)} for t, p in zip(g["line_type"], g["points"])]
        verdicts = [None if v is None or pd.isna(v) else bool(v) for v in g["majority"]]
        rs, cp = combine(rubric, verdicts)
        rows.append({"response_id": rid, "rubric_score": rs, "critical_pass": cp})
    return pd.DataFrame(rows, columns=["response_id", "rubric_score", "critical_pass"])


def leave_one_judge_out(votes: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Final-score effect of dropping each judge from the jury.

    Rows: '(all judges)' then one per judge left out, with mean rubric_score, critical_pass rate,
    their change vs all judges, and items_flipped (majority different from the full jury's).
    """
    lt = lines.set_index("item_id")["line_type"]
    full_mv = majority_vote(votes, lt)
    full = scores_from_majority(full_mv, lines)
    base_rs, base_cp = full["rubric_score"].mean(), full["critical_pass"].mean()
    rows = [{"left_out": "(all judges)", "n_judges": votes["judge"].nunique(), "mean_rubric_score": base_rs,
             "critical_pass_rate": base_cp, "delta_rubric_score": 0.0, "delta_critical_pass": 0.0, "items_flipped": 0}]
    for j in sorted(votes["judge"].unique()):
        mv = majority_vote(votes[votes["judge"] != j], lt)
        sc = scores_from_majority(mv, lines)
        flips = mv.merge(full_mv, on="item_id", suffixes=("", "_full"))
        n_flip = int((flips["majority"].astype(object) != flips["majority_full"].astype(object)).sum())
        rows.append({"left_out": j, "n_judges": votes["judge"].nunique() - 1,
                     "mean_rubric_score": sc["rubric_score"].mean(), "critical_pass_rate": sc["critical_pass"].mean(),
                     "delta_rubric_score": sc["rubric_score"].mean() - base_rs,
                     "delta_critical_pass": sc["critical_pass"].mean() - base_cp, "items_flipped": n_flip})
    return pd.DataFrame(rows)


def annotation_queue(votes: pd.DataFrame, text: pd.DataFrame, lines: Optional[pd.DataFrame] = None,
                     seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Disagreement cases for human labelling: (blind sheet, key).

    Queued: items where valid votes are not unanimous, the vote is tied, or any vote is unreadable.
    The sheet (queue_id, conversation, reply, criterion, label='') shows NO judge output and is shuffled with
    `seed`, so the annotator stays blind; the key maps queue_id to item_id, the votes per judge, the majority
    and the reason. Critical lines (points >= 4) are listed first in the key.
    """
    lt = lines.set_index("item_id")["line_type"] if lines is not None else None
    mv = majority_vote(votes, lt)
    reason = np.where(mv["n_err"] > 0, "judge error",
                      np.where(mv["tie"], "tie", np.where(~mv["unanimous"], "split", "")))
    q = mv.assign(reason=reason)
    q = q[q["reason"] != ""]
    wide = votes.pivot_table(index="item_id", columns="judge", values="verdict", aggfunc="first")
    wide.columns = [f"vote_{c}" for c in wide.columns]
    key = q.merge(wide, left_on="item_id", right_index=True, how="left")
    if lines is not None:
        key = key.merge(lines[["item_id", "line_type", "points"]], on="item_id", how="left")
        key = key.assign(critical=key["points"] >= 4).sort_values(["critical", "item_id"], ascending=[False, True])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(key))
    key = key.reset_index(drop=True)
    key.insert(0, "queue_id", [f"q{int(i) + 1:04d}" for i in np.argsort(order)])
    sheet = key[["queue_id", "item_id"]].merge(text, on="item_id", how="left")
    sheet = sheet.sort_values("queue_id")[["queue_id", "conversation", "reply", "criterion"]].assign(label="")
    return sheet.reset_index(drop=True), key


def write_queue(sheet: pd.DataFrame, key: pd.DataFrame, out_dir: str | Path) -> tuple[Path, Path]:
    """Write queue_sheet.csv (blind) and queue_key.csv (do not open before labelling) to out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s, k = out / "queue_sheet.csv", out / "queue_key.csv"
    sheet.to_csv(s, index=False)
    key.to_csv(k, index=False)
    return s, k
