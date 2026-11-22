"""Does an uncertainty score predict critical_pass failure? AUROC with a cluster bootstrap, and risk-coverage.

EXPLORATORY. With a few dozen cases the intervals are wide; every report prints the interval first.

    failure   a case counts as a failure if its graded reply failed critical_pass in ANY epoch (critical_pass < 1
              averaged over epochs). The graded reply and the N sampled replies are different draws, so the score
              predicts how reliably the model handles the case, not whether that particular reply was bad.
    AUROC     P(score of a failing case > score of a passing case) + 0.5 * P(tie)   (Mann-Whitney U / (n_pos n_neg)).
              0.5 = no signal, 1 = every failure scores above every pass.
    interval  percentile interval from resampling whole scenarios (cases of one scenario are not independent).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from evals.abstention import aurc, risk_coverage
from evals.uncertainty.semantic_entropy import EXPLORATORY


def auroc(score, failure) -> float:
    """AUROC of ``score`` for ``failure`` (1 = failure), higher score = more uncertain.

        AUROC = (sum of the average ranks of failures - n_f (n_f + 1) / 2) / (n_f * n_p)
    Example: scores [.9, .8, .4, .3], failure [1, 1, 0, 0] -> 1.0 ; scores [.5, .5], failure [1, 0] -> 0.5 (tie).
    NaN if there is no failure or no pass.
    """
    s, y = np.asarray(score, float), np.asarray(failure, int)
    nf, npass = int(y.sum()), int((1 - y).sum())
    if nf == 0 or npass == 0:
        return math.nan
    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - nf * (nf + 1) / 2) / (nf * npass))


def _members(cluster):
    codes = pd.factorize(np.asarray(cluster))[0]
    return [np.flatnonzero(codes == i) for i in range(codes.max() + 1)]


def _draws(members, B, seed):
    rng = np.random.default_rng(seed)
    g = len(members)
    return [np.concatenate([members[i] for i in rng.integers(0, g, g)]) for _ in range(B)]


def auroc_bootstrap(score, failure, cluster, B: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """AUROC with a scenario-cluster bootstrap percentile interval (``numpy.random.default_rng(seed)``).

    Resamples whole clusters with replacement; resamples containing only failures or only passes are skipped and
    counted (``n_boot_valid`` of B are used). The interval is listed first among the keys on purpose.
    """
    s, y = np.asarray(score, float), np.asarray(failure, int)
    members = _members(cluster)
    vals = []
    for idx in _draws(members, B, seed) if len(members) >= 2 else []:
        a = auroc(s[idx], y[idx])
        if not math.isnan(a):
            vals.append(a)
    lo, hi = (np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if len(vals) >= 2
              else (math.nan, math.nan))
    return {"ci_low": float(lo), "ci_high": float(hi), "auroc": auroc(s, y), "n": len(s),
            "n_failures": int(y.sum()), "n_clusters": len(members), "n_boot_valid": len(vals), "B": B,
            "alpha": alpha, "label": EXPLORATORY}


def compare_predictors(scores: dict[str, np.ndarray], failure, cluster, B: int = 2000, seed: int = 0,
                       alpha: float = 0.05) -> dict:
    """AUROC (with cluster-bootstrap interval) for several scores, plus the difference from the first one.

    ``scores`` maps a name to its score array (first = reference, e.g. semantic entropy; second e.g. the
    log-probability baseline). All scores are resampled with the same clusters in each draw, so the interval for
    ``difference`` (reference minus other) is paired. Draws where any AUROC is undefined are skipped.
    """
    y = np.asarray(failure, int)
    names = list(scores)
    arr = {k: np.asarray(v, float) for k, v in scores.items()}
    members = _members(cluster)
    per = {k: [] for k in names}
    for idx in _draws(members, B, seed) if len(members) >= 2 else []:
        vals = {k: auroc(arr[k][idx], y[idx]) for k in names}
        if not any(math.isnan(v) for v in vals.values()):
            for k in names:
                per[k].append(vals[k])
    out: dict = {"label": EXPLORATORY, "n": len(y), "n_failures": int(y.sum()), "n_clusters": len(members),
                 "n_boot_valid": len(per[names[0]]), "scores": {}}
    for k in names:
        v = per[k]
        lo, hi = (np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if len(v) >= 2 else (math.nan,) * 2)
        out["scores"][k] = {"ci_low": float(lo), "ci_high": float(hi), "auroc": auroc(arr[k], y)}
    if len(names) > 1:
        ref = names[0]
        for k in names[1:]:
            d = np.array(per[ref]) - np.array(per[k])
            lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if len(d) >= 2 else (math.nan,) * 2
            out.setdefault("difference", {})[f"{ref} - {k}"] = {
                "ci_low": float(lo), "ci_high": float(hi), "auroc_difference": out["scores"][ref]["auroc"]
                - out["scores"][k]["auroc"]}
    return out


def uncertainty_risk_coverage(score, failure) -> pd.DataFrame:
    """Risk-coverage curve when the LEAST uncertain cases are answered first.

    Sort by score ascending (ties: failures first, pessimistic); risk at coverage k/n = failures among the first k / k.
    Implemented through evals.abstention.risk_coverage with confidence = 1 - min-max(score), which keeps order and
    ties. Example: scores [.1, .2, .3, .4], failure [0, 0, 1, 0] -> risks 0, 0, 1/3, 1/4.
    """
    s = np.asarray(score, float)
    span = s.max() - s.min()
    conf = 1 - (s - s.min()) / span if span > 0 else np.full(len(s), 0.5)
    return risk_coverage(conf, np.asarray(failure, float))


def selective_summary(score, failure, cluster=None, B: int = 1000, seed: int = 0) -> dict:
    """AURC (area under the risk-coverage curve, lower is better) and the always-answer baseline (the failure rate)."""
    s = np.asarray(score, float)
    span = s.max() - s.min()
    conf = 1 - (s - s.min()) / span if span > 0 else np.full(len(s), 0.5)
    return {**aurc(conf, np.asarray(failure, float), cluster, B=B, seed=seed), "label": EXPLORATORY}


def build_eval_table(scores: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """Join per-case uncertainty scores to graded outcomes (the P0 sample table, docs/results_schema.md).

    scores : case_id, semantic_entropy, logprob_score [, scenario_id, model]
    samples: case_id, critical_pass, scored [, epoch, model]
    One row per case: failure = 1 if mean critical_pass over the scored epochs < 1. Cases with no scored sample, or
    with NaN scores, are dropped (count in ``table.attrs["n_dropped"]``).
    """
    s = samples[samples["scored"].astype(bool)] if "scored" in samples else samples
    keys = ["case_id"] + (["model"] if "model" in scores and "model" in s else [])
    outcome = s.groupby(keys, as_index=False)["critical_pass"].mean()
    outcome["failure"] = (outcome["critical_pass"] < 1).astype(int)
    t = scores.merge(outcome[keys + ["failure"]], on=keys, how="inner")
    if "scenario_id" not in t:
        t["scenario_id"] = t["case_id"]
    ok = t[["semantic_entropy", "logprob_score"]].notna().all(axis=1)
    out = t[ok].reset_index(drop=True)
    out.attrs["n_dropped"] = int(len(scores) - len(out))
    return out
