"""Split-conformal abstention on the semantic-entropy score (EXPLORATORY).

Procedure. Split the cases by scenario into a calibration part and a test part. On the calibration part, keep the
cases the model handled correctly (failure = 0) and take their uncertainty scores as nonconformity scores. With
alpha chosen, the threshold is
    q_hat = the k-th smallest calibration score,  k = ceil((n_cal + 1)(1 - alpha))
    (evals.abstention.conformal_quantile)
and the rule is: answer when score <= q_hat, otherwise abstain (ask or defer).

Guarantee (marginal, if calibration and test cases are exchangeable): a new case that the model would handle
correctly is answered with probability >= 1 - alpha. It says nothing about failing cases; ``failures_caught`` and
``risk_answered`` report that empirically. Exchangeability is a real assumption: cases of one scenario are not
exchangeable, so the split is by scenario, and a case set of ~40 is too small for tight statements.

Guard: with n_cal < 1/alpha we refuse with a clear message. (Below about 1/alpha - 1 the threshold is infinite, i.e.
never abstain; just above that it is the largest calibration score. Both are meaningless as a calibration, so the
guard is deliberately a little stricter than strictly necessary.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from evals.abstention import conformal_quantile
from evals.uncertainty.semantic_entropy import EXPLORATORY


class CalibrationTooSmall(ValueError):
    """Raised when fewer than 1/alpha correct calibration cases are available."""


def split_by_scenario(table: pd.DataFrame, cal_fraction: float = 0.5,
                      seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random split of whole scenarios into (calibration, test) (``numpy.random.default_rng(seed)``).

    Each scenario goes entirely to one side, so related cases never straddle the split. Needs >= 2 scenarios.
    """
    if not 0 < cal_fraction < 1:
        raise ValueError("cal_fraction must be in (0, 1)")
    ids = np.array(sorted(table["scenario_id"].astype(str).unique()))
    if len(ids) < 2:
        raise ValueError("need at least 2 scenarios to split")
    rng = np.random.default_rng(seed)
    n_cal = min(len(ids) - 1, max(1, round(cal_fraction * len(ids))))
    cal_ids = set(rng.permutation(ids)[:n_cal])
    is_cal = table["scenario_id"].astype(str).isin(cal_ids)
    return table[is_cal].reset_index(drop=True), table[~is_cal].reset_index(drop=True)


def calibrate(cal_scores, cal_failure, alpha: float = 0.1) -> float:
    """q_hat from the scores of the correctly handled calibration cases; refuses if n_cal < 1/alpha.

    n_cal counts only the cases with failure = 0 (the ones that define "correct"). Example: alpha = 0.1 needs at
    least 10 such cases; with exactly 10, k = ceil(11 * 0.9) = 10, so q_hat is the largest of them.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    s, y = np.asarray(cal_scores, float), np.asarray(cal_failure, int)
    good = s[y == 0]
    needed = math.ceil(round(1 / alpha, 9))
    if len(good) < needed:
        raise CalibrationTooSmall(
            f"calibration set too small: {len(good)} correctly handled calibration cases, but alpha={alpha} needs at "
            f"least 1/alpha = {needed}. Use a larger alpha, more cases, or a bigger calibration share.")
    return conformal_quantile(good, alpha)


@dataclass
class ConformalResult:
    alpha: float
    q_hat: float
    n_cal: int                 # correct calibration cases used
    n_test: int
    coverage: float            # share of correctly handled test cases answered (target >= 1 - alpha)
    n_test_correct: int
    abstention_rate: float     # share of ALL test cases abstained
    failures_caught: float     # share of failing test cases abstained (NaN if none fail)
    risk_answered: float       # failure rate among answered test cases (NaN if none answered)
    risk_all: float            # failure rate over all test cases (the no-abstention baseline)
    label: str = EXPLORATORY


def conformal_abstention(cal: pd.DataFrame, test: pd.DataFrame, score_col: str = "semantic_entropy",
                         alpha: float = 0.1) -> ConformalResult:
    """Calibrate on ``cal``, then apply "answer iff score <= q_hat" to ``test`` and report the empirical numbers.

        coverage         = #(test cases with failure=0 and score <= q_hat) / #(test cases with failure=0)
        abstention_rate  = #(score > q_hat) / #test
        failures_caught  = #(failure=1 and score > q_hat) / #(failure=1)
        risk_answered    = #(failure=1 and score <= q_hat) / #(score <= q_hat)
    One split gives a noisy coverage figure (a handful of test cases); the guarantee is an average over splits.
    """
    q = calibrate(cal[score_col], cal["failure"], alpha)
    s, y = test[score_col].to_numpy(float), test["failure"].to_numpy(int)
    answered = s <= q
    n_correct, n_fail, n_ans = int((y == 0).sum()), int((y == 1).sum()), int(answered.sum())

    def ratio(a, b):
        return a / b if b else math.nan
    return ConformalResult(alpha, q, int((cal["failure"] == 0).sum()), len(test),
                           ratio(int((answered & (y == 0)).sum()), n_correct), n_correct,
                           ratio(int((~answered).sum()), len(test)), ratio(int((~answered & (y == 1)).sum()), n_fail),
                           ratio(int((answered & (y == 1)).sum()), n_ans), ratio(n_fail, len(test)))
