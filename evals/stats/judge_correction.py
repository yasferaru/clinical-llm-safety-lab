"""Correct a judge-labelled rate for the judge's own errors, and say how uncertain the corrected number is.

An LLM judge labels every rubric line (or every critical_pass) 0/1, but it makes mistakes. The raw judge rate is then a
biased estimate of the true rate, and a tight-looking interval around it is false precision. With a CALIBRATION set
(human label Y and judge label Yhat on the same items) and a larger set labelled by the judge only, we can correct.

References (cited as given by the repo owner, not re-checked here): Lee et al., "How to Correctly Report LLM-as-a-Judge
Evaluations" (arXiv:2511.21140); Chen et al. 2026 (arXiv:2601.05420); Angelopoulos et al., prediction-powered inference
(PPI) and PPI++.

Notation: f = judge label (0/1), Y = human label (0/1). Unlabeled set: N items with f only. Calibration set: n items
with (Y, f). theta = the true rate P(Y=1). Everything is cluster-robust, with ``scenario_id`` as the cluster unit,
using evals/stats/inference.py (clustered_se).

METHODS (``method=``)
  naive          mean of f over the unlabeled set. Biased unless the judge is perfect; shown for comparison only.
  human_only     mean of Y over the calibration set. Unbiased, but uses only the n human labels.
  rogan_gladen   theta = (p_hat + q0 - 1) / (q0 + q1 - 1), clipped to [0, 1], with p_hat = mean of f over the unlabeled
                 set, sensitivity q1 = P(f=1 | Y=1) and specificity q0 = P(f=0 | Y=0) estimated on the calibration set.
                 Needs J = q0 + q1 - 1 away from 0: if J < min_j (default 0.1) the judge is nearly uninformative and
                 theta is not identified, so we refuse (estimate NaN, warning), or raise if strict=True.
                 Intervals: ``ci="bootstrap"`` (default) resamples whole scenarios from both datasets jointly;
                 ``ci="delta"`` uses the delta method (below). Both propagate the uncertainty of p_hat and of q0, q1.
  ppi            theta = mean_unlabeled(f) + mean_calibration(Y - f); variance = V(mean_unlabeled f) + V(mean(Y - f)).
  ppi++          theta = lam * mean_unlabeled(f) + mean_calibration(Y - lam * f); variance = lam^2 V(mean_unlabeled f)
                 + V(mean_calibration(Y - lam * f)), lam = Cov(Y, f) / (Var(f) * (1 + n/N)) clipped to [0, 1],
                 with Cov from the calibration set and Var(f) from calibration and unlabeled f pooled (ddof=1).
                 lam = 0 gives human_only, lam = 1 gives ppi.
  V(.) is the cluster-robust variance of a mean, sum_c (sum_{i in c}(x_i - xbar))^2 / n^2.

DELTA METHOD for Rogan-Gladen. With J = q0 + q1 - 1 and theta = (p + q0 - 1)/J, the gradient with respect to
(p, q0, q1) is (1/J, (1-theta)/J, -theta/J), so
    Var(theta) = [V_p + (1-theta)^2 V_q0 + theta^2 V_q1 - 2 theta (1-theta) C_q0q1] / J^2,
where V_q1 and V_q0 are the cluster-robust variances of the two ratio estimators and C_q0q1 their covariance
(calibration clusters contain both Y=1 and Y=0 items). The delta method treats the unlabeled and calibration sets
as independent; if they share scenarios it ignores that covariance, which the joint bootstrap does not.

NO VALIDATION, NO CORRECTION. With no calibration labels, or fewer than MIN_CALIBRATION_N human labels in total or
MIN_PER_CLASS in either class, the result carries the banner "JUDGE NOT VALIDATED". With no calibration labels at
all the function returns the naive judge-only rate, labelled as uncorrected. See docs/judge_correction.md for how many
human labels a target interval needs (``required_calibration_size``).
"""
from __future__ import annotations

import math
import warnings as _warnings
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd

from evals.stats.inference import clustered_se

MIN_CALIBRATION_N = 30   # fewer human labels than this: banner (at 30, sensitivity from ~15 positives is +-0.15)
MIN_PER_CLASS = 10       # fewer human positives or negatives than this: banner
MIN_J = 0.1              # Rogan-Gladen refuses below this q0 + q1 - 1
MIN_CLUSTERS = 30        # fewer scenarios than this: warn that normal-theory intervals are optimistic
METHODS = ("rogan_gladen", "ppi", "ppi++", "naive", "human_only")
BANNER = "JUDGE NOT VALIDATED"

_N = NormalDist()


class NonIdentifiable(ValueError):
    """Raised (strict=True) when the judge is nearly uninformative: q0 + q1 - 1 < min_j."""


@dataclass
class CorrectedRate:
    """Result of corrected_rate. ``warnings`` are human-readable; ``validated`` is False when the banner applies."""
    estimate: float
    ci_low: float
    ci_high: float
    se: float
    method: str                 # method requested
    method_used: str            # method actually applied (naive if there were no calibration labels)
    ci_method: str
    alpha: float
    diagnostics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    validated: bool = True

    @property
    def banner(self) -> str:
        return (f"*** {BANNER}: {self.diagnostics.get('banner_reason', '')} ***" if not self.validated else "")

    def report(self) -> str:
        """Text for a report: the banner first (if any), then estimate, interval and the numbers behind it."""
        d = self.diagnostics
        out = [self.banner] if self.banner else []
        out.append(f"{self.method_used}: {_f(self.estimate)}  {int(round(100 * (1 - self.alpha)))}% CI "
                   f"[{_f(self.ci_low)}, {_f(self.ci_high)}] ({self.ci_method})")
        out.append(f"q1 (sensitivity) {_f(d.get('q1'))}, q0 (specificity) {_f(d.get('q0'))}, n_cal {d.get('n_cal')} "
                   f"({d.get('n_cal_pos')} human positives, {d.get('n_cal_neg')} negatives), "
                   f"N unlabeled {d.get('n_unlabeled')}")
        out += [f"warning: {w}" for w in self.warnings]
        return "\n".join(out)


def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------

def _binary(series: pd.Series, name: str) -> np.ndarray:
    v = pd.to_numeric(series.astype(float), errors="coerce").to_numpy(float)
    if not np.isin(v[~np.isnan(v)], (0.0, 1.0)).all():
        raise ValueError(f"column '{name}' must contain only 0/1 (or False/True)")
    return v


def _clean(df: pd.DataFrame, cols: dict[str, str], cluster_col: str):
    """Rows with all value columns present; returns (DataFrame with keys 'f','y'?, cluster str array, n_dropped)."""
    for c in list(cols.values()) + [cluster_col]:
        if c not in df.columns:
            raise ValueError(f"missing column '{c}'")
    vals = {k: _binary(df[c], c) for k, c in cols.items()}
    ok = ~np.any([np.isnan(v) for v in vals.values()], axis=0) & df[cluster_col].notna().to_numpy()
    return ({k: v[ok] for k, v in vals.items()}, df[cluster_col].astype(str).to_numpy()[ok], int((~ok).sum()))


def estimate_judge_error(human, judge) -> dict:
    """Sensitivity q1 = P(f=1 | Y=1) and specificity q0 = P(f=0 | Y=0) from paired 0/1 labels.

        q1 = sum(Y f) / sum(Y);  q0 = sum((1-Y)(1-f)) / sum(1-Y);  J = q0 + q1 - 1 (Youden's J)
    Example: Y = [1,1,1,1,0,0,0,0,0,0], f = [1,1,1,0,0,0,0,0,1,0] -> q1 = 3/4, q0 = 5/6, J = 7/12.
    A class with no human labels gives NaN for its rate.
    """
    y, f = np.asarray(human, float), np.asarray(judge, float)
    n1, n0 = float(y.sum()), float((1 - y).sum())
    q1 = float((y * f).sum() / n1) if n1 else math.nan
    q0 = float(((1 - y) * (1 - f)).sum() / n0) if n0 else math.nan
    return {"q1": q1, "q0": q0, "J": q0 + q1 - 1, "n_pos": int(n1), "n_neg": int(n0)}


def rogan_gladen(p_hat: float, q0: float, q1: float, min_j: float = MIN_J) -> tuple[float, float, float]:
    """Rogan-Gladen estimator: returns (theta clipped to [0, 1], J, unclipped theta).

        theta = (p_hat + q0 - 1) / (q0 + q1 - 1)
    NaN theta if J = q0 + q1 - 1 is below ``min_j`` (or either rate is NaN). Example: p_hat = 0.45, q0 = 0.85, q1 = 0.9
    -> J = 0.75, theta = 0.30 / 0.75 = 0.40; p_hat = 0.05 -> raw theta = -0.1333, clipped to 0.
    """
    j = q0 + q1 - 1
    if math.isnan(j) or j < min_j:
        return math.nan, j, math.nan
    raw = (p_hat + q0 - 1) / j
    return min(1.0, max(0.0, raw)), j, raw


# --------------------------------------------------------------------------------------------
# Cluster table for the bootstrap and the delta method
# --------------------------------------------------------------------------------------------

def _cluster_table(f_u, cl_u, y_c, f_c, cl_c) -> pd.DataFrame:
    """Per-scenario sums over the union of scenarios from both datasets.

    us, un: sum and count of f in the unlabeled set;  a, n1: sum of f and count among Y=1 calibration items;
    b, n0: count of f=0 and count among Y=0 calibration items. Every estimator below is a function of these sums.
    """
    u = pd.DataFrame({"c": cl_u, "us": f_u, "un": 1.0}).groupby("c").sum()
    c = pd.DataFrame({"c": cl_c, "a": f_c * y_c, "n1": y_c, "b": (1 - f_c) * (1 - y_c),
                      "n0": 1 - y_c}).groupby("c").sum()
    return u.join(c, how="outer").fillna(0.0)


def _delta_rogan_gladen(t: pd.DataFrame, min_j: float) -> dict:
    """Delta-method estimate, standard error and pieces for Rogan-Gladen from the cluster table."""
    un, n1, n0 = t["un"].sum(), t["n1"].sum(), t["n0"].sum()
    p, q1, q0 = t["us"].sum() / un, t["a"].sum() / n1, t["b"].sum() / n0
    theta, j, raw = rogan_gladen(p, q0, q1, min_j)
    if math.isnan(theta):
        return {"theta": math.nan, "se": math.nan, "J": j}
    ru, rp, rq = t["us"] - p * t["un"], t["a"] - q1 * t["n1"], t["b"] - q0 * t["n0"]
    v_p, v_q1, v_q0 = (ru ** 2).sum() / un ** 2, (rp ** 2).sum() / n1 ** 2, (rq ** 2).sum() / n0 ** 2
    c01 = (rp * rq).sum() / (n1 * n0)
    var = (v_p + (1 - raw) ** 2 * v_q0 + raw ** 2 * v_q1 - 2 * raw * (1 - raw) * c01) / j ** 2
    return {"theta": theta, "se": math.sqrt(max(var, 0.0)), "J": j, "raw": raw}


def _bootstrap_rogan_gladen(t: pd.DataFrame, B: int, seed: int, min_j: float) -> np.ndarray:
    """Rogan-Gladen estimates over B joint resamples of scenarios (``numpy.random.default_rng(seed)``).

    Each draw takes G scenarios with replacement from the union and keeps ALL their unlabeled and calibration rows, so
    sampling variability of p_hat, q1 and q0 enter together. Draws with an empty class or J < min_j are dropped
    (returned array is shorter than B; the caller reports how many).
    """
    m = t[["us", "un", "a", "n1", "b", "n0"]].to_numpy(float)
    g = len(m)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, g, size=(B, g))
    counts = np.zeros((B, g))
    np.add.at(counts, (np.arange(B)[:, None], draws), 1.0)
    s = counts @ m
    with np.errstate(divide="ignore", invalid="ignore"):
        p, q1, q0 = s[:, 0] / s[:, 1], s[:, 2] / s[:, 3], s[:, 4] / s[:, 5]
        j = q0 + q1 - 1
        theta = np.clip((p + q0 - 1) / j, 0.0, 1.0)
    ok = np.isfinite(theta) & np.isfinite(j) & (j >= min_j)
    return theta[ok]


# --------------------------------------------------------------------------------------------
# The estimators
# --------------------------------------------------------------------------------------------

def _ppi_lambda(y_c, f_c, f_u) -> float:
    """lam = Cov(Y, f) / (Var(f) (1 + n/N)), clipped to [0, 1].

    Cov comes from the calibration set, Var(f) from calibration and unlabeled f pooled.

    Example: calibration Y = [1,0,0,1], f = [1,0,1,1]; unlabeled f = [1,0,1,0]: Cov = 1/6, pooled Var(f) = 15/56,
    n/N = 1 -> lam = (1/6) / ((15/56) * 2) = 28/90 = 0.3111.
    """
    n, big_n = len(y_c), len(f_u)
    pooled = np.concatenate([f_c, f_u])
    var_f = float(pooled.var(ddof=1)) if len(pooled) > 1 else 0.0
    if var_f == 0 or n < 2:
        return 0.0
    cov = float(np.cov(y_c, f_c, ddof=1)[0, 1])
    return float(min(1.0, max(0.0, cov / (var_f * (1 + n / big_n)))))


def _ppi(f_u, cl_u, y_c, f_c, cl_c, lam: float) -> tuple[float, float]:
    """Estimate and SE of lam * mean(f_u) + mean(Y - lam f_c). lam = 1 is PPI; lam = 0 is human only.

    SE = sqrt( lam^2 V(f_u) + V(Y - lam f_c) ) with V the cluster-robust variance of a mean.
    """
    est = lam * f_u.mean() + (y_c - lam * f_c).mean()
    v_u = clustered_se(f_u, cl_u).se ** 2 if lam else 0.0
    v_c = clustered_se(y_c - lam * f_c, cl_c).se ** 2
    return float(est), math.sqrt(lam ** 2 * v_u + v_c)


def corrected_rate(df_unlabeled: pd.DataFrame, df_calibration: pd.DataFrame | None = None,
                   method: str = "rogan_gladen", ci: str = "bootstrap", judge_col: str = "judge",
                   human_col: str = "human", cluster_col: str = "scenario_id", alpha: float = 0.05, B: int = 2000,
                   seed: int = 0, min_j: float = MIN_J, strict: bool = False) -> CorrectedRate:
    """Judge-corrected rate of a binary outcome, with an interval and diagnostics.

    df_unlabeled   items labelled by the judge only: columns ``judge_col`` (0/1) and ``cluster_col``.
    df_calibration items with BOTH labels: ``human_col`` (0/1), ``judge_col``, ``cluster_col``.
    Use it at the rubric-line level (the judged_pass column of the line table, human labels per line) and for
    critical_pass (sample-level 0/1). Which outcome is the "positive" is up to the caller; q1 and q0 refer to it.

    method: see the module docstring. ci: "bootstrap" or "delta" (Rogan-Gladen only; PPI intervals are analytic).
    Returns CorrectedRate (estimate, ci_low, ci_high, se, diagnostics, warnings). Diagnostics: q1, q0, J, n_cal,
    n_cal_pos, n_cal_neg, n_unlabeled, clusters, judge_rate, human_rate, lambda, n_boot_valid.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if ci not in ("bootstrap", "delta"):
        raise ValueError("ci must be 'bootstrap' or 'delta'")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    uv, cl_u, drop_u = _clean(df_unlabeled, {"f": judge_col}, cluster_col)
    f_u = uv["f"]
    if len(f_u) == 0:
        raise ValueError("no judge-labelled items in df_unlabeled")
    z = _N.inv_cdf(1 - alpha / 2)
    notes: list[str] = []
    diag: dict = {"n_unlabeled": len(f_u), "clusters_unlabeled": len(set(cl_u)), "judge_rate": float(f_u.mean()),
                  "n_dropped_unlabeled": drop_u}
    if drop_u:
        notes.append(f"{drop_u} unlabeled rows dropped for missing values")
    no_cal = df_calibration is None or len(df_calibration) == 0
    if not no_cal:
        cv, cl_c, drop_c = _clean(df_calibration, {"y": human_col, "f": judge_col}, cluster_col)
        y_c, f_c = cv["y"], cv["f"]
        no_cal = len(y_c) == 0
    if no_cal:
        res = _mean_result(clustered_se(f_u, cl_u), "naive", method, "analytic", alpha, diag, notes, False)
        return _emit(res, "no calibration labels (human vs judge); the judge's error rate is unknown, so the number "
                          "below is the raw judge rate and is NOT corrected")

    err = estimate_judge_error(y_c, f_c)
    diag.update(q1=err["q1"], q0=err["q0"], J=err["J"], n_cal=len(y_c), n_cal_pos=err["n_pos"],
                n_cal_neg=err["n_neg"], clusters_calibration=len(set(cl_c)), human_rate=float(y_c.mean()),
                judge_rate_calibration=float(f_c.mean()), n_dropped_calibration=drop_c)
    if drop_c:
        notes.append(f"{drop_c} calibration rows dropped for missing values")
    small = (len(y_c) < MIN_CALIBRATION_N or err["n_pos"] < MIN_PER_CLASS or err["n_neg"] < MIN_PER_CLASS)
    reason = (f"only {len(y_c)} human labels ({err['n_pos']} positive, {err['n_neg']} negative); need at least "
              f"{MIN_CALIBRATION_N} in total and {MIN_PER_CLASS} per class before the judge's error rates mean anything"
              if small else "")
    if min(diag["clusters_unlabeled"], diag["clusters_calibration"]) < MIN_CLUSTERS:
        notes.append(f"fewer than {MIN_CLUSTERS} scenarios in a dataset; normal-theory and bootstrap intervals are "
                     "optimistic")

    if method == "naive":
        return _emit(_mean_result(clustered_se(f_u, cl_u), "naive", method, "analytic", alpha, diag, notes,
                                  not small), reason)
    if method == "human_only":
        return _emit(_mean_result(clustered_se(y_c, cl_c), "human_only", method, "analytic", alpha, diag, notes,
                                  not small), reason)
    if method in ("ppi", "ppi++"):
        lam = 1.0 if method == "ppi" else _ppi_lambda(y_c, f_c, f_u)
        est, se = _ppi(f_u, cl_u, y_c, f_c, cl_c, lam)
        diag["lambda"] = lam
        lo, hi = max(0.0, est - z * se), min(1.0, est + z * se)
        res = CorrectedRate(est, lo, hi, se, method, method, "analytic (cluster-robust)", alpha, diag, notes, not small)
        return _emit(res, reason)

    # Rogan-Gladen
    table = _cluster_table(f_u, cl_u, y_c, f_c, cl_c)
    delta = _delta_rogan_gladen(table, min_j)
    if math.isnan(delta["theta"]):
        msg = (f"judge nearly uninformative: q0 + q1 - 1 = {_f(delta['J'])} < {min_j}; the Rogan-Gladen correction is "
               "not identifiable and no corrected rate is reported")
        if strict:
            raise NonIdentifiable(msg)
        notes.append(msg)
        return _emit(CorrectedRate(math.nan, math.nan, math.nan, math.nan, method, method, ci, alpha, diag, notes,
                                   False), msg)
    if ci == "delta":
        se = delta["se"]
        lo, hi = max(0.0, delta["theta"] - z * se), min(1.0, delta["theta"] + z * se)
    else:
        boot = _bootstrap_rogan_gladen(table, B, seed, min_j) if len(table) >= 2 else np.array([])
        diag["n_boot_valid"] = len(boot)
        if len(boot) < 2:
            notes.append("bootstrap not possible (need at least 2 scenarios)")
            se, lo, hi = math.nan, math.nan, math.nan
        else:
            se = float(boot.std(ddof=1))
            lo, hi = (float(x) for x in np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
            if len(boot) < 0.95 * B:
                notes.append(f"{B - len(boot)} of {B} bootstrap draws were unusable (empty class or J < {min_j}); "
                             "the interval is conditional on an informative judge")
    if delta["raw"] < 0 or delta["raw"] > 1:
        notes.append(f"raw Rogan-Gladen value {delta['raw']:.3f} lies outside [0, 1] and was clipped")
    diag["raw_estimate"] = delta["raw"]
    return _emit(CorrectedRate(delta["theta"], lo, hi, se, method, method, f"{ci} (cluster)", alpha, diag, notes,
                               not small), reason)


def _mean_result(cse, used, method, ci_method, alpha, diag, notes, validated) -> CorrectedRate:
    """CorrectedRate for a plain (cluster-robust) mean: the naive and human-only methods."""
    z = _N.inv_cdf(1 - alpha / 2)
    est, se = cse.mean, cse.se
    lo, hi = (max(0.0, est - z * se), min(1.0, est + z * se)) if not math.isnan(se) else (math.nan, math.nan)
    return CorrectedRate(est, lo, hi, se, method, used, f"{ci_method} (cluster-robust)", alpha, diag, notes, validated)


def _emit(res: CorrectedRate, reason: str = "") -> CorrectedRate:
    """Store the banner reason on an unvalidated result and raise a UserWarning so the banner is seen."""
    if not res.validated:
        res.diagnostics["banner_reason"] = reason
        _warnings.warn(f"{BANNER}: {reason}", UserWarning, stacklevel=3)
    return res


# --------------------------------------------------------------------------------------------
# How many human labels?
# --------------------------------------------------------------------------------------------

def required_calibration_size(method: str, halfwidth: float, prevalence: float = 0.5, q1: float = 0.9,
                              q0: float = 0.9, n_unlabeled: float = math.inf, alpha: float = 0.05,
                              deff: float = 1.0) -> float:
    """Number n of human-labelled calibration items so that the (1-alpha) interval has half-width ``halfwidth``.

    With z = Phi^-1(1 - alpha/2), true rate theta = ``prevalence``, and ``deff`` the design effect from clustering
    (variances are multiplied by it):
      human_only   n = z^2 theta (1 - theta) deff / h^2
      ppi          Var(Y - f) = d - (theta - p)^2 with d = P(Y != f) = theta (1 - q1) + (1 - theta)(1 - q0) and
                   p = P(f = 1) = theta q1 + (1 - theta)(1 - q0);   n = deff Var(Y - f) / (h^2/z^2 - deff p(1-p)/N)
      rogan_gladen n = deff [ (1-theta)^2 q0 (1-q0) / (1-theta) + theta^2 q1 (1-q1) / theta ]
                           / (J^2 h^2/z^2 - deff p(1-p)/N),  J = q0 + q1 - 1
    (calibration items are assumed to have the same prevalence as the target; RG's two terms are the variances of
    q0 and q1 divided by their class sizes n(1-theta) and n theta; Cov(q0, q1) = 0 for unclustered items.)
    Returns inf if the unlabeled set alone already exceeds the target half-width (no n can reach it), and the
    unrounded value otherwise; take ceil() for a plan. PPI++ is never worse than ppi asymptotically, so ppi is a
    safe upper bound for it. Example: human_only, theta = 0.5, h = 0.05 -> 384.1 (so 385 labels).
    """
    z = _N.inv_cdf(1 - alpha / 2)
    th = prevalence
    big = (halfwidth / z) ** 2
    p = th * q1 + (1 - th) * (1 - q0)
    unl = deff * p * (1 - p) / n_unlabeled if math.isfinite(n_unlabeled) else 0.0
    if method == "human_only":
        return deff * th * (1 - th) / big
    if method == "ppi":
        d = th * (1 - q1) + (1 - th) * (1 - q0)
        num, den = deff * (d - (th - p) ** 2), big - unl
    elif method == "rogan_gladen":
        j = q0 + q1 - 1
        if j <= 0:
            return math.inf
        num = deff * ((1 - th) * q0 * (1 - q0) + th * q1 * (1 - q1))
        den = j ** 2 * big - unl
    else:
        raise ValueError("method must be 'human_only', 'ppi' or 'rogan_gladen'")
    return num / den if den > 0 else math.inf
