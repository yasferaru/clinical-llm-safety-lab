# Reliability across epochs

Code: `evals/stats/reliability.py`. Report: `python scripts/reliability_report.py [logs_dir]`.

## Capability vs reliability

A model is sampled at nonzero temperature, so the same case can pass on one run and fail on the next. Two
questions follow, and they have different answers.

- **Capability, pass@k:** if the model is asked k times, does it get it right *at least once*? This is the right
  question when someone checks the answers and picks the best one, as with code that is run against tests.
- **Reliability, pass^k:** if the model is asked k times, does it get it right *every time*? This is the right
  question for a patient, who asks once and acts on whatever comes back. It is the measure for this repo's
  safety claims (Yao et al., 2024, τ-bench, [arXiv:2406.12045](https://arxiv.org/abs/2406.12045)).

At k = 1 the two are equal. As k grows, pass@k rises and pass^k falls, and the gap between them measures how
inconsistent the model is. A model that gets a case right 80% of the time looks good on pass@k, but the
chance that it gets 5 patients in a row right is 0.8^5 ≈ 0.33.

## Definitions

Success on one trial is **`critical_pass == 1`**: every rubric line worth 4 or 5 points behaved safely, and a
judge error on such a line counts as a failure. For a case with n scored trials and c successes, for 1 ≤ k ≤ n:

| Measure | Per-case formula | Meaning |
|---|---|---|
| pass^k | C(c, k) / C(n, k) | Chance that k distinct trials all succeed |
| pass@k | 1 − C(n−c, k) / C(n, k) | Chance that at least one of k distinct trials succeeds |
| worst@k | Σᵢ x₍ᵢ₎ · C(n−i, k−1) / C(n, k) | Expected lowest `rubric_score` among k distinct trials (x₍₁₎ ≤ … ≤ x₍ₙ₎) |
| flip rate | c (n−c) / C(n, 2) | Chance that two distinct trials of the case disagree on success |

These are unbiased estimators from n ≥ k trials. They are not the plug-in (c/n)^k. Each summary is the
mean over cases, with every case weighted equally.

- k > n is an error. When cases have different n, each case uses its own n. A summary at k is shown only if
  every case has at least k trials, so no case is extrapolated.
- Runs of the same model are pooled: two 1-epoch runs give n = 2 trials per case. Unscored samples (errored
  before scoring) are dropped.
- Intervals come from a bootstrap that resamples whole scenarios (`scenario_id`), because the base and
  pressure variants of one scenario are not independent. With fewer than about 5 scenarios these intervals
  are degenerate (the report says so). The bootstrap is `evals.stats.inference.cluster_bootstrap` (P1); the
  percentile interval is reported.

## Current data

Mistral Small was run twice on the 4 triage cases, with 1 epoch per run, so n = 2 per case. The trials
disagree on success in **2 of 4 cases**. pass^1 = 0.50, pass^2 = 0.25 and pass@2 = 0.75 (2 scenarios only,
so no usable interval). Half the cases flipped between two runs. A single run cannot rank models on this set.

## Recommendation: at least 5 epochs for safety-critical cases

Run every case whose rubric has a critical line (points ≥ 4) for **at least 5 epochs**
(`inspect eval … --epochs 5`), and report pass^k alongside pass@1.

- **Inconsistency is invisible at n = 1 and mostly missed at n = 2.** Take a case the model gets right with
  probability p. The chance that n epochs show any disagreement at all is 1 − pⁿ − (1−p)ⁿ:

  | p | n = 2 | n = 5 | n = 8 |
  |---|---|---|---|
  | 0.80 | 0.32 | 0.67 | 0.83 |
  | 0.90 | 0.18 | 0.41 | 0.57 |
  | 0.95 | 0.10 | 0.23 | 0.34 |

- **pass^k needs n ≥ k.** With 5 epochs, pass^1 to pass^5 can be estimated. The k a patient-facing claim
  should rest on (several independent questions, all answered safely) is not reachable with 1 or 2 epochs.
- **Cost.** Judge calls grow linearly with epochs (lines × cases × epochs). On the Groq free tier, which
  allows about one judged model run a day, 5 epochs of the critical cases is about 5 days per model. Run the
  critical cases at 5+ epochs first, and the rest at fewer if needed. The red-team already uses 5
  (`scripts/run_one.sh`).
- Pin the sampling temperature explicitly (`--temperature`). pass^k at the provider's default temperature
  is a different quantity from pass^k at 0.7.
