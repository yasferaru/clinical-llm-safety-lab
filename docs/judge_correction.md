# Correcting judge-labelled rates

`evals/stats/judge_correction.py` turns a rate measured by an LLM judge into an estimate of the rate a human would
have measured, with an interval that includes the judge's own error. Without it, a judge that is wrong 10% of the
time gives a confident-looking number that is simply biased. References (cited as given, not re-checked): Lee et al.
(arXiv:2511.21140), Chen et al. 2026 (arXiv:2601.05420), Angelopoulos et al. (PPI, PPI++).

## What you need

| Set | Contents | Columns |
|---|---|---|
| Calibration | items with **both** a human label and the judge's label | `human`, `judge`, `scenario_id` |
| Unlabeled | the (larger) set with judge labels only | `judge`, `scenario_id` |

Both 0/1. Use it at the **rubric-line level** (judge column = `judged_pass`; human = the human's verdict on the same line)
or for **critical_pass** (0/1 per sample). Calibrate at the level you report: critical_pass errors are not the
line errors, because one wrong line can flip it. For critical_pass, derive the human value from the human line
labels with the same rule (`evals.run_eval.combine`).

```python
from evals.stats.judge_correction import corrected_rate
r = corrected_rate(df_unlabeled, df_calibration, method="rogan_gladen")   # or "ppi", "ppi++"
print(r.report())     # banner first when the judge is not validated
r.estimate, r.ci_low, r.ci_high, r.diagnostics, r.warnings
```

## Methods

| `method` | Estimate | Use when |
|---|---|---|
| `rogan_gladen` | (p_hat + q0 - 1) / (q0 + q1 - 1), clipped to [0, 1] | the judge's sensitivity q1 and specificity q0 are about the same for every item |
| `ppi` | mean_unlabeled(f) + mean_cal(Y - f) | the calibration set is a random sample of the same population as the unlabeled set |
| `ppi++` | lam * mean_unlabeled(f) + mean_cal(Y - lam f), lam = Cov(Y,f) / (Var(f)(1 + n/N)) | as `ppi`; never materially wider, often narrower |
| `human_only` | mean of Y over the calibration set | baseline: what the human labels alone give |
| `naive` | mean of f over the unlabeled set | comparison only; biased, and its interval is too narrow |

Rogan-Gladen intervals: `ci="bootstrap"` (default) resamples whole scenarios from both datasets together;
`ci="delta"` is the delta-method interval. In simulation (300 runs, 80 + 50 scenarios of 5 items, true rate 0.40,
q1 = 0.90, q0 = 0.85) they cover the true rate 93.7% (bootstrap) and 94.3% (delta); the raw judge rate's interval covered
21% of the time, the corrected one 95%. These are properties of that simulation, not guarantees. In a second simulation
(3000 judge-only items, 150 human labels) the mean 95% interval width was 0.156 for human-only, 0.117 for PPI and 0.107 for
PPI++, against 0.036 for the naive judge-only interval, which covered the true rate 0% of the time.

### When it refuses
If J = q0 + q1 - 1 is below 0.1 the judge is almost uninformative, the correction divides by a number near zero and the
rate is not identified. Rogan-Gladen then returns NaN with a warning (`strict=True` raises `NonIdentifiable`).

### The banner
`JUDGE NOT VALIDATED` appears (as `r.banner`, first in `r.report()`, and as a `UserWarning`) when there are no
calibration labels, or fewer than **30** human labels in total, or fewer than **10** human positives or negatives. These
are minimums for computing anything meaningful, not targets (see below). With no calibration labels the function
returns the raw judge rate, labelled `method_used="naive"`, and says it is not corrected.

## Assumptions to check before you trust the output

- **Calibration sample.** PPI needs a *random* sample of the target population. Rogan-Gladen only needs the judge's error
  rates *given the human label* to be the same in both sets, so it tolerates a calibration set stratified on the human
  label. It does **not** tolerate one stratified on the judge's label (typical when you pick "some judge-pass and some
  judge-fail"): that biases q1 and q0. If the labelling sample is stratified, say how.
- **Human labels are the truth.** If humans disagree among themselves (see `evals/judge_validation/`), the "corrected"
  rate is a rate relative to those humans.
- **Same judge, same prompt, same model-under-test mix** in both sets. A judge that is better on one model's replies than
  another's breaks the constant-error assumption; correct per model if you can afford the labels.
- **Few scenarios.** With fewer than 30 scenarios in either set the intervals are optimistic (a warning is added). Items from
  one scenario are not independent, so everything is cluster-robust on `scenario_id`.

## How many human labels?

`required_calibration_size(method, halfwidth, prevalence, q1, q0, n_unlabeled, deff)` solves for the number of
human-labelled items so that the 95% interval has the target half-width. Formulas are in its docstring; they assume
unclustered items (`deff = 1`) and a calibration set with the target prevalence. Numbers below come from that
function (rounded up; "N large" means the unlabeled set is big enough that its own sampling error is negligible; "never"
means a half-width that small is unreachable with that many unlabeled items).

| half-width | judge q1=q0 | prevalence | human only | PPI (N=5000) | PPI (N large) | Rogan-Gladen (N large) |
|---|---|---|---|---|---|---|
| 0.10 | 0.80 | 0.3 | 81 | 76 | 75 | 171 |
| 0.10 | 0.80 | 0.5 | 97 | 79 | 77 | 171 |
| 0.10 | 0.90 | 0.3 | 81 | 39 | 38 | 55 |
| 0.10 | 0.90 | 0.5 | 97 | 40 | 39 | 55 |
| 0.10 | 0.95 | 0.3 | 81 | 20 | 20 | 23 |
| 0.10 | 0.95 | 0.5 | 97 | 20 | 20 | 23 |
| 0.05 | 0.80 | 0.3 | 323 | 321 | 298 | 683 |
| 0.05 | 0.80 | 0.5 | 385 | 333 | 308 | 683 |
| 0.05 | 0.90 | 0.3 | 323 | 163 | 152 | 217 |
| 0.05 | 0.90 | 0.5 | 385 | 167 | 154 | 217 |
| 0.05 | 0.95 | 0.3 | 323 | 82 | 77 | 91 |
| 0.05 | 0.95 | 0.5 | 385 | 84 | 77 | 91 |

How to read it:

- A good judge (q = 0.9 or better) roughly halves the human labels needed under PPI. A mediocre one (q = 0.8) saves almost nothing.
- Rogan-Gladen needs *more* labels than PPI for the same width, because it divides by J, which inflates the
  uncertainty of q0 and q1 (factor 1/J^2). Prefer PPI/PPI++ when the calibration set is a random sample; use Rogan-Gladen
  when it is not.
- Clustering costs labels: with a design effect of 2 (at half-width 0.05, q = 0.9, prevalence 0.5) the need rises to 769
  (human only), 308 (PPI) and 433 (Rogan-Gladen). Estimate the design effect from `clustered_se` on pilot data.
- With the planned ~100 blind-labelled items (unclustered, prevalence 0.5, N large) the 95% half-width is about
  0.098 human-only, 0.062 under PPI with a q = 0.90 judge and 0.044 with q = 0.95 (h = 1.96 sqrt(Var / n), Var = 0.25, 0.10
  and 0.05). With clustering it is wider. Plan for the interval you can afford and report it.
