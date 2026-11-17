# Judge bias audits, jury and drift

Code is in `evals/judge_validation/`. Every test uses mocked judges, so none calls a model.

## Independence (`independence.py`, `config/model_families.yaml`)

`assert_judge_independent(judge_model, models_under_test)` maps every model string to a family with the
regexes in `config/model_families.yaml`, after dropping the first provider segment
(`groq/openai/gpt-oss-120b` → `openai/gpt-oss-120b` → `openai`). It raises `JudgeIndependenceError` in two cases:

- the judge shares a family with any model under test;
- a model matches no family, or more than one.

It is called by `run_bias_audits` and `jury.check_jury_independent`.

**For P10:** wire it into `evals/run_eval.py`, e.g. at the top of `clinical_safety()`, with the model under
test from `inspect_ai.model.get_model()`. That file is not edited here (`CLAUDE.md`).

## Perturbation audits (`bias_audit.py`)

Each audit takes already-scored items (`items_from_logs(log_dir)`: one row per judged rubric line, with
conversation, reply, criterion and verdict). It changes the reply's **form only**, re-judges the same line,
and reports the verdict-change rate `(b + c) / n` as k/n with a 95% Wilson CI, plus an exact McNemar test
of b (YES→NO) against c (NO→YES).

| Audit | Change | Pair orientation |
|---|---|---|
| `length_pad` | Neutral openers and closers ("Thank you for your question.") | original → padded |
| `length_compress` | Remove filler sentences and extra whitespace | original → compressed |
| `format` | Markdown ↔ plain prose | **prose → Markdown**, whichever was the original |
| `order` | Permute sections, else bullet items, else paragraphs | original → reordered |

- `check_content` makes every perturbation raise if it adds or removes content words, apart from the
  padding and filler vocabulary.
- The original reply is re-judged alongside the perturbed one, and judged once more as a **control**. The
  control's change rate is the judge's own noise floor, so a perturbation effect should be read against it.
- Without the fixed orientation of the `format` audit, a judge that prefers Markdown flips in both
  directions and McNemar cancels out. This was found while testing, and the orientation fixes it.
- `compress` is rule-based and conservative: it only removes known filler. Real compression needs a rewrite
  model, plus a content check before it can be trusted.
- `order` has no natural direction, so read its change rate rather than its McNemar p.

**Self-preference** (`self_preference_audit`) compares one judge's verdicts on replies from its **own**
family with replies from each **other** family, matched on (case, rubric line), with McNemar on the
discordant pairs.

- With human labels, the outcome is a *false pass* (the judge passes a line the human fails). This removes
  real quality differences between the families.
- Without human labels, the raw pass rate is compared, and quality differences confound it.
- A significant own-family advantage raises `SelfPreferenceError` (fail loudly), and the result is attached
  to the error.
- Power: with 5 discordant pairs the smallest exact p is 0.0625, so at least about 6 pairs all in one
  direction are needed to flag at α = 0.05. This probe is the one place a judge may grade its own family.

## Jury (`jury.py`)

- **Majority vote** per rubric line, with unreadable votes ignored. A tie resolves to the **unsafe** side
  (`must` → not met, `must_not` → triggered), so a split jury never passes a line.
- **Fleiss' kappa** across judges, on items every judge rated readably. The number of dropped items is reported.
- **Per-judge agreement** with the human labels (agreement, Cohen's kappa, AC1), plus a row for the jury majority.
- **Leave-one-judge-out:** the mean `rubric_score` and `critical_pass` rate without each judge, the change
  against the full jury, and the number of majorities that flip. Scores come from `run_eval.combine`.
- **Annotation queue:** every split, tie or judge-error item is written to `queue_sheet.csv`. The sheet is
  blind (no votes, seeded shuffle, blank label column) and has a separate `queue_key.csv`.

## Drift (`drift.py`)

- `JudgeFingerprint(judge_model, judge_version, prompt_sha)`: `prompt_sha` is the sha256 of
  `run_eval.JUDGE_PROMPT`, and `judge_version` is the model name the provider reported.
  `inspect_line_judge(...).last_version` captures it.
- `record_judge_run` appends to `evals/judge_validation/judge_runs.jsonl`.
- `retest_agreement(run_a, run_b, fp_a, fp_b)` reports flips, agreement, kappa and AC1, with 95% bootstrap
  CIs that resample whole responses (seeded). It labels the comparison *test-retest* when the fingerprints
  match and *drift* when they differ.

## Agreement statistics: bugs found and fixed (`analyze_agreement.py`)

| Bug | Before | Now |
|---|---|---|
| Empty input | `ZeroDivisionError` in `agreement`, `cohen_kappa`, `gwet_ac1` | Returns `None` |
| Unequal-length rater lists | Silently wrong (`[T,F,T]` vs `[T,F]` gave kappa 0.40: `zip` truncated, n came from the first list) | `ValueError` |
| `None` among labels | `TypeError` from `sum` | Clear `ValueError` |
| Bootstrap unit | Resampled rubric lines, which are correlated within a reply, so the CI was too narrow | `groups=` resamples whole replies; the CLI clusters by reply |
| Bootstrap quantiles | Hand-indexed, slightly asymmetric | `numpy.quantile` |
| Output paths | Always wrote to `labelling/` whatever `--sheet` said, and crashed if that folder was missing | Writes next to the sheet, or to `--out` |

Checked and correct:

- Constant columns: kappa is `None` when both raters are constant on the same label (pe = 1), and 0 when
  one rater is constant.
- AC1 is defined for every n > 0.
- Tiny n gives the textbook values.

Added: **PABAK**, prevalence and bias indices, and a **kappa-paradox flag**. Example: 90 both-YES,
5 + 5 discordant and 0 both-NO gives po = 0.90, kappa = −0.05 and AC1 = 0.89. The report now prints a
warning to read AC1 and PABAK alongside kappa.
