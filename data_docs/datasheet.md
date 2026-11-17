# Datasheet: Clinical LLM Safety Lab case set (v0, draft)

Format follows Gebru et al., "Datasheets for Datasets" (2021). Written during construction; sections marked
**TO COMPLETE** need real run results or decisions and are left open on purpose.

## Motivation
- **Purpose.** Measure, case by case, how language models fail on clinical safety behaviours: recognising
  emergencies, dosing and interaction advice, asking for missing context, and keeping a correct position under
  patient pressure.
- **Author.** Yasmyna Feraru (medical student). Built as an independent project.
- **Funding.** None.

## Composition
- **Records.** One record per conversation. A clinical scenario is written as a cluster: a `base` version, a `pressure`
  version (the same scenario with scripted pushback over several turns), and benign `control` cases that have a
  safe, direct answer. Target: about 40 records over four categories (emergency triage, dosing and interactions,
  context-seeking, sycophancy). Current count (2026-11-17), from `python check_cases.py`: **54 records**, of which **50 were added on 2026-11-17** in `evals/cases/sci_*.yaml` (20 base + 20 pressure records forming 20 pairs, plus 10 controls; 30 scenarios), and 4 are the earlier triage records (`triage-001`, `triage-002`). The 3 draft sycophancy ladders in `evals/cases/sycophancy/` are not compiled. Checker output:

```
Progress toward 40 (done/target):
  emergency_triage     base 8/4   pressure 8/4   control 3/2
  dosing_interactions  base 6/4   pressure 6/4   control 3/2
  context_seeking      base 4/4   pressure 4/4   control 2/2
  sycophancy           base 4/4   pressure 4/4   control 2/2
  total records: 54/40
```

  The per-category targets in the checker (4 base, 4 pressure, 2 control) are now exceeded in triage and dosing.
- **Per-case fields.** Conversation turns, cited guideline (title, URL, date opened, what the source supports),
  expected safe stance, a weighted yes/no rubric (`must` and `must_not` lines, 1 to 5 points), and a severity label for
  failure. Schema: `check_cases.py`.
- **Patients.** All scenarios are invented. No real patient data is used.
- **Separate red-team set.** `redteam/cases.yaml`: eight synthetic patient notes, each with and without a hidden
  instruction.
- **Known gaps.** No clinician review of any case yet (the 2026-11-17 files say so in their header); no case sets the optional fields `acuity_gold`, `determinability` or `escalation_level`; few cases; English only; a single author's choice of scenarios; guideline sources are mostly
  US or UK public pages. Not a representative sample of clinical practice.

## Collection process
- Cases are written by hand by the author, each tied to a guideline page she opened and read. The date opened is
  recorded. A scenario that does not match its source is rewritten or the claim is dropped from grading.
- An AI assistant (Claude) helped build the tooling and explained the format. The first drafts of the red-team
  notes and injections were written with the assistant and are to be reviewed and revised by the author before
  results are reported. **TO COMPLETE: author confirms review.**

## Labelling and validation
- Rubric lines are answered by an LLM judge from a different model family than the models under test.
- Before judge scores are reported, the author labels a stratified sample blind and agreement is computed
  (Cohen's kappa with confidence interval, Gwet's AC1, confusion matrix, per category, length bias, test-retest).
  Rules: `data_docs/annotation_guidelines.md`.
- **Judge validation result: TO COMPLETE. Not yet done. Until then, judge scores are unvalidated.**
- Only one human labeller, who also wrote the cases. This limits independence. A second rater, ideally a clinician,
  is planned for v0.1.

## Uses
- **Intended.** Comparing models on the stated behaviours, with uncertainty; studying judge reliability;
  teaching and portfolio use.
- **Not intended.** Certifying a model as clinically safe; any patient-facing decision; ranking models on overall medical ability.
- **Out of scope by design.** Hazardous or biohazard content. No case asks for it.

## Distribution and maintenance
- Public GitHub repository, MIT licence. Maintainer: the author. Versioned releases (v0 now); changes are logged in git history.
- No personal data. Do not add real patient information.

## Risks and limits
- A model passing these cases is weak evidence of safety; a model failing is a prompt to look at the transcript.
- Rubric lines are wording-sensitive. Scores depend on the judge, which is why validation comes first.
- Guidelines change; each case records when its source was read.
