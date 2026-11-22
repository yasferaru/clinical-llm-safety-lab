# AUDIT v2: clinical-llm-safety-lab after the v0 integration

Re-audit on 2026-11-17 of branch `integration`, against the original checklist (A–G) in `AUDIT.md` and its
29 findings. Integrated so far: P0 scoring hardening; P1 inference; P2 reliability; P3 triage; P4 red-team; P5
sycophancy; P6 abstention; P7 judge validation; P8 uncertainty; P9 judge correction; P10 report and wiring.

Checks run:

- `pytest`: 436 passed, no network.
- `ruff check .`: clean.
- `check_cases.py`: 0 errors, 4/40 records; `cases.jsonl` matches the YAML.
- The full report: built on synthetic fixtures (all 11 sections) and on the real `logs/`.
- Offline Inspect runs with the `mockllm` model, for the clinical task and the red-team task.
- No paid API was called.

## 1. Executive summary

1. **The engineering findings are fixed.** Of the 29 original findings, 15 are fixed, 5 are partly fixed and 9
   remain (tables below). The scoring is fail-safe, the judge must be independent (enforced at task start and in
   the report), every number has an interval and a denominator, and runs record what they used.
2. **Everything that remains is data, labels or clinical review**, not code: 4 cases, no human labels, no
   clinician review, AI-drafted red-team notes and ladders.
3. **The judge is still unvalidated.** The report prints a "JUDGE NOT VALIDATED" banner per model on every real
   run, because no calibration labels exist.
4. **The existing results are from runs that predate the harness**: uncommitted code, 1 epoch, a moving alias,
   unpinned temperature. The README now says so. Re-scoring them under the fail-safe rule changed nothing.
5. **Statistical power is the binding constraint.** At the planned 40 cases, the minimum detectable difference
   in critical-pass rate is 24–40 percentage points. A 10-point difference needs about 390 cases.
6. The three highest-value next steps are in section 5.

## 2. Original findings: status

### Fixed (15 original findings, plus 2 items outside the original list)

| # | Original finding | Fix | Where |
|---|---|---|---|
| 1 | `critical_pass` passed when the judge errored on a critical `must_not` | A judge error fails every critical line and is logged as `judge_error` | `evals/run_eval.py` `combine`; `tests/test_scoring.py` |
| 2 | Default judge graded its own family; no guard | No default judge; `assert_judge_independent` at task start (clinical and red-team); the report excludes and lists dependent-judge runs; `run_one.sh` checks before any call | `evals/judge_validation/independence.py`, `config/model_families.yaml` |
| 4 | README reported judge scores despite "validation first" | README states the judge is unvalidated; the report shows a per-model banner (P9) | `README.md`, `evals/report.py` |
| 5 | "Paired comparisons" claimed, no code | Cluster-robust paired differences plus minimum detectable effect | `evals/stats/inference.py`, report section 3 |
| 6 | SE ignored clustering | Report: cluster-robust CI by scenario, per-case means over epochs, no interval below 5 scenarios. Inspect's live metric: `stderr(cluster="cluster_id")` | `evals/report.py`, `evals/run_eval.py` |
| 8 | Red-team summary pooled all logs and epochs | Latest run per (model, prompt, split); the note count is shown next to n | `redteam/report.py` |
| 11 | Agreement bootstrap resampled correlated lines | `groups=` resamples whole replies; the CLI clusters by reply | `analyze_agreement.py` |
| 14 | `VERDICT: NOT` parsed as NO | Word-boundary parse; unreadable output becomes a judge error | `run_eval.parse_verdict` |
| 17 | Epoch reduction: critical fraction, epoch-1 metadata only | `rubric_epochs` reducer keeps every epoch; `critical_pass_all` gives worst-of-k; pass^k in P2 | `run_eval.py`, `evals/stats/reliability.py` |
| 22 | ASR pooled cosmetic and dangerous attacks | ASR and CHER are separate, per attack type, with a complied × harmful 2×2 | `redteam/harm.py`, `harm_taxonomy.yaml` |
| 23 | Run manifest incomplete | Records the git commit and dirty flag, Inspect and Python versions, reported model versions, Ollama ID, judge and its prompt sha256, temperature, seed, epochs, and case and prompt hashes; written even when a step fails | `scripts/write_manifest.py`, `run_one.sh` |
| 25 | Bootstrap quantiles hand-indexed | `numpy.quantile` | `analyze_agreement.py` |
| 27 | Kappa paradox not flagged | PABAK, prevalence and bias indices, and a printed paradox warning | `analyze_agreement.py` |
| 28 | No lock file | Hashed `requirements.lock` (uv, universal, Python 3.12); `requirements.txt` is the exact tested environment, including the provider SDKs | `requirements.*` |
| 29 | Tests were scripts; cwd-relative paths; duplicated helpers | pytest; `check_cases` independent of cwd; shared bootstrap and McNemar helpers | `tests/`, `evals/stats/` |
| — | CI missing | Lint, tests, case check, stale-`cases.jsonl` check, and a report smoke job on synthetic fixtures | `.github/workflows/ci.yml` |
| — | Agreement bugs found in P7 (empty input crash; silent wrong kappa on unequal lists; `None` crash; output paths) | Fixed, with regression tests | `docs/judge_bias_audit.md` |

### Partly fixed (5)

| # | Finding | Done | Still open |
|---|---|---|---|
| 7 | Results not reproducible (dirty tree, alias, default temperature) | `run_one.sh` pins temperature, seed and epochs, and records everything | The only clinical results are still the 2026-11-15 runs. They must be re-run with `run_one.sh` |
| 9 | Red-team regex false positives and negatives | Judge plus high-precision rules for compliance and harm, measured on a 24-reply set | `task_ok` (utility on clean notes) is still a regex presence check: it passed a reply stating warfarin 50 mg |
| 15 | Judge `max_tokens=1500` can truncate the reasoning | Unreadable verdicts are counted (`judge_errors`) and shown in the report | The limit is unchanged; the reasoning effort is not set |
| 21 | `hardened_v1` tuned on its own test notes | Dev/test split, prompt guard, caveat printed in reports | `hardened_v1` stays contaminated; a new prompt tuned on dev only is needed |
| 26 | Length-bias check pools must and must_not | Content-preserving perturbation audits (P7) measure verbosity bias directly | The old Pearson check in `analyze_agreement.py` is unchanged |

### Remain (9)

| # | Finding | Why it remains | Next action |
|---|---|---|---|
| 3 | triage-002 critical lines reward an explanation (tumour), and the 5-point action line has no time frame | Clinical content | Clinician review of every critical line (step 3 below) |
| 10 | Pressure variants differ from base in more than pressure (a scripted correct answer, different rubric) | Case design | Keep the rubric identical across a cluster; report the pressure effect on shared lines |
| 12 | Labelling sample stratified by judge verdict; kappa unweighted | `export_labelling_sheet.py` unchanged | Store the stratum weights and report weighted agreement, or stratify by category only |
| 13 | Single rater, who wrote the cases and read the transcripts; `key.csv` sits next to `sheet.csv` | Process | A second rater (step 2); label only logs not yet read; write the key elsewhere |
| 16 | `rubric_score` is 0 when there are no `must` points; no N/A verdict; floor hides severity | Scoring design | Add N/A verdicts; report the unfloored score |
| 18–20 | Checker gaps (missing control, base/pressure turn equality, per-line citations); ambiguous pediatrician line; unresolved review note in `triage-002.yaml` | Checker and case content | Fold into the case-writing and review pass |
| 24 | Public cases, no canary string | Not done | Add a canary GUID; keep a private held-out split |

## 3. New findings from the integration

| Sev | Where | Finding | Fix |
|---|---|---|---|
| MED | process | P1, P3, P5, P6, P8 and P9 were left uncommitted in the shared working tree. They were reviewed, tested and committed by the integrator with attribution, unchanged except for one test adjusted to the new schema field. | Each prompt should commit its own work on a branch |
| MED | `evals/report.py` section 2 | `critical_pass_trials` uses Clopper-Pearson over case × epoch trials, which treats correlated trials as independent. It is labelled as such; the per-case row comes first | Read `critical_pass_all_epochs` |
| MED | `run_eval.py` | The judge runs at temperature 0, but no judge seed is set, and gpt-oss is a reasoning model, so verdicts can still vary | Run `drift.retest_agreement` on real logs |
| LOW | `evals/triage_metrics.py` | Its own `mcnemar_exact` duplicates `evals/stats/paired.py` (the formulas agree) | Import the shared one |
| LOW | `strip_think` | Defined in `run_eval.py`, `redteam/run_redteam.py` and `redteam/harm.py` | Move it to one helper |
| LOW | `redteam/prompt_log.jsonl` | Runs append to a tracked file | Commit it together with the results it documents |
| INFO | `evals/report.py` | `MIN_CLUSTERS = 5` is a judgement call. P1 notes that normal intervals stay optimistic below about 30 clusters | Revisit when there are more cases |

## 4. Checklist A–G, re-run

**A. README vs code.**
- Every number in the README was checked against the logs or the code.
- The critical-pass intervals are named Clopper-Pearson and computed by `evals/report.py`; the red-team intervals
  are named Wilson.
- The metric formulas in the README match `combine`, `reliability.py`, `triage_metrics.py`, `abstention.py`,
  `harm.py` and `judge_correction.py`.
- Claims the code did not support were removed: section-level citations, benign controls, and "judge scores only
  after validation".

**B. Scoring.**
- The scoring is fail-safe and tested end to end with a fake judge (malformed output, refusal, timeout).
- Epoch reduction is correct and tested.
- Critical lines per case are listed in `AUDIT.md`; their clinical content is unchanged (finding 3).

**C. Judge validation.**
- The kappa, AC1 and bootstrap implementations are verified on hand-computed examples, constant columns, tiny n
  and the kappa paradox.
- The bootstrap resamples replies, and its seed is fixed.
- Independence is enforced in code.
- Nothing stops a raw judge number from being shown, but the report labels it unvalidated per model.
- The labelling sheet is blind and shuffled. The annotator's prior exposure (13) and the stratification weights
  (12) remain.

**D. Data.**
- `check_cases.py` rejects duplicate ids, weights outside 1–5, missing citations, bad optional fields, and
  ladders without `new_information_provided`.
- It still does not require a control or check that a pressure case starts like its base.
- `cases.jsonl` is current, and CI enforces it.
- Confounds (10), contamination (24) and clinical review (3, 19) remain.

**E. Red-team.**
- A judge plus high-precision rules replace the regex for compliance. Harm is labelled separately (H0–H3).
- Dev/test split with a guard; 5 repeats by default; k/n with Wilson intervals and note counts.
- The paired McNemar test now includes utility.
- `task_ok` is still a regex check (9). `hardened_v1` is still contaminated (21).

**F. Reproducibility and engineering.**
- `run_one.sh` uses `set -euo pipefail`, quoting, explicit temperature, seed and epochs, a full manifest, and a
  per-run directory and report.
- Pins and a lock file; no secrets in git history (re-checked); CI as above.

**G. Methodological gaps, ranked by impact on validity.**

1. **No human labels.** The judge's error rate is unknown, so every rate is a raw judge rate. Tooling is ready
   (labelling sheet, agreement, correction); the labels are not.
2. **Too few cases and no clinician review.** With 4 cases nothing can be compared. Even at 40, the minimum
   detectable difference in critical-pass rate (80% power, α = 0.05; `evals.stats.inference.minimum_detectable_effect`):

   | Comparison | Minimum detectable difference |
   |---|---|
   | Paired, 30% discordant cases | 24 points |
   | Paired, scenario-clustered (design effect 40/24) | 31 points |
   | Unpaired-equivalent | 31 points |
   | Unpaired, clustered | 40 points |
   | One 10-case category, paired | 49 points |

   Detecting a 10-point paired difference needs about 390 cases.
3. **A single rater who also wrote the cases**, with no inter-rater agreement. This caps every validity claim.
4. **Rubric validity:** critical lines that reward explanations (3), and pressure confounds (10).
5. **Format sensitivity is unmeasured:** free text only, no forced-choice twins (P3 tooling is ready).
6. **Quantized local vs hosted models:** quantization, chat template and sampling differ. Temperature and seed are
   now pinned and recorded, but the quantization comparison (4-bit vs 8-bit) is not done.
7. **Over-refusal cannot be measured:** 0 benign controls today. Even the planned 8 give a Wilson upper bound of
   32% at 0/8.

## 5. Three highest-value next steps (validity gain / effort)

1. **Human labels and a second rater.** Label about 100 stratified rubric lines blind, plus a `critical_pass`
   calibration sample (`judge_validation/critical_pass_human.csv`). Have a second rater, ideally a clinician,
   label at least 30% of the same items. This turns on the judge-corrected rates, removes the "not validated"
   banner where the labels support it, and gives the first inter-rater agreement. Effort: days. Gain: every
   judge-scored number becomes interpretable.
2. **Clinician review of the rubrics and the drafts.** Review every critical (4–5 point) line, the 3 draft
   sycophancy ladders and the red-team notes. Fix triage-002 so the critical line rewards the safe action and a
   time frame, and keep each cluster's rubric identical between base and pressure. Effort: one review session per
   ~10 cases. Gain: critical_pass measures safety rather than wording.
3. **More cases, written to the schema the tools already use.** Reach the planned ~40 first, with:
   - `acuity_gold` on every triage case;
   - forced-choice twins;
   - determinable, undeterminable and benign cases;
   - compiled ladders.

   Then rerun each model with `scripts/run_one.sh` (5 epochs). Effort: the largest. Gain: every section of the
   report has data. Power stays limited (section 4G), so treat between-model claims as exploratory until
   n ≈ 150+.

## 6. Claims I could not verify

- That CI passes on GitHub: the branch is not pushed. A local replay of the CI steps in a fresh venv built from
  `requirements.lock` passed.
- Whether Ollama honours `--seed`, and whether Groq and Mistral report a concrete model version rather than the
  alias. The manifest records whatever is returned.
- The papers cited by the P3, P5, P6 and P9 modules (e.g. ClinDet-Bench, MedAbstain, arXiv:2511.21140). Those
  modules say they are "cited as given by the repo owner and not re-checked". I did not check them either.
- The clinical correctness of the draft sycophancy ladders and of the red-team harm examples.
- That the 2026-11-15 clinical runs used code identical to what was later committed (they ran on a dirty tree).
