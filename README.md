# Clinical LLM Safety Lab

Clinical AI tends to fail at the level of a single patient: a missed red flag in triage, a wrong dose, a model
that agrees with a patient when it should push back. Aggregate benchmark scores hide these failures. I build tools
and sit next to clinicians, and this lab is where I measure how language models fail on clinical tasks, case by case,
with the uncertainty stated. It is built on [Inspect](https://inspect.aisi.org.uk), the open evaluation framework
from the UK AI Security Institute.

## Status

**v0, in progress.** The evaluation itself has not been run. No results exist yet.

What exists today:

- [x] Python environment with pinned dependencies (`requirements.txt`, Inspect 0.3.289)
- [x] Inspect smoke test (`smoke_test.py`): one trivial sample, checks that the pipeline runs end to end on a local model

Planned:

- [ ] Case set for the four categories below, grounded in cited guidelines
- [ ] Inspect tasks and weighted rubric scorers
- [ ] Runs on small and mid-size local models plus one hosted reference model
- [ ] Judge-validation study (blind human labels vs an LLM judge)
- [ ] Red-team: medical prompt injection via patient notes, open-weight models only
- [ ] Datasheet and data documentation

Results (filled only from runs that can be reproduced from this repo):

| Category | Small local model | Mid-size local model | Hosted reference model | n cases |
|---|---|---|---|---|
| Emergency triage | not run yet | not run yet | not run yet | not run yet |
| Dosing and drug interactions | not run yet | not run yet | not run yet | not run yet |
| Context-seeking and abstention | not run yet | not run yet | not run yet | not run yet |
| Multi-turn sycophancy | not run yet | not run yet | not run yet | not run yet |

## What it measures

1. **Emergency triage.** Does the model recognise red-flag presentations and recommend the right level of urgency?
2. **Dosing and drug interactions.** Does it give doses and interaction warnings consistent with the cited guideline,
   and flag when it lacks the information to answer safely?
3. **Context-seeking and abstention.** Does it ask for missing clinical context, or say "see a clinician", when that
   is the safe answer, without refusing benign questions?
4. **Multi-turn sycophancy.** Does it keep a correct, guideline-based position when a patient pushes back over
   several turns?

Out of scope: hazardous or biohazard content. No case asks for it.

## Method

- **Weighted binary rubrics per case.** Each case has its own list of yes/no criteria, each with a weight, and
  undesirable behaviours carry negative weight. This follows the rubric design of HealthBench
  (Arora et al., 2025, [arXiv:2505.08775](https://arxiv.org/pdf/2505.08775)).
- **Cases grounded in cited guidelines.** Every case names the guideline section its correct answer comes from.
- **Paired base and pressure variants.** Each clinical scenario has a neutral version and a version where the
  patient applies pressure, so the effect of pressure is measured within the same scenario. The multi-turn pressure
  design draws on MedPRESS (Joy and Farhan, 2026, [arXiv:2608.02520](https://arxiv.org/pdf/2608.02520)) and
  SycoEval-EM (Peng et al., 2026, [arXiv:2601.16529](https://arxiv.org/html/2601.16529v4)).
- **Benign controls.** Ordinary questions with a safe, direct answer, to measure the cost of over-refusal.
- **Error bars.** Scores are reported with confidence intervals and paired comparisons between models, following
  Miller, "Adding Error Bars to Evals" (2024, [arXiv:2411.00640](https://arxiv.org/pdf/2411.00640)).
- **Judge validation protocol.** An LLM judge scores the rubrics. Before any judge score is reported, I label a
  stratified sample blind and compare: Cohen's kappa with a confidence interval, Gwet's AC1, agreement per category,
  a confusion matrix, and checks for judge bias. The judge is from a different model family than the models under test.
- **Red-team.** Indirect prompt injection hidden in patient notes, run against open-weight models only. Benchmark
  context: MPIB (Lee et al., 2026, [arXiv:2602.06268](https://www.arxiv.org/pdf/2602.06268)), which covers direct
  and retrieval-mediated injection. The patient-note setting here is my adaptation.

## Repo layout

```
README.md          this file
TODO.md            public work plan
LICENSE            MIT
requirements.txt   pinned Python dependencies
smoke_test.py      Inspect smoke test (one sample)
```

Planned (not in the repo yet): `cases/`, `tasks/`, `scorers/`, `judge_validation/`, `redteam/`, `data_docs/`,
`reports/`.

## Limitations

- Single rater for the human labels: me, a medical student. No external clinician review yet.
- Small number of cases, so confidence intervals will be wide. Differences between models may not be resolvable.
- Main runs use local, quantized models. Results may not transfer to full-precision or hosted versions.
- Model versions are pinned per run (exact tag and digest recorded). Results apply to those versions only.
- This is a research evaluation, not a clinical tool or a certification of any model.

## Reproduce

The smoke test works today, on a Mac with [Ollama](https://ollama.com) installed and `qwen3:8b` pulled:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
ollama pull qwen3:8b
.venv/bin/inspect eval smoke_test.py --model ollama/qwen3:8b
```

The evaluation itself: coming with v0.1.

## Contact

Open an issue on this repository, or find me on GitHub: [@yasferaru](https://github.com/yasferaru).

## License

MIT. See `LICENSE`.
