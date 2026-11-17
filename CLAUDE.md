# Rules for agents working in this repo

Several agents may work here in parallel, each from its own prompt. These rules keep their work from
colliding. When a rule and your prompt disagree, the prompt wins only if it explicitly names the file or rule.

## Files you do not edit unless your prompt explicitly owns them

- `README.md`
- `requirements.txt`, `requirements.lock`
- `evals/run_eval.py`
- `scripts/run_one.sh`
- `check_cases.py`. If you need a schema change, write a request to
  `docs/schema_requests/<your-prompt-name>.md` (the field, its type, the validation rule, and why) and leave
  `check_cases.py` alone.

## Dependencies

- Never add to `requirements.txt`. Put a new dependency in `requirements-extra/<name>.txt` with an exact pin
  (`package==x.y.z`), where `<name>` describes the feature (e.g. `requirements-extra/plots.txt`).
- Dev tools (pytest, ruff) are in `requirements-extra/dev.txt`.

## Metrics and statistics

- Metrics are pure functions on the results DataFrames from `evals/results.py` (`load_lines`, `load_samples`;
  schema in `docs/results_schema.md`). They do no file or network I/O.
- Anything random takes a `seed` argument with a fixed default and uses `numpy.random.default_rng(seed)`.
- Intervals come from `evals/stats/intervals.py`. Name the method wherever a number is shown:
  Clopper-Pearson for `critical_pass`, Wilson for red-team rates.
- Cases that share a `scenario_id` are not independent, and neither are the epochs of one case.
  Cluster accordingly.

## Code

- Every function has a docstring that states its formula or rule.
- Every function has a unit test against a hand-computed example (write the arithmetic in a comment).
- Test fixtures come from `tests/fixtures.py` (seeded). Do not read `logs/` in tests.
- No network calls in tests, and no paid model APIs anywhere unless the prompt says so. Use fakes for the
  judge and the model.
- Run before committing: `ruff check .`, `pytest`, `python check_cases.py`.
- Commit only the files you created or own. Stage paths explicitly, never `git add -A`.
