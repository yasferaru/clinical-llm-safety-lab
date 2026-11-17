# Extra requirements

A prompt or branch that needs a new dependency pins it here as `requirements-extra/<name>.txt` (`package==x.y.z`)
instead of editing `requirements.txt` (see `CLAUDE.md`). The integration step merges these files into
`requirements.txt`, pinned to the versions the tests ran with, and regenerates `requirements.lock`.

Merged on 2026-11-17: `plots.txt` (matplotlib), `triage-metrics.txt` (scikit-learn), `dev.txt` (pytest, ruff).
The provider SDKs used for hosted runs (`mistralai`, `groq`) are pinned in `requirements.txt` too.
