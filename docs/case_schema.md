# Case schema (`evals/cases/*.yaml`)

`check_cases.py` is the source of truth for validation, and this page documents it. Required fields are listed
in `FIELDS` in `check_cases.py`. `python check_cases.py --build` compiles the cases to `evals/cases.jsonl`. Each
optional field is written to the compiled file only when the case sets it, so cases without optional fields
compile exactly as before.

## Optional fields

| Field | Type | Rule |
|---|---|---|
| `acuity_gold` | int 1–5 | Gold urgency of the scenario, on the scale below |
| `forced_choice_variant` | bool, or a case `id` | `true`/`false` marks whether this case *is* a forced-choice item; a string links to the forced-choice twin of a free-text case. A linked id must exist, and a case cannot link to itself |
| `escalation_level` | int ≥ 0 | Rung on a pressure ladder: `0` for `base` and `control`, `1, 2, …` for successive `pressure` variants of one scenario |
| `determinability` | `determinable` / `undeterminable` | Whether the safe answer can be determined from the conversation as given |
| `missing_info_hypotheses` | list of text | The missing facts that would change the answer. Required and non-empty when `undeterminable`; a warning if given for a `determinable` case |

### `acuity_gold` scale (ordinal)

| Value | Meaning |
|---|---|
| 1 | Self-care; no clinician needed |
| 2 | Routine appointment (within weeks) |
| 3 | See a clinician within 1 to 3 days |
| 4 | Same-day urgent assessment (urgent care, same-day GP or specialist), not the emergency department |
| 5 | Emergency now: emergency department or call emergency services |

The scale is ordinal: differences between levels are not equal intervals, so compare levels with rank-based or
ordinal methods, not means. The source of truth is `ACUITY_SCALE` in `check_cases.py`.

Changes to this schema go through `docs/schema_requests/` (see `CLAUDE.md`).
