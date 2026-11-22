# Three-target xAI-primary benchmark plan — no inference authorization

Owner-approved design, provisionally updated 2026-11-20. Active providers are Anthropic, Google,
DeepSeek, and xAI. Historical Groq and Mistral artifacts are preserved but excluded.

## Active matrix

| Target | Common primary judge | Separate secondary judge |
|---|---|---|
| `anthropic/claude-haiku-4-5-20251001` | `xai/grok-4.7` | `google/gemini-3.5-flash-lite` |
| `google/gemini-3.5-flash-lite` | `xai/grok-4.7` | `anthropic/claude-haiku-4-5-20251001` |
| `deepseek/deepseek-flash` | `xai/grok-4.7` | `anthropic/claude-haiku-4-5-20251001` |

All six target–judge cells are family-independent. Grok is the only common primary
judge and supports cross-target comparisons. Secondary results are target-specific
robustness analyses; do not pool or average them as a calibrated jury.

## Public pricing / capabilities checked 2026-11-20

`grok-4.7` has documented structured JSON-schema outputs and is $2/M input plus $6/M
output. [xAI models](https://docs.x.ai/developers/models),
[xAI structured outputs](https://docs.x.ai/developers/model-capabilities/text/structured-outputs)
The active configuration records the existing documented Claude, Gemini, and DeepSeek
prices. Google documents `gemini-3.5-flash-lite` at $0.30/M input and $2.50/M output
(including thinking tokens when billed); see [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing).
Its exact account availability and quota remain a generation-access gate, not a catalog-only claim.

## Smoke budget

The four stratified smoke cases contain 19 rubric lines. With 3 targets this is 12
target calls and 114 assigned judgment cells. The current selective resume has 8 stored
target responses and 19 compatible Anthropic judgments; it needs 4 target and 95 judge
calls after the separately approved one-call Google access probe succeeds. The shared
ledger conservatively accounts for $0.2016675: $0.191666 prior high reconciliation plus
a $0.01 DNS-failed probe reservation that is **not** confirmed provider spending.
The successful 3.5 Flash-Lite access probe reconciled from its $0.01 reservation to
$0.0000015 estimated usage; its returned text was not retained, so it establishes
generation access but not usable response text. New live calls use a 1,024-token target
cap and a 64-token judge cap; Google bills thinking tokens as output. At the configured
one retry per cell, resuming the 99 unfinished calls requires a $0.249416 local
reservation ($0.124708 single-attempt estimate). With all historical and unresolved
entries, the worst-case smoke commitment is $0.4510835, leaving $0.0489165 below the
$0.50 stage ceiling. This estimate uses configured token assumptions, not a promise of
provider charges.
This local reservation cannot guarantee delayed provider billing; configure provider-side
spend controls as well.

## v3 smoke outcome (2026-11-20)

The selective smoke completed all 12 target responses, but only 92 of 114 judgment
cells had valid boolean verdicts; 22 are preserved as invalid. This is an engineering
acceptance failure, not an evaluation result. Do not compare target models or proceed to
pilot until an owner-approved rejudging fix is validated. The 99 newly executed cells
had observed usage-based cost of $0.2584261 (including $0.233666 xAI reported cost),
which exceeded the $0.249416 pre-reservation because real judge prompts were longer than
the token assumptions. Cumulative conservative smoke accounting is $0.4600936, still
below the $0.50 ceiling.

The owner subsequently raised only the smoke-stage ceiling to $0.60, with the $25
total ceiling unchanged. Ledger policy history preserves both limits. The planned
22-cell v4 rejudge has a $0.05760375 safety-margined maximum with retries disabled;
combined with current smoke accounting it is $0.51769735, leaving $0.08230265.

## Post-smoke cost calibration

Future judge forecasts apply a 25% margin to observed v3 per-call cost: $0.0040994
for Grok, $0.0002461 for Gemini, and $0.0008769 for Claude. Grok's observed figure
uses provider-reported ticks and therefore captures billed reasoning not visible in
completion-token counts. At these calibrated rates, a 54-case, three-target,
three-epoch production design projects to about $14.9 before any further safety
reserve: within the $25 experiment ceiling but above the current $13 production-stage
limit. This is a budget-policy blocker, not authorization to change the matrix or cap.

```bash
# GET-only model-catalog check; no prompts or generation
.venv/bin/python scripts/check_provider_access.py

# deterministic, no-network topology check
.venv/bin/python -m evals.matrix config/matrix.mock_four_provider.yaml --mock
```

The live smoke configuration is `config/matrix.smoke_four_provider.yaml`. It must not
be executed until the owner explicitly authorizes billable inference.
