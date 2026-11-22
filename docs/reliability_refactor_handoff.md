# Reliability refactor handoff

Last updated: 2026-11-22

Read this file before working on the clinical LLM safety benchmark's reliability refactor. It records implemented work, production safety boundaries, and remaining integration work.

## Production state

Production is paused at `matrix-runs/production-three-target-strict-resume-v7`.

- 486 target responses are completed.
- 1,768 valid judgments are completed.
- 2,984 logical judgment cells remain unresolved.
- 2,982 cells are baseline-continuation eligible.
- Two historical cells are blocked and must remain excluded from automatic continuation:
  - `b0a4e7c325b2bef84baf57cca655cbec0a0c41a8feea1b4b4a1546b5fc64a914`
  - `d1f48091baf614debc0d315e457e0ceeb0663b4a104f80bfe549185036828352`

Never make billable provider calls or resume production without explicit user authorization.

## Completed: Chunk 1

`evals/attempt_artifacts.py` introduces immutable per-provider-attempt evidence.

```text
attempts/
  numbers/<logical_cell_id>/<attempt_number>
  records/<attempt_id>.json
```

- Logical cell IDs identify deterministic benchmark work; attempt IDs identify individual provider invocations.
- Attempt numbers are unique per logical cell and reserved atomically.
- Records contain provider/model, request identity/parameters, timestamps, response/error, provider request ID, usage/billing evidence when available, and terminal status.
- Record writes are atomic create-only operations; historical evidence is never overwritten.
- Legacy `responses/*.json` and `judgments/*.json` artifacts remain readable without migration.

`evals/providers.py` returns evidence for individual HTTP attempts, including intermediate failures. `evals/matrix.py` writes immutable attempt records alongside legacy logical artifacts.

## Completed: Chunk 2

Attempts have fsynced append-only lifecycle journals:

```text
attempts/lifecycle/<attempt_id>.jsonl
```

Lifecycle states:

```text
reserved
  -> dispatch_intent
  -> dispatch_started | possibly_started
  -> provider_response_received
  -> validated
  -> completed | confirmed_rejection

dispatch_intent | dispatch_started | possibly_started -> ambiguous
```

Recovery rules:

- `reserved`: safe to abandon because no dispatch could have occurred.
- `dispatch_intent`, `dispatch_started`, `possibly_started`, or `ambiguous`: block automatic retry; the outcome may have reached the provider.
- `provider_response_received`: validate from durable evidence, without a provider call.
- `validated`: finalize and select from durable evidence, without a provider call.
- `completed`: skip execution.

Atomic logical selections live at:

```text
attempts/selections/<logical_cell_id>.json
```

They select at most one validated completed attempt per logical cell, retain provenance, are idempotent for the same attempt, and cannot replace an existing selection. Legacy completed artifacts can be recognized read-only as selections.

`MatrixRun` durably records `dispatch_intent` before invoking a provider. A crash after intent cannot look like a safe automatic resend.

## Tests already added

- `tests/test_attempt_artifacts.py`
- `tests/test_attempt_lifecycle.py`

Last focused verification: 50 tests passed; Ruff and `git diff --check` passed.

## Important current gaps

The lifecycle and selection abstractions are implemented but are **not yet fully integrated as the production runner's source of truth**.

- The matrix runner records dispatch intent, but does not yet consistently advance receipt, validation, terminal finalization, and atomic selection states.
- Legacy logical artifacts still control normal resume behavior. Do not replace them until compatibility reads and runner integration are complete.
- Lifecycle mutation assumes a single writer. Selection is atomic under concurrency, but future live concurrency may need a writer lease for lifecycle mutation.
- `resolved_selection.json` describes both blocked cells as uncertain ReadTimeout outcomes, while earlier context calls `d1f48091baf614debc0d315e457e0ceeb0663b4a104f80bfe549185036828352` a confirmed xAI 429. Resolve this from immutable evidence; do not rewrite history casually.
- The repository worktree has many pre-existing modified and untracked files. Do not infer ownership from a broad `git status`; stage only files explicitly owned by the current task.

## Remaining work

1. **Chunk 3:** durable retry-budget ledger and reservations. Do not change retry behavior yet.
2. **Chunk 4:** controlled retry eligibility and exhaustion policy. Ambiguous attempts must never be retried automatically.
3. **Chunk 5:** end-to-end production-runner integration: lifecycle transitions, terminalization, selection, compatibility reads, and restart recovery.
4. **Chunk 6:** scope has not yet been supplied; do not infer it.

Before any continuation, preserve the two blocked cells above in the exclusion set and verify that the 1,768 historical completed judgments are recognized rather than regenerated.
