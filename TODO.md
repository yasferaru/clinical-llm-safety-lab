# TODO

Updated 2026-11-19 after the reproducible matrix-engineering phase (`AUDIT_v2.md` has
the earlier reasoning). **Keep this file current whenever work is completed, deferred,
or newly discovered.** In priority order.

## Evaluation-matrix engineering (before Phase 2)

- [x] Stop reports from parsing `manifest.json` as an Inspect log; regression test added.
- [x] Add offline-safe immutable response and judgment artifacts, dataset snapshots,
      provenance, content-addressed cells, resume/idempotency, and a matrix validator.
- [x] Add a deterministic end-to-end mock configuration and offline coverage.
- [x] Add a machine-verifiable 54-case/264-line integrity audit and a versioned
      automated statistical-analysis specification; clinical findings remain pending review.
- [x] Add an offline $25-budget policy, persistent local spend ledger, proposed
      independent-judge matrix, and provider/access checklist; no inference authorized.
- [x] Add mocked Anthropic, Gemini, Mistral, Groq, direct DeepSeek, and Together request adapters with credential
      presence checks, provider-specific payloads, error classification, retries,
      structured-judge parsing, usage/request-ID capture, and provenance retention.
- [x] Replace the proposed Together judge with the owner-selected four-provider,
      target-specific judge assignment. GPT-OSS is the common primary judge; second
      judges remain separate, non-pooled analyses. Verify all eight pairs offline.
- [x] Add an offline-only provider preflight command. It never contacts a provider and
      reports credential presence by environment-variable name only.
- [x] Add a guarded live matrix CLI and a GET-only authenticated model-catalog checker.
      The runner refuses provider calls without `--execute-billable`; its smoke forecast
      reserves all three configured attempts atomically before any request. Real access and billing reconciliation
      remain an operator gate.
- [x] Make access checks and live adapters share normalized credential loading and
      non-secret diagnostics for whitespace/quotes and a shell-versus-`.env` override.
- [ ] Wire approved provider-adapter selection and per-batch budget reservations into
      a production run CLI. Do not enable or execute that CLI without execution approval.
- [ ] Implement configured concurrency limits in the matrix executor, with provider-
      specific ceilings and global back-pressure.
- [ ] Implement retry policies in the matrix executor, recording every attempt; retain
      the current rule that an uncertain `in_flight` provider outcome is never retried
      automatically.
- [ ] Add an explicit operator reconciliation command for stale `in_flight` cells
      (mark failed, import confirmed provider result, or intentionally skip), with an
      audit trail.
- [ ] Export completed matrix artifacts into the existing `evals.results` / report
      contract, preserving target-response reuse and per-judge provenance.
- [ ] Add a report section/table for matrix completeness: completed, failed, skipped,
      pending, invalid, and unresolved provider outcomes by target/judge/case/epoch.
- [ ] Add a documented production config template with fixed model IDs, per-provider
      rate limits, retry policy, output retention, and privacy controls; never include
      credentials.
- [ ] Re-verify model IDs, access tiers, price sheets, and rate limits in the owner's
      provider accounts immediately before any spend.
- [ ] Before approved smoke/pilot spending, verify account access, current pricing and
      rate limits for all six requested IDs. In particular, Google currently marks
      `gemini-2.5-flash` deprecated; do not substitute it without owner approval.
- [x] Owner ran the GET-only model-catalog check from a network-enabled environment:
      Anthropic, Gemini, DeepSeek, and xAI returned HTTP 200 with every active model
      available. This verifies catalog access, not billable generation behavior.
- [ ] Resolve a reproducibility policy for DeepSeek's mutable `deepseek-flash` alias:
      record provider-returned `DeepSeek-V4.1-Flash-1017` metadata, or approve a
      frozen provider identifier if DeepSeek exposes one.
- [ ] Configure provider-side spend caps/alerts where available; local reservations
      cannot guarantee a provider will not bill delayed or uncertain requests.
- [x] Owner selected the three-target design: Grok is the universal independent
      primary judge; secondary judges remain target-specific and non-pooled.
- [ ] Add a protected migration/import path for legacy Inspect logs only if historical
      results need to be displayed beside matrix artifacts; do not rewrite old runs.

## Data and validation (highest value)

- [ ] Write the remaining ~36 cases (16 base+pressure clusters and 8 benign controls in total), each tied to a cited
      source. Add `acuity_gold` to every triage case, forced-choice twins for format sensitivity, and determinable,
      undeterminable and benign cases for abstention.
- [ ] Clinician review of every rubric. Start with the critical (4–5 point) lines; triage-002's critical lines
      reward an explanation (auditory-nerve tumour) rather than the safe action.
- [ ] Clinician review of the 3 draft sycophancy ladders (`evals/cases/sycophancy/`), then compile them.
- [ ] Blind-label about 100 stratified rubric lines (`export_labelling_sheet.py`), then run `analyze_agreement.py`.
      Label a second pass at least a day later for test-retest.
- [ ] Recruit a second human rater, ideally a clinician, for at least 30% overlap.
- [ ] Human-label `critical_pass` on a calibration sample (`judge_validation/critical_pass_human.csv`) so the
      report can show judge-corrected rates instead of the banner.
- [ ] Review and rewrite the AI-drafted red-team notes, injections and harm taxonomy. Write a second omission note
      so omission has a dev note.
- [ ] Review the two base/pressure pairs with non-identical rubrics
      (`triage-001`, `triage-002`). Approve a shared-line paired endpoint or clinician-approved
      rubric alignment; do not silently alter the historical benchmark.
- [ ] Review whether scripted assistant histories in all 22 pressure cases are an intended
      manipulation and document the resulting causal limitation before reporting pressure effects.
- [ ] Review machine-flagged shared-subcategory clusters for semantic overlap using clinical
      judgment; automated text similarity cannot establish duplicate clinical scenarios.

## Runs

- [ ] Smoke run `matrix-runs/smoke-three-target-xai-v1` is blocked in this execution
      environment: all 12 target cells ended in transport-level `ConnectError` before
      an HTTP response. Preserve its manifest, failed cells, and $0.399613 reservation;
      do not retry here. Run from the owner's network-enabled terminal after deciding
      how to reconcile/release the local reservation.
- [ ] P0 smoke finding: `google/gemini-2.5-flash` returned HTTP 404 to all four target
      calls and 19 judge calls although its catalog listing was visible. Stop and obtain
      owner approval for a replacement; do not silently substitute.
- [x] P0 smoke finding: 19 completed Grok judgment artifacts have an unparseable/null
      verdict despite JSON-mode requests. The xAI JSON-schema judge contract is now
      strict and missing/null verdicts are invalid, never completed.
- [ ] Re-run the 57 strict-schema Grok judgment cells only after owner authorization and
      a budget/retry decision; retain the 19 invalid historical Grok artifacts.
- [x] Retire the unavailable `gemini-2.5-flash-lite` provisional entry without deleting
      its 404 evidence. Provisionally configure `gemini-3.5-flash-lite` for the Google
      target and Claude's secondary judge, with six independent pairings. Generation
      access is still a required gate; a visible model-list entry alone is insufficient.
- [x] Prepare selective smoke-v2 resume: imported 8 completed target responses and 19
      compatible valid Anthropic judgments, leaving 4 target and 95 judge cells.
- [ ] The one-request 2.5 Flash-Lite probe attempted from this sandbox but failed at DNS/
      transport level before HTTP. Its $0.01 local reservation is now marked
      `unconfirmed_transport` (confirmed provider spend $0.00) but continues to count
      against the smoke cap conservatively; do not repeat without new authorization
      from a network-enabled terminal.
- [x] Obtain the owner-authorized, one-request non-clinical generation-access probe for
      `gemini-3.5-flash-lite`; first run the GET-only catalog check locally and retain
      its redacted output as evidence. Do not resume the smoke matrix until it succeeds.
- [x] `gemini-3.5-flash-lite` access probe returned HTTP 200, the requested provider
      model, and a request ID; reconcile its $0.01 reservation to $0.0000015. The old
      probe script did not retain returned text or completion-token count, so HTTP 200
      is access evidence only, not proof of a usable `OK` response.
- [x] Cap future v3 live requests at 1,024 target output tokens and 64 judge output
      tokens. Persist requested caps on newly written artifacts; Google thinking tokens
      count toward billed output. Reusing the eight historical stored responses remains
      intentional and their predecessor artifacts do not contain this new cap field.
- [x] Treat an HTTP-successful but empty target response as an invalid terminal cell;
      it cannot be silently reused or automatically retried as though no provider call
      occurred. The historical Gemini probe cannot be retroactively text-validated.
- [x] Execute the owner-authorized selective v3 smoke from the network-enabled local
      Terminal. This Codex shell has provider DNS restrictions and the attached macOS
      Terminal is unavailable to computer-use control. The staged first Gemini target
      passed (STOP, 466 output tokens, 2,151 characters, expected model/request ID);
      the remaining authorized v3 cells completed under the existing reservation.
- [x] Complete the selective v3 smoke: 12/12 target responses completed; 92/114 judge
      cells completed and 22/114 are invalid (not failed or silently completed).
- [ ] P0 smoke acceptance failure: do not report comparative smoke results or proceed to
      pilot. Rejudge only after explicit owner approval with a revised, validated judge
      output contract/cap: 15 Anthropic, 4 Google, and 3 Grok verdicts were invalid;
      saved raw outputs indicate likely cap-truncated verdicts or schema noncompliance.
- [x] P0 guard gap: the full runner did not halt when invalid judge cells were written.
      Add a fail-fast invalid-verdict threshold before any owner-approved rejudging;
      preserve this completed smoke run as evidence.
- [x] Offline remediation: classify all 22 invalid artifacts as truncation-suspected,
      add native verdict-first schemas and 128-token caps, preserve invalid cells as
      terminal checkpoints, and add fail-fast invalid/budget-reservation guards.
- [ ] Owner decision required: default rejudging all 22 invalid cells costs a capped,
      safety-margined maximum of $0.05760375, exceeding the $0.0399064 smoke remainder.
      Three malformed Grok cells expose an unambiguous verdict prefix, but remain invalid
      unless the owner explicitly approves an offline recovery policy.
- [x] Owner raised the smoke-stage cap to $0.60 while retaining the $25 total cap.
      The full 22-cell v4 plan now fits with $0.08230265 remaining; no inference has
      been authorized by this configuration-only budget revision.
- [ ] Before any owner-approved rejudge, import only the 12 immutable target responses
      into `matrix.smoke_rejudge_v4.yaml` and use `rejudge_plan_v4.json` as the exact
      allowlist. The guarded runner will refuse the plan under the current smoke cap.
- [x] v4 rejudge stopped with one preserved Google HTTP 503 failed cell after 21 valid
      new strict-schema judgments. Do not retry it without explicit authorization;
      the runner now checkpoints then fails fast on provider failures.
- [x] Offline retry/remediation: provider failures now checkpoint then halt; a v5
      single-cell no-retry plan and isolated mock passed. The resolver selects 113
      single-provenance valid cells and leaves exactly the preserved Google failure
      unresolved until an authorized retry succeeds.
- [x] Correct v5 retry accounting: imported v4 artifacts are now fingerprinted at
      process start and excluded from the new-request reservation check; any artifact
      created or changed during the retry is still charged and can trigger fail-fast.
- [x] Harden uncertain-outcome handling: v5 left the selected Google cell as an
      `in_flight` pre-response checkpoint with no raw output or usage. Future runners
      now stop before making any request when they encounter such a checkpoint; it
      requires explicit operator reconciliation rather than an automatic retry.
- [x] Classify v5 as an uncertain provider outcome (not a failed/invalid verdict):
      no local dispatch/receipt log existed before instrumentation, and no provider
      request/billing record is locally configured. Prepare the isolated v6 plan with
      a new artifact namespace and durable non-sensitive lifecycle evidence.
- [x] Prepare the owner-authorized isolated pilot: 16 stratified cases (eight intact
      base/pressure pairs), 78 rubric lines, 48 target responses, and 468 strict-schema
      judgments. The zero-retry $1.541726564 forecast fits the $2.50 pilot cap.
- [ ] Pilot is fail-stopped after 17 completed target responses and one preserved Google
      503 target failure (`ctx-malaria-which-pill-pressure`); no judgments ran. Prepare
      a separately authorized single-attempt recovery and response-selection resume.
- [x] The isolated Google recovery succeeded; response selection now has 18/48 pilot
      targets without overwriting the original 503. Reconcile its reported token usage
      to $0.000689500 while retaining the original pilot reservation.
- [ ] Pilot resume v2 is fail-stopped after 20 completed targets and one preserved Google
      ReadTimeout (`dos-warfarin-ibuprofen-base`). A generic zero-retry recovery-plan
      generator and provenance-preserving v3 resume path are prepared; wait for billing authorization.
- [ ] Pilot resume v3 has 45 valid targets, 438 valid judgments, and three preserved
      invalid DeepSeek targets whose 1,024-token caps were consumed entirely by default
      thinking. Target-invalid fail-fast and DeepSeek non-thinking recovery are ready;
      do not run the three prepared recoveries without billing authorization.
- [x] Pilot finalized: 48/48 targets and 468/468 strict-schema judgments complete;
      42/234 paired disagreements remain judge-specific descriptive evidence. Ledger
      retains two uncertain Google calls in the $1.120706400 conservative pilot high.
- [x] Owner raised the production-stage allocation from $13 to $16 while retaining the
      $20 operational stop and $25 total ceiling; production remains separately gated.
- [ ] Fix cost forecasting before any re-run: observed v3 usage-based cost was $0.2584261,
      exceeding its $0.249416 reservation because judge-input token assumptions were too
      low. The $0.50 cumulative smoke ceiling was still not exceeded ($0.4600936).
- [x] Diagnosed probe transport failure: the Codex sandbox cannot resolve provider DNS
      (`gaierror`, errno 8) for Google, Anthropic, DeepSeek, or xAI; no proxy is
      configured and TLS/HTTP were never reached. The owner's local Mac Terminal is a
      distinct network environment and successfully completed authenticated GET checks.
- [ ] Choose the mid-size local model.
- [ ] Run `scripts/run_one.sh` for each model: 5 epochs, explicit temperature and seed.
- [ ] Red-team: tune any new hardened prompt on dev only, freeze it, then run the test split (`REDTEAM_TEST=1`).
- [ ] Run the judge bias audit and drift test-retest on real logs (`bias_audit.py`; calls the judge API).

## Engineering

- [ ] Remaining schema checks from `docs/schema_requests/p3.md`, `p5.md`, `p6.md`:
      - ladder consistency (levels 0..L, 2k+1 turns);
      - forced-choice twin consistency;
      - `hypothesis_actions`;
      - confidence elicitation for selective prediction.
- [ ] A rubric-line flag marking the lines that define the sycophancy stance.
- [ ] Add a canary string to the case files, and keep a private held-out case split.

## Later (v0.1)

- [ ] ~150 cases, a second judge family, a quantization experiment (4-bit vs 8-bit).
- [ ] Datasheet completed after the runs (`data_docs/datasheet.md`).
- [ ] One-page client-style failure report.
# Current execution blocker (2026-11-20)

- [x] Recovered the Google quota-failed production target in an isolated one-cell attempt; retain the original HTTP 429 artifact as historical evidence.
- [x] Preserved the metadata-only v2 merge attempt after its pre-copy filesystem error; v3 is the active fresh resume root.
- [x] Built and validated the production v3 selection root: 179 unique reusable target responses and no selected failed, invalid, or in-flight artifacts.
- [ ] Obtain explicit authorization before issuing the remaining billable production requests from the v3 resume root.
- [ ] Execute only the isolated xAI judgment recovery after a separate billable authorization; then select it into a fresh production resume root.
- [x] Prepared the isolated xAI judgment recovery and production v4 selection root: 486 targets and 23 valid judgments imported, with 4,729 judgment cells pending.
- [x] Recovered and selected the xAI judgment in production v5; 24 valid judgments are selected and 4,728 remain pending.
- [ ] Obtain explicit authorization before issuing the remaining 4,728 billable production judgment requests from the v5 resume root.
- [ ] Execute the owner-authorized remaining production v5 judgments with the retained production reservation; stop on the first failure or budget discrepancy.
- [ ] Obtain explicit authorization to execute only the v6 safe-continuation allowlist; the v5 ReadTimeout cell remains blocked pending operator reconciliation.
- [ ] Obtain explicit authorization for v7's bounded 429/503 recovery policy (maximum three total attempts); two ambiguous/failed historical cells remain excluded.
- [x] Added provider timeout/pacing/backoff/circuit-breaker policy plumbing and an immutable, allowlisted continuation workflow; automatic retries remain disabled in the approved production policy.
