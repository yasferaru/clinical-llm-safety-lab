# TODO

Updated 2026-11-17 after the v0 integration (`AUDIT_v2.md` has the reasoning). In priority order.

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

## Runs

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
