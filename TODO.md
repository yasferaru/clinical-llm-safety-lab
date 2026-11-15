# TODO

- [ ] choose same-family small + mid-size local models, hosted reference model, different-family judge (chosen so far: small local qwen3:8b, hosted reference Mistral Small, judge gpt-oss-120b via Groq; mid-size local model to choose)
- [ ] write ~40 cases (16 clusters base+pressure, 8 controls) grounded in cited guidelines
- [ ] smoke-test, time, run all models, record exact model tags and digests
- [ ] labelling guidelines, blind-label ~100 stratified items, run judge, kappa+CI, AC1, per-category, confusion matrix, bias checks
- [ ] second labelling pass (test-retest) and judge test-retest
- [ ] one red-team finding (indirect injection via patient note) with severity scale and fix (harness and taxonomy built; runs pending)
- [ ] baseline vs hardened_v1 prompt ablation with benign-control refusal cost
- [ ] datasheet and data_docs (draft in data_docs/datasheet.md; to finish after runs)
- [ ] v0.1: ~150 cases, second judge family, clinician rater, quantization experiment (4-bit vs 8-bit), probes on a public medical QA set
- [ ] one-page client-style failure report; pilot offer section
