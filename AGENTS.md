# Session handoff requirement

Before making any code, configuration, artifact, or production-run changes, read
[`docs/reliability_refactor_handoff.md`](docs/reliability_refactor_handoff.md)
completely. It is the current authoritative handoff for the six-chunk reliability
refactor, including production safety boundaries and known integration gaps.

Do not resume production or make provider API calls unless the current user
explicitly authorizes that action.
