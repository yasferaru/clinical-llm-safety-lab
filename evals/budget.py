# ruff: noqa: E501
"""Persistent conservative local budget guard; it never guarantees provider billing."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BudgetPolicy:
    ceiling_usd: float = 25.0
    stop_usd: float = 20.0
    stage_limits: dict[str, float] | None = None

    def __post_init__(self):
        if not 0 < self.stop_usd <= self.ceiling_usd:
            raise ValueError("require 0 < stop_usd <= ceiling_usd")


def estimate_usd(input_tokens: int, output_tokens: int, input_per_million: float, output_per_million: float) -> float:
    return input_tokens * input_per_million / 1_000_000 + output_tokens * output_per_million / 1_000_000


class SpendLedger:
    """Atomic local reservations plus reconciled provider usage, keyed by run/stage/cell."""
    def __init__(self, path: str | Path, policy: BudgetPolicy):
        self.path, self.policy = Path(path), policy

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            value = json.loads(self.path.read_text())
            return value.get("entries", []) if isinstance(value, dict) else []
        except json.JSONDecodeError:
            raise ValueError(f"corrupt spend ledger: {self.path}")

    def record_policy_revision(self, stage: str, limit_usd: float, note: str) -> None:
        """Append an auditable stage-limit revision without altering spend entries."""
        if limit_usd <= 0:
            raise ValueError("stage limit must be positive")
        document = self._document()
        history = document.setdefault("policy_history", [])
        history.append({"stage": stage, "limit_usd": round(limit_usd, 6), "note": note})
        self._write(self.entries(), document)

    def spent(self) -> float:
        return sum(float(x.get("usd", 0)) for x in self.entries() if x.get("state") != "released")

    def reserve(self, key: str, stage: str, usd: float, provider: str) -> None:
        if usd < 0:
            raise ValueError("negative reservation")
        entries = self.entries()
        if any(x.get("key") == key for x in entries):
            return  # idempotent resume: never double-reserve a cell/batch
        stage_spent = sum(float(x.get("usd", 0)) for x in entries if x.get("stage") == stage and x.get("state") != "released")
        limit = (self.policy.stage_limits or {}).get(stage, self.policy.stop_usd)
        if self.spent() + usd > self.policy.stop_usd:
            raise RuntimeError("budget stop threshold would be exceeded; $5 reserve remains untouched")
        if stage_spent + usd > limit:
            raise RuntimeError(f"stage budget would be exceeded: {stage}")
        # Keep sub-millicent reservations exact enough for isolated retries.  The
        # ledger is a safety control, so rounding down a small reservation would
        # make its post-request check stricter than the approved plan.
        entries.append({"key": key, "stage": stage, "provider": provider, "usd": round(usd, 9), "state": "reserved"})
        self._write(entries)

    def reconcile(self, key: str, actual_usd: float, provider_usage: dict | None = None) -> None:
        entries = self.entries()
        for entry in entries:
            if entry.get("key") == key:
                entry.update({"reserved_usd": entry.get("reserved_usd", entry.get("usd")),
                              "usd": round(actual_usd, 9), "estimated_usd": round(actual_usd, 9),
                              "state": "reconciled", "provider_usage": provider_usage})
                self._write(entries)
                return
        raise KeyError(key)

    def reconcile_range(self, key: str, low_usd: float, high_usd: float, provider_usage: dict | None = None) -> None:
        """Retain reservation provenance when one provider's final charge is uncertain."""
        if low_usd < 0 or high_usd < low_usd:
            raise ValueError("invalid reconciliation range")
        entries = self.entries()
        for entry in entries:
            if entry.get("key") == key:
                entry.update({"reserved_usd": entry.get("usd"), "usd": round(high_usd, 9),
                              "state": "reconciled_range", "actual_low_usd": round(low_usd, 9),
                              "actual_high_usd": round(high_usd, 9), "provider_usage": provider_usage})
                self._write(entries)
                return
        raise KeyError(key)

    def mark_unconfirmed_transport(self, key: str, maximum_usd: float, note: str) -> None:
        """Keep a failed-before-HTTP reservation conservative without calling it spend.

        The reservation remains in ``spent()`` so later work cannot overcommit the
        local stage ceiling.  ``confirmed_usd`` deliberately remains zero: a DNS or
        connection failure supplies no evidence that the provider received a call.
        """
        if maximum_usd < 0:
            raise ValueError("negative maximum reservation")
        entries = self.entries()
        for entry in entries:
            if entry.get("key") == key:
                entry.update({"reserved_usd": entry.get("usd"), "usd": round(maximum_usd, 6),
                              "confirmed_usd": 0.0, "state": "unconfirmed_transport",
                              "reconciliation_note": note})
                self._write(entries)
                return
        raise KeyError(key)

    def _document(self) -> dict:
        if not self.path.exists():
            return {"schema_version": 1, "entries": []}
        try:
            value = json.loads(self.path.read_text())
        except json.JSONDecodeError:
            raise ValueError(f"corrupt spend ledger: {self.path}")
        return value if isinstance(value, dict) else {"schema_version": 1, "entries": []}

    def _write(self, entries: list[dict], document: dict | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = document or self._document()
        document.update({"schema_version": 1, "entries": entries})
        fd, tmp = tempfile.mkstemp(dir=self.path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(document, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, self.path)
