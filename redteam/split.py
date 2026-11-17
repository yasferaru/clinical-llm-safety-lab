"""Fixed, stratified dev / held-out test split of red-team notes.

Rule (also in redteam/scoring.md). Within each attack_type, sort note ids and pair them in order:
(1st, 2nd), (3rd, 4th), ... For each pair, the FIRST note goes to
    'test' if int(sha256(f"{SPLIT_SALT}:{id}").hexdigest(), 16) % 2 == 0 else 'dev'
and the second note to the other split. A note without a partner yet uses the same hash rule; when a
partner is added later it takes the opposite split. Existing notes therefore never move as notes are
added (provided new ids sort after old ones within their attack type, e.g. rt-009, rt-010).
The salt is the fixed seed and must not change. Prompts (e.g. hardened_v*) may be tuned only on DEV.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable, Mapping

SPLIT_SALT = "clinical-llm-safety-lab/redteam-split/v1"
SPLITS = ("dev", "test")


def _hash_split(note_id: str) -> str:
    """'test' if sha256(salt:id) is even, else 'dev'."""
    h = int(hashlib.sha256(f"{SPLIT_SALT}:{note_id}".encode("utf-8")).hexdigest(), 16)
    return "test" if h % 2 == 0 else "dev"


def assign_splits(cases: Iterable[Mapping]) -> dict[str, str]:
    """{note id: 'dev' | 'test'} for cases with 'id' and 'attack_type' (rule in the module docstring)."""
    by_type: dict[str, list[str]] = defaultdict(list)
    for c in cases:
        by_type[c["attack_type"]].append(c["id"])
    out: dict[str, str] = {}
    for ids in by_type.values():
        ids = sorted(ids)
        for i in range(0, len(ids), 2):
            first = ids[i]
            out[first] = _hash_split(first)
            if i + 1 < len(ids):
                out[ids[i + 1]] = "dev" if out[first] == "test" else "test"
    return out


def select(cases: list[Mapping], split: str) -> list[Mapping]:
    """The cases in `split` ('dev', 'test' or 'all'), in input order."""
    if split == "all":
        return list(cases)
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS + ('all',)}, got {split!r}")
    assigned = assign_splits(cases)
    return [c for c in cases if assigned[c["id"]] == split]
