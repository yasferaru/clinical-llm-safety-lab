"""Judge bias audits on already-scored responses.

Each perturbation audit changes the FORM of a judged reply but not its content, re-judges the same rubric
line, and reports how often the verdict changes:

    length_pad       add neutral, content-free sentences before and after the reply
    length_compress  remove content-free filler sentences and redundant whitespace
    format           Markdown -> plain prose, or plain prose -> Markdown bullets
    order            permute independent Markdown sections / bullet items / paragraphs

    change rate = (b + c) / n,  b = YES->NO pairs, c = NO->YES pairs, n = pairs with two readable verdicts
    reported with a 95% Wilson interval and an exact McNemar test of b vs c (direction of the bias).
    Pairs are oriented (before -> after) as original -> padded, original -> compressed, original -> reordered,
    and for format as PROSE version -> MARKDOWN version whichever one was the original, so that a judge that
    prefers Markdown shows up in one direction instead of cancelling out.

By default the original reply is re-judged at the same time as the perturbed one (so both verdicts come from
the same judge state) and judged a third time as a no-perturbation control: the control change rate is the
judge's own noise floor. A perturbation that changes content tokens raises (see `check_content`).

Self-preference: the judge's verdicts on replies from its OWN family vs OTHER families, matched on case and
rubric line, with McNemar on the discordant pairs. With human labels the outcome is a false pass (judge
passes, human fails), which removes real quality differences between families; without them the raw pass
rate is compared and quality differences confound it. A significant own-family advantage raises
SelfPreferenceError (fail loudly).

All audits first call assert_judge_independent(judge_model, models under test), except the self-preference
probe, which by design judges its own family and checks that the family labels are right instead.
A judge is any `async (conversation, reply, criterion) -> True | False | None` (None = unreadable).
"""
from __future__ import annotations

import asyncio
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.judge_validation.independence import (JudgeIndependenceError, assert_judge_independent,  # noqa: E402
                                                 family_of)
from evals.stats.intervals import wilson  # noqa: E402
from evals.stats.paired import mcnemar_exact  # noqa: E402

LineJudge = Callable[[str, str, str], Awaitable[Optional[bool]]]
ITEM_COLUMNS = ["item_id", "response_id", "case_id", "model", "line_type", "points", "criterion",
                "conversation", "reply", "verdict"]

# --------------------------------------------------------------------------------------------
# Perturbations (pure, deterministic given the seed)
# --------------------------------------------------------------------------------------------
OPENERS = ["Thank you for your question.", "I have read your message carefully.", "Here is my response."]
CLOSERS = ["I hope this response is useful.", "Thank you for reading.", "That is the end of my response."]
NEUTRAL_SENTENCES = OPENERS + CLOSERS
FILLER_SENTENCES = NEUTRAL_SENTENCES + [
    "I hope this helps.", "I hope this helps!", "Great question.", "Great question!",
    "Thanks for reaching out.", "Thanks for asking.", "Let me know if you have any other questions.",
    "Feel free to ask if you have more questions.",
]
_WORD = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
_BULLET = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+")
_HEADER = re.compile(r"^\s*#{1,6}\s+")


class ContentChangedError(ValueError):
    """A perturbation added or removed content words."""


def content_words(text: str) -> set[str]:
    """Lower-case word/number tokens (decimals and apostrophes kept: '2.4', "don't")."""
    return set(_WORD.findall((text or "").lower()))


def check_content(original: str, perturbed: str, allowed: Iterable[str] = ()) -> None:
    """Raise ContentChangedError unless the two texts have the same content words, up to the
    words of the `allowed` sentences (the padding / filler vocabulary)."""
    allow = set().union(*(content_words(s) for s in allowed)) if allowed else set()
    a, b = content_words(original), content_words(perturbed)
    added, removed = (b - a) - allow, (a - b) - allow
    if added or removed:
        raise ContentChangedError(f"content changed: added {sorted(added)[:5]}, removed {sorted(removed)[:5]}")


def pad(reply: str, seed: int = 0, n_before: int = 2, n_after: int = 2) -> str:
    """n_before OPENERS before and n_after CLOSERS after the reply (seeded choice, in list order)."""
    rng = np.random.default_rng(seed)
    before = " ".join(OPENERS[i] for i in sorted(rng.choice(len(OPENERS), size=n_before, replace=False)))
    after = " ".join(CLOSERS[i] for i in sorted(rng.choice(len(CLOSERS), size=n_after, replace=False)))
    out = "\n\n".join(p for p in (before, reply.strip(), after) if p)
    check_content(reply, out, NEUTRAL_SENTENCES)
    return out


def compress(reply: str, seed: int = 0) -> str:
    """Drop FILLER_SENTENCES (whole sentences, case-insensitive), collapse runs of spaces and blank lines."""
    fillers = {s.lower() for s in FILLER_SENTENCES}
    lines = []
    for line in reply.splitlines():
        parts = re.split(r"(?<=[.!?])\s+", line.strip())
        kept = [p for p in parts if p.lower() not in fillers]
        lines.append(re.sub(r"[ \t]+", " ", " ".join(kept)).strip() if kept else "")
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    check_content(reply, out, FILLER_SENTENCES)
    return out


def has_markdown(text: str) -> bool:
    """Headers, bullet/numbered items, or **bold** / __bold__ markers."""
    return any(_HEADER.match(ln) or _BULLET.match(ln) for ln in text.splitlines()) or bool(
        re.search(r"\*\*[^*]+\*\*|__[^_]+__", text))


def to_prose(text: str) -> str:
    """Markdown -> prose: drop header marks, bullets and emphasis; bullet items become sentences of one paragraph."""
    paras, current = [], []
    for ln in text.splitlines():
        s = _HEADER.sub("", ln)
        s = _BULLET.sub("", s)
        s = re.sub(r"\*\*([^*]+)\*\*|__([^_]+)__", lambda m: m.group(1) or m.group(2), s)
        s = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"\1", s).strip()
        if not s:
            if current:
                paras.append(" ".join(current))
                current = []
            continue
        if not re.search(r"[.!?:]$", s):
            s += "."
        current.append(s)
    if current:
        paras.append(" ".join(current))
    return "\n\n".join(paras)


def to_markdown(text: str) -> str:
    """Prose -> Markdown: every sentence becomes a '- ' bullet; paragraphs stay separated."""
    out = []
    for para in re.split(r"\n\s*\n", text.strip()):
        sents = [s for s in re.split(r"(?<=[.!?])\s+", para.strip()) if s]
        out.append("\n".join(f"- {s}" for s in sents))
    return "\n\n".join(out)


def format_swap(reply: str, seed: int = 0) -> str:
    """to_prose if the reply has Markdown, else to_markdown."""
    out = to_prose(reply) if has_markdown(reply) else to_markdown(reply)
    check_content(reply, out)
    return out


def _permute(units: list, rng: np.random.Generator) -> list:
    """A seeded permutation that differs from the input order when len >= 2 (rotate if identity)."""
    if len(units) < 2:
        return units
    order = list(rng.permutation(len(units)))
    if order == list(range(len(units))):
        order = order[1:] + order[:1]
    return [units[i] for i in order]


def reorder(reply: str, seed: int = 0) -> str:
    """Permute Markdown sections (by header), else bullet items within each list, else paragraphs."""
    rng = np.random.default_rng(seed)
    lines = reply.strip().splitlines()
    heads = [i for i, ln in enumerate(lines) if _HEADER.match(ln)]
    if len(heads) >= 2:
        pre = "\n".join(lines[:heads[0]]).strip()
        sections = ["\n".join(lines[a:b]).strip() for a, b in zip(heads, heads[1:] + [len(lines)])]
        out = "\n\n".join(([pre] if pre else []) + _permute(sections, rng))
    elif sum(bool(_BULLET.match(ln)) for ln in lines) >= 2:
        out_lines, block = [], []
        for ln in lines + [""]:
            if _BULLET.match(ln):
                block.append(ln)
                continue
            out_lines += _permute(block, rng)
            block = []
            out_lines.append(ln)
        out = "\n".join(out_lines[:-1])
    else:
        paras = [p for p in re.split(r"\n\s*\n", reply.strip()) if p.strip()]
        out = "\n\n".join(_permute(paras, rng))
    check_content(reply, out)
    return out


PERTURBATIONS: dict[str, Callable[[str, int], str]] = {
    "length_pad": pad, "length_compress": compress, "format": format_swap, "order": reorder}


# --------------------------------------------------------------------------------------------
# Audits
# --------------------------------------------------------------------------------------------
@dataclass
class AuditResult:
    """Verdict-change summary of one audit."""
    name: str
    n_items: int
    n_skipped: int          # perturbation left the reply unchanged
    n_judge_errors: int     # pairs dropped because a verdict was unreadable
    n_pairs: int
    yes_before: int
    yes_after: int
    yes_to_no: int          # b
    no_to_yes: int          # c
    change_rate: float
    change_ci: tuple
    p_mcnemar: float
    control_changes: Optional[int] = None   # re-judging the unchanged reply: the noise floor
    control_pairs: Optional[int] = None
    control_ci: Optional[tuple] = None

    def to_dict(self) -> dict:
        return asdict(self)


def summarise_pairs(name: str, before: list, after: list, n_items: int, n_skipped: int,
                    control: Optional[tuple[list, list]] = None) -> AuditResult:
    """AuditResult from paired verdict lists (None = unreadable, dropped and counted)."""
    pairs = [(x, y) for x, y in zip(before, after) if x is not None and y is not None]
    b = sum(x and not y for x, y in pairs)
    c = sum((not x) and y for x, y in pairs)
    n = len(pairs)
    res = AuditResult(name, n_items, n_skipped, len(before) - n, n,
                      sum(x for x, _ in pairs), sum(y for _, y in pairs), b, c,
                      (b + c) / n if n else float("nan"), wilson(b + c, n) if n else (float("nan"),) * 2,
                      mcnemar_exact(b, c))
    if control is not None:
        cp = [(x, y) for x, y in zip(*control) if x is not None and y is not None]
        k = sum(x != y for x, y in cp)
        res.control_changes, res.control_pairs = k, len(cp)
        res.control_ci = wilson(k, len(cp)) if cp else (float("nan"),) * 2
    return res


async def perturbation_audit(items: pd.DataFrame, judge: LineJudge, name: str, seed: int = 0,
                             rejudge_original: bool = True, control: bool = True,
                             perturb: Optional[Callable[[str, int], str]] = None) -> tuple[AuditResult, pd.DataFrame]:
    """Run one perturbation over every item; returns the summary and a per-item table.

    rejudge_original=False compares against the stored `verdict` column instead of a fresh judgement.
    """
    perturb = perturb or PERTURBATIONS[name]
    rows, skipped = [], 0
    for i, it in enumerate(items.itertuples(index=False)):
        new = perturb(it.reply, seed + i)
        if new.strip() == it.reply.strip():
            skipped += 1
            continue
        calls = [judge(it.conversation, new, it.criterion)]
        if rejudge_original:
            calls.append(judge(it.conversation, it.reply, it.criterion))
        if control:
            calls.append(judge(it.conversation, it.reply, it.criterion))
        got = await asyncio.gather(*calls)
        perturbed_v = got[0]
        original_v = got[1] if rejudge_original else it.verdict
        retest = got[-1] if control else None
        before, after = original_v, perturbed_v
        if name == "format" and has_markdown(it.reply):
            before, after = after, before          # orient prose -> markdown
        rows.append({"item_id": it.item_id, "before": before, "after": after, "original": original_v,
                     "retest": retest, "changed": None if before is None or after is None else before != after,
                     "perturbed_reply": new})
    cols = ["item_id", "before", "after", "original", "retest", "changed", "perturbed_reply"]
    detail = pd.DataFrame(rows, columns=cols)
    # the control re-judges the unchanged original reply: pair it with the original's verdict
    ctrl = (list(detail["original"]), list(detail["retest"])) if control else None
    res = summarise_pairs(name, list(detail["before"]), list(detail["after"]), len(items), skipped, ctrl)
    return res, detail


async def run_bias_audits(items: pd.DataFrame, judge: LineJudge, judge_model: str,
                          names: Iterable[str] = tuple(PERTURBATIONS), seed: int = 0,
                          **kwargs) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Independence check, then every named perturbation audit. Returns (summary table, details by name)."""
    assert_judge_independent(judge_model, sorted(set(items["model"])))
    results, details = [], {}
    for name in names:
        res, det = await perturbation_audit(items, judge, name, seed=seed, **kwargs)
        results.append(res.to_dict())
        details[name] = det
    return pd.DataFrame(results), details


# --------------------------------------------------------------------------------------------
# Self-preference
# --------------------------------------------------------------------------------------------
class SelfPreferenceError(AssertionError):
    """The judge favours its own family (significant McNemar, own-family advantage)."""

    def __init__(self, message: str, result: dict):
        super().__init__(message)
        self.result = result


def _favourable(verdict, line_type) -> Optional[bool]:
    """Judge says the line behaved well: verdict for 'must', not verdict for 'must_not'; None stays None."""
    if verdict is None or pd.isna(verdict):
        return None
    return bool(verdict) if line_type == "must" else not bool(verdict)


def self_preference_audit(scored: pd.DataFrame, judge_model: str, alpha: float = 0.05,
                          strict: bool = True) -> pd.DataFrame:
    """Own-family vs other-family verdicts of one judge, matched on (case_id, line_index).

    scored columns: case_id, line_index, line_type, model, verdict, and optionally human_label (bool,
    'criterion true' like verdict). Outcome per reply-line: with human labels, a FALSE PASS (judge favourable,
    human not); otherwise judge favourable. For each other family, pairs = (case, line) judged for both
    families; when a family has several replies for a pair, the first by model name is used. Reports k/n
    with Wilson CIs per side, b = own-only, c = other-only, exact McNemar p, and
    flagged = (p < alpha and b > c). strict=True raises SelfPreferenceError if any family is flagged.
    """
    judge_family = family_of(judge_model)
    df = scored.copy()
    df["family"] = [family_of(m) for m in df["model"]]
    if (df["family"] == judge_family).sum() == 0:
        raise JudgeIndependenceError(f"no replies from the judge's own family '{judge_family}': nothing to compare")
    use_human = "human_label" in df.columns and df["human_label"].notna().any()
    fav = [_favourable(v, t) for v, t in zip(df["verdict"], df["line_type"])]
    if use_human:
        hum = [_favourable(h, t) for h, t in zip(df["human_label"], df["line_type"])]
        df["outcome"] = [None if f is None or h is None else (f and not h) for f, h in zip(fav, hum)]
    else:
        df["outcome"] = fav
    df = df[df["outcome"].notna()].sort_values(["case_id", "line_index", "model"])
    first = df.groupby(["family", "case_id", "line_index"], sort=True)["outcome"].first().reset_index()
    own = first[first["family"] == judge_family].drop(columns="family")
    rows = []
    for fam in sorted(set(first["family"]) - {judge_family}):
        other = first[first["family"] == fam].drop(columns="family")
        m = own.merge(other, on=["case_id", "line_index"], suffixes=("_own", "_other"))
        xo, xt = m["outcome_own"].astype(bool), m["outcome_other"].astype(bool)
        n, b, c = len(m), int((xo & ~xt).sum()), int((~xo & xt).sum())
        p = mcnemar_exact(b, c)
        rows.append({"judge_family": judge_family, "other_family": fam,
                     "outcome": "false_pass" if use_human else "judge_pass", "n_pairs": n,
                     "k_own": int(xo.sum()), "k_other": int(xt.sum()),
                     "own_ci": wilson(int(xo.sum()), n) if n else None,
                     "other_ci": wilson(int(xt.sum()), n) if n else None,
                     "own_only": b, "other_only": c, "p_mcnemar": p, "flagged": bool(p < alpha and b > c)})
    out = pd.DataFrame(rows)
    if strict and len(out) and out["flagged"].any():
        bad = out[out["flagged"]].iloc[0]
        raise SelfPreferenceError(
            f"judge '{judge_model}' favours its own family '{judge_family}' over '{bad['other_family']}': "
            f"{bad['own_only']} own-only vs {bad['other_only']} other-only {bad['outcome']} pairs "
            f"(McNemar p = {bad['p_mcnemar']:.3g})", out.to_dict("records"))
    return out


# --------------------------------------------------------------------------------------------
# Inputs and the real judge (used by the CLI; tests use fakes)
# --------------------------------------------------------------------------------------------
def items_from_logs(log_dir: str | Path) -> pd.DataFrame:
    """ITEM_COLUMNS from every clinical_safety log: one row per (run, case, epoch, rubric line) with text."""
    from inspect_ai.log import read_eval_log

    from evals.run_eval import conversation_text
    rows = []
    for p in sorted(Path(log_dir).glob("*.eval")):
        log = read_eval_log(str(p))
        if "clinical" not in str(log.eval.task):
            continue
        for s in log.samples or []:
            sc = next(iter(s.scores.values()), None) if s.scores else None
            if not sc or not sc.metadata or "lines" not in sc.metadata:
                continue
            rid = f"{log.eval.eval_id}|{s.id}|e{s.epoch}"
            for i, ln in enumerate(sc.metadata["lines"]):
                rows.append({"item_id": f"{rid}|L{i}", "response_id": rid, "case_id": str(s.id),
                             "model": str(log.eval.model), "line_type": ln["type"], "points": ln["points"],
                             "criterion": ln["criterion"], "conversation": conversation_text(s.metadata["turns"]),
                             "reply": sc.answer or "", "verdict": ln.get("verdict")})
    return pd.DataFrame(rows, columns=ITEM_COLUMNS)


def inspect_line_judge(model: str) -> LineJudge:
    """The run_eval judge prompt and verdict parser on an Inspect model, as a LineJudge (temperature 0).
    The provider-reported model version of the last call is kept in `judge.last_version`."""
    from inspect_ai.model import GenerateConfig, get_model

    from evals.run_eval import JUDGE_PROMPT, parse_verdict, strip_think
    m = get_model(model, config=GenerateConfig(temperature=0, max_tokens=1500))

    async def judge(conversation: str, reply: str, criterion: str) -> Optional[bool]:
        out = await m.generate(JUDGE_PROMPT.format(conversation=conversation, reply=reply, criterion=criterion))
        judge.last_version = getattr(out, "model", None)
        return parse_verdict(strip_think(out.completion))

    judge.last_version = None
    return judge


def results_markdown(summary: pd.DataFrame) -> str:
    """Markdown table of run_bias_audits output: k/n changes with Wilson CI, direction, McNemar, noise floor."""
    out = ["| Audit | Pairs | Changed | YES→NO | NO→YES | McNemar p | Control (unchanged reply) | Skipped | "
           "Judge errors |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        k = r["yes_to_no"] + r["no_to_yes"]
        lo, hi = r["change_ci"]
        ctrl = "–"
        if r.get("control_pairs"):
            clo, chi = r["control_ci"]
            ctrl = f"{r['control_changes']}/{r['control_pairs']} (95% Wilson {clo:.0%}–{chi:.0%})"
        out.append(f"| {r['name']} | {r['n_pairs']} | {k}/{r['n_pairs']} (95% Wilson {lo:.0%}–{hi:.0%}) | "
                   f"{r['yes_to_no']} | {r['no_to_yes']} | {r['p_mcnemar']:.3g} | {ctrl} | {r['n_skipped']} | "
                   f"{r['n_judge_errors']} |")
    return "\n".join(out) + "\n"


def main(argv: Optional[list[str]] = None) -> None:
    """CLI: python evals/judge_validation/bias_audit.py LOGS --judge MODEL [--n 200] [--seed 0].
    Calls the judge model (paid API for hosted judges)."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("logs_dir")
    ap.add_argument("--judge", required=True)
    ap.add_argument("--n", type=int, default=200, help="items to audit (seeded sample)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    items = items_from_logs(a.logs_dir)
    items = items[items["verdict"].notna()]
    if len(items) > a.n:
        items = items.sample(n=a.n, random_state=a.seed)
    summary, _ = asyncio.run(run_bias_audits(items, inspect_line_judge(a.judge), a.judge, seed=a.seed))
    print(results_markdown(summary), end="")


if __name__ == "__main__":
    main()
