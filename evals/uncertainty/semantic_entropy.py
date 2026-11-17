"""Semantic entropy of sampled replies (EXPLORATORY analysis layer for locally run open-weight models).

Method: Farquhar et al. 2024 (semantic entropy); clinical use: arXiv:2503.00269 (cited as given by the repo owner,
not re-checked here). For each case, draw N replies at temperature T (default N=10, T=1), group the replies that
mean the same thing, and measure how spread out the groups are. Many different meanings -> high entropy -> the
model is unsure what to say.

    clustering   two replies a, b share a cluster iff entails(a, b) AND entails(b, a)  (bidirectional entailment),
                 checked greedily against the first member of each existing cluster. Transitivity is not enforced.
    discrete SE  = - sum_c p_c ln p_c,  p_c = (replies in cluster c) / N.   Range 0 .. ln N.
    baseline     sequence-log-probability score: mean over replies of the per-token mean negative log-probability
                 (higher = less certain). It needs token log-probabilities from the sampler.

Nothing here calls a model: the sampler and the entailment judge are injected callables (a local NLI model, a
judge model, or a fake in tests). ``entails`` returning None (unreadable) is treated as "not entailed", which can
only split clusters and raise entropy; the number of unreadable calls is reported.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

EXPLORATORY = "exploratory"


@dataclass(frozen=True)
class Sample:
    """One sampled reply. ``token_logprobs`` are the natural-log probabilities of the generated tokens, if known."""
    text: str
    token_logprobs: tuple[float, ...] | None = None


Sampler = Callable[[str, int, float], "list[Sample]"]     # (prompt, n, temperature) -> n samples
# (question, premise, hypothesis) -> does the premise entail the hypothesis?
Entails = Callable[[str, str, str], "bool | None"]


@dataclass
class ClusterResult:
    labels: list[int]
    n_clusters: int
    n_unknown: int          # entailment calls that returned None


def cluster_by_entailment(question: str, responses: list[str], entails: Entails) -> ClusterResult:
    """Cluster replies by bidirectional entailment.

    Each reply joins the first cluster whose first member it matches in both directions, else starts a new cluster.
    Example: ["yes, go", "go, yes", "wait"] with the first two equivalent -> labels [0, 0, 1].
    """
    reps: list[str] = []
    labels: list[int] = []
    unknown = 0
    for r in responses:
        placed = None
        for ci, rep in enumerate(reps):
            fwd, bwd = entails(question, rep, r), entails(question, r, rep)
            unknown += (fwd is None) + (bwd is None)
            if fwd is True and bwd is True:
                placed = ci
                break
        if placed is None:
            reps.append(r)
            placed = len(reps) - 1
        labels.append(placed)
    return ClusterResult(labels, len(reps), unknown)


def discrete_semantic_entropy(labels: list[int]) -> float:
    """Discrete semantic entropy -sum_c p_c ln p_c (natural log), p_c = count_c / N.

    Example: cluster counts [5, 3, 2] of N=10 -> -(0.5 ln 0.5 + 0.3 ln 0.3 + 0.2 ln 0.2) = 1.0297; one cluster -> 0.
    """
    n = len(labels)
    if n == 0:
        raise ValueError("need at least one label")
    return float(-sum((c / n) * math.log(c / n) for c in Counter(labels).values()))


def sequence_logprob_score(samples: list[Sample], length_normalize: bool = True) -> float:
    """Baseline uncertainty from token log-probabilities: higher = less certain.

        per reply: mean of -logp over its tokens  (or the sum if length_normalize=False)
        score    : mean over the N replies
    Example: token log-probs [[-0.5, -1.5], [-1.0]] -> per-reply 1.0 and 1.0 -> 1.0. NaN if any reply lacks them.
    """
    vals = []
    for s in samples:
        lp = s.token_logprobs
        if not lp:
            return math.nan
        neg = [-x for x in lp]
        vals.append(sum(neg) / len(neg) if length_normalize else sum(neg))
    return float(sum(vals) / len(vals)) if vals else math.nan


@dataclass
class CaseScore:
    semantic_entropy: float
    logprob_score: float
    n_samples: int
    n_clusters: int
    n_unknown: int
    temperature: float
    responses: list[str] = field(default_factory=list)
    label: str = EXPLORATORY


def score_case(prompt: str, sampler: Sampler, entails: Entails, n: int = 10, temperature: float = 1.0) -> CaseScore:
    """Sample ``n`` replies at ``temperature`` and compute semantic entropy and the log-probability baseline.

    ``sampler(prompt, n, temperature)`` must return exactly n Samples. The same ``prompt`` is used as the question
    for the entailment judge.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    samples = list(sampler(prompt, n, temperature))
    if len(samples) != n:
        raise ValueError(f"sampler returned {len(samples)} samples, expected {n}")
    cl = cluster_by_entailment(prompt, [s.text for s in samples], entails)
    return CaseScore(discrete_semantic_entropy(cl.labels), sequence_logprob_score(samples), n, cl.n_clusters,
                     cl.n_unknown, temperature, [s.text for s in samples])
