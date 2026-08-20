"""
submission/_scorers.py — the scorer registry (plan.md Section 4.4).

Every ranking function in this submission is expressed as a pure, vectorised
contribution over one query term's postings list. That buys three things:

  1. One postings traversal serves N scorers (see submission/_traverse.py), so
     fusing several rankers costs barely more than running one.
  2. Adding a scorer is ~20 lines, which is what makes plan.md Section 5.0's
     "fuse, don't select" strategy affordable at all.
  3. Each scorer is independently unit-testable against a hand-computed example
     (assignment Section 7: "graded by unit tests against known small examples").

Interface
---------
A scorer supplies:

    term_contribution(tfs, doc_lens, df, cf, query_tf, stats, **params) -> ndarray
        Score contribution of ONE query term to each document in its postings
        list. `tfs` and `doc_lens` are parallel arrays over that postings list.

    doc_prior(doc_lens, query_len, stats, **params) -> ndarray | None
        Optional per-document term-independent term, applied once to candidate
        documents. Language models need this (the smoothing normaliser); BM25
        does not and returns None.

Parameters are always explicit keyword arguments with defaults declared in the
registry -- never constants captured in the function body. The assignment
requires k1/b to be tunable, and the oral defense perturbs exactly these.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class CollectionStats:
    """Corpus-level statistics every scorer may read."""
    N: int                    # number of documents
    avg_doc_len: float        # mean document length in tokens
    total_tokens: int         # total tokens in the collection (LM/DFR need this)


@dataclass(frozen=True)
class Scorer:
    name: str
    term_contribution: Callable
    defaults: Dict[str, float]
    doc_prior: Optional[Callable] = None
    description: str = ""


_REGISTRY: Dict[str, Scorer] = {}


def register(name: str, defaults: Dict[str, float], doc_prior=None, description: str = ""):
    def wrap(fn):
        _REGISTRY[name] = Scorer(name, fn, dict(defaults), doc_prior, description)
        return fn
    return wrap


def get(name: str) -> Scorer:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown scorer {name!r}; available: {sorted(_REGISTRY)}") from None


def available() -> Dict[str, Scorer]:
    return dict(_REGISTRY)


def resolve_params(name: str, overrides: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Merge caller overrides onto a scorer's declared defaults."""
    params = dict(get(name).defaults)
    if overrides:
        unknown = set(overrides) - set(params)
        if unknown:
            raise KeyError(f"scorer {name!r} has no parameter(s) {sorted(unknown)}")
        params.update(overrides)
    return params


# ---------------------------------------------------------------------------
# IDF
# ---------------------------------------------------------------------------

def robertson_idf(df: int, N: int) -> float:
    """Robertson-Sparck Jones IDF, +1-smoothed so it stays non-negative even for
    terms appearing in more than half the collection:

        IDF = ln( (N - df + 0.5) / (df + 0.5) + 1 )

    This is the form given in the assignment's bm25.py docstring.
    """
    return float(np.log((N - df + 0.5) / (df + 0.5) + 1.0))


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

@register(
    "bm25",
    defaults={"k1": 1.2, "b": 0.75},
    description="Okapi BM25 (Robertson & Zaragoza 2009).",
)
def bm25_contribution(tfs, doc_lens, df, cf, query_tf, stats: CollectionStats, k1=1.2, b=0.75):
    """
        IDF(q) * [ tf * (k1 + 1) ] / [ tf + k1 * (1 - b + b * dl / avgdl) ]

    k1 controls term-frequency saturation, b controls length normalisation.
    """
    tf = tfs.astype(np.float64)
    avgdl = stats.avg_doc_len or 1.0
    norm = k1 * (1.0 - b + b * (doc_lens / avgdl))
    return robertson_idf(df, stats.N) * (tf * (k1 + 1.0)) / (tf + norm)


@register(
    "bm25plus",
    defaults={"k1": 1.2, "b": 0.75, "delta": 1.0},
    description="BM25+ (Lv & Zhai 2011): lower-bounds the normalised tf.",
)
def bm25plus_contribution(tfs, doc_lens, df, cf, query_tf, stats: CollectionStats,
                          k1=1.2, b=0.75, delta=1.0):
    """BM25 with a constant floor added to the normalised term frequency:

        IDF(q) * ( [ tf * (k1+1) ] / [ tf + k1*(1 - b + b*dl/avgdl) ] + delta )

    Fixes BM25's over-penalisation of long documents -- a matching term in a
    long document can otherwise score arbitrarily close to zero. Relevant here
    because this collection's lengths are strongly bimodal (notes/findings.md
    F2), which is exactly the regime the delta floor was designed for.
    """
    tf = tfs.astype(np.float64)
    avgdl = stats.avg_doc_len or 1.0
    norm = k1 * (1.0 - b + b * (doc_lens / avgdl))
    return robertson_idf(df, stats.N) * ((tf * (k1 + 1.0)) / (tf + norm) + delta)


# ---------------------------------------------------------------------------
# Language model with Dirichlet smoothing
# ---------------------------------------------------------------------------

def _lm_dirichlet_prior(doc_lens, query_len, stats: CollectionStats, mu=1500.0):
    """The term-independent half of the Dirichlet-smoothed query likelihood:

        |Q| * log( mu / (|D| + mu) )

    Applied once per candidate document. Safe for empty documents (dl = 0 gives
    log(1) = 0) -- this collection has 8 of them (notes/findings.md F2).
    """
    return query_len * np.log(mu / (doc_lens + mu))


@register(
    "lmd",
    defaults={"mu": 1500.0},
    doc_prior=_lm_dirichlet_prior,
    description="Query-likelihood LM with Dirichlet smoothing (Zhai & Lafferty 2001).",
)
def lm_dirichlet_contribution(tfs, doc_lens, df, cf, query_tf, stats: CollectionStats, mu=1500.0):
    """
        qtf * log( 1 + tf / (mu * p(t|C)) ),    p(t|C) = cf / total_tokens

    Paired with `_lm_dirichlet_prior` above, this is the standard rearrangement
    of Dirichlet-smoothed query likelihood into a term-matched part plus a
    document-length normaliser, so only documents containing a query term need
    to be visited.

    Ranks differently from BM25 on the same index, which is precisely why it
    earns its place: plan.md Section 5.0 wants decorrelated runs to fuse, not a
    marginally better single scorer.
    """
    tf = tfs.astype(np.float64)
    total = stats.total_tokens or 1
    p_collection = cf / total
    if p_collection <= 0.0:
        # Term is in the dictionary but has zero collection frequency, which
        # should be impossible; fall back to a floor rather than dividing by 0.
        p_collection = 1.0 / total
    return query_tf * np.log1p(tf / (mu * p_collection))
