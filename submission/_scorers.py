"""
submission/_scorers.py -- scorer registry. each scorer is a vectorised
per-term contribution fn so one traversal can feed all of them

    term_contribution(tfs, doc_lens, df, cf, query_tf, stats, **params) -> ndarray
    doc_prior(doc_lens, query_len, stats, **params) -> ndarray | None
"""
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class CollectionStats:
    N: int
    avg_doc_len: float
    total_tokens: int


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
    params = dict(get(name).defaults)
    if overrides:
        unknown = set(overrides) - set(params)
        if unknown:
            raise KeyError(f"scorer {name!r} has no parameter(s) {sorted(unknown)}")
        params.update(overrides)
    return params


def robertson_idf(df: int, N: int) -> float:
    """IDF = ln( (N - df + 0.5) / (df + 0.5) + 1 )"""
    return float(np.log((N - df + 0.5) / (df + 0.5) + 1.0))


@register(
    "bm25",
    defaults={"k1": 1.2, "b": 0.75},
    description="Okapi BM25 (Robertson & Zaragoza 2009).",
)
def bm25_contribution(tfs, doc_lens, df, cf, query_tf, stats: CollectionStats, k1=1.2, b=0.75):
    """IDF(q) * [ tf*(k1+1) ] / [ tf + k1*(1 - b + b*dl/avgdl) ]"""
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
    """bm25 with a delta floor added, fixes long doc over-penalisation"""
    tf = tfs.astype(np.float64)
    avgdl = stats.avg_doc_len or 1.0
    norm = k1 * (1.0 - b + b * (doc_lens / avgdl))
    return robertson_idf(df, stats.N) * ((tf * (k1 + 1.0)) / (tf + norm) + delta)


def _lm_dirichlet_prior(doc_lens, query_len, stats: CollectionStats, mu=1500.0):
    """|Q| * log( mu / (dl + mu) ), per doc term"""
    return query_len * np.log(mu / (doc_lens + mu))


# DFR scorers below, ported from Terrier's PL2.java/DPH.java
_LOG2_E = float(np.log2(np.e))
_LOG2_2PI = float(np.log2(2.0 * np.pi))


def _log2(x):
    return np.log2(x)


@register(
    "pl2",
    defaults={"c": 1.0},
    description="DFR PL2: Poisson model, Laplace after-effect, Normalisation 2.",
)
def pl2_contribution(tfs, doc_lens, df, cf, query_tf, stats: CollectionStats, c=1.0):
    """tfn = tf*log2(1 + c*avgdl/dl), then poisson+laplace+norm2 formula"""
    tf = tfs.astype(np.float64)
    avgdl = stats.avg_doc_len or 1.0
    dl = np.maximum(doc_lens, 1.0)
    tfn = tf * _log2(1.0 + c * avgdl / dl)

    out = np.zeros(tf.shape, dtype=np.float64)
    ok = tfn > 0.0  # tfn<=0 makes the logs undefined
    if not ok.any():
        return out

    t = tfn[ok]
    lam = cf / stats.N if stats.N else 0.0
    if lam <= 0.0:
        lam = 1.0 / max(stats.N, 1)
    out[ok] = query_tf * (1.0 / (t + 1.0)) * (
        t * _log2(t / lam)
        + (lam + 1.0 / (12.0 * t) - t) * _LOG2_E
        + 0.5 * (_LOG2_2PI + _log2(t))
    )
    return out


@register(
    "dph",
    defaults={},
    description="DFR DPH: hypergeometric, parameter-free (no tuning knob at all).",
)
def dph_contribution(tfs, doc_lens, df, cf, query_tf, stats: CollectionStats):
    """parameter free dfr scorer, formula from terrier"""
    tf = tfs.astype(np.float64)
    avgdl = stats.avg_doc_len or 1.0
    dl = np.maximum(doc_lens, 1.0)
    f = tf / dl

    out = np.zeros(tf.shape, dtype=np.float64)
    ok = (f < 1.0) & (tf > 0.0)  # f>=1 = doc is nothing but this term, skip
    if not ok.any():
        return out

    t, ff, d = tf[ok], f[ok], dl[ok]
    norm = (1.0 - ff) ** 2 / (t + 1.0)
    inv_p = (t * avgdl / d) * (stats.N / max(cf, 1))
    inv_p = np.maximum(inv_p, 1.0 + 1e-12)
    out[ok] = query_tf * norm * (
        t * _log2(inv_p) + 0.5 * (_LOG2_2PI + _log2(t * (1.0 - ff)))
    )
    return out


@register(
    "lmd",
    defaults={"mu": 1500.0},
    doc_prior=_lm_dirichlet_prior,
    description="Query-likelihood LM with Dirichlet smoothing (Zhai & Lafferty 2001).",
)
def lm_dirichlet_contribution(tfs, doc_lens, df, cf, query_tf, stats: CollectionStats, mu=1500.0):
    """qtf * log(1 + tf/(mu*p(t|C))), p(t|C) = cf/total_tokens"""
    tf = tfs.astype(np.float64)
    total = stats.total_tokens or 1
    p_collection = cf / total
    if p_collection <= 0.0:
        p_collection = 1.0 / total
    return query_tf * np.log1p(tf / (mu * p_collection))
