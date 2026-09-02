"""
submission/_traverse.py -- decode postings once per query term, feed to all
active scorers so N scorers = 1 traversal not N traversals
"""
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from submission import _scorers
from submission._analysis import analyze
from submission._scorers import CollectionStats

ScorerSpec = Dict[str, Tuple[str, Optional[Dict[str, float]]]]


def collection_stats(index) -> CollectionStats:
    return CollectionStats(
        N=index.N,
        avg_doc_len=index.avg_doc_len,
        total_tokens=index.total_tokens,
    )


def query_terms(index, query: str) -> List[Tuple[str, int]]:
    """uses index's own saved config, not module default, so query never
    gets tokenised differently than the corpus"""
    tokens = analyze(query, index.config)
    return list(Counter(tokens).items())


def score_query(
    index,
    query: str,
    specs: ScorerSpec,
    k: int = 10,
) -> Dict[str, List[Tuple[str, float]]]:
    """score with multiple scorers over one traversal, returns
    {alias: [(doc_id, score), ...]} sorted desc, truncated to k"""
    resolved = {
        alias: (_scorers.get(name), _scorers.resolve_params(name, overrides))
        for alias, (name, overrides) in specs.items()
    }
    stats = collection_stats(index)
    terms = query_terms(index, query)

    n_docs = index.N
    accum = {alias: np.zeros(n_docs, dtype=np.float64) for alias in resolved}
    touched = np.zeros(n_docs, dtype=bool)
    query_len = sum(qtf for _term, qtf in terms)

    for term, query_tf in terms:
        tid = index.term_id(term)
        if tid < 0:
            continue
        doc_ids, tfs = index.postings_by_id(tid)
        if doc_ids.size == 0:
            continue
        doc_lens = index.doc_len[doc_ids].astype(np.float64)
        df = int(index.df[tid])
        cf = int(index.cf[tid])
        touched[doc_ids] = True

        for alias, (scorer, params) in resolved.items():
            accum[alias][doc_ids] += scorer.term_contribution(
                tfs, doc_lens, df, cf, query_tf, stats, **params
            )

    candidates = np.flatnonzero(touched)
    out: Dict[str, List[Tuple[str, float]]] = {}
    for alias, (scorer, params) in resolved.items():
        scores = accum[alias]
        if scorer.doc_prior is not None and candidates.size:
            prior_params = {p: v for p, v in params.items()
                            if p in scorer.doc_prior.__code__.co_varnames}
            scores[candidates] += scorer.doc_prior(
                index.doc_len[candidates].astype(np.float64), query_len, stats, **prior_params
            )
        out[alias] = _top_k(index, candidates, scores, k)
    return out


def score_single(index, query: str, scorer_name: str, k: int = 10, **params):
    return score_query(index, query, {"_": (scorer_name, params or None)}, k)["_"]


def _top_k(index, candidates: np.ndarray, scores: np.ndarray, k: int) -> List[Tuple[str, float]]:
    """top-k, ties broken on ascending doc id for determinism"""
    if candidates.size == 0 or k <= 0:
        return []
    cand_scores = scores[candidates]
    # full lexsort not argpartition -- argpartition ties break arbitrarily,
    # disagreed w/ the C kernel on a few dev topics because of it
    order = np.lexsort((candidates, -cand_scores))[:k]
    return [(index.doc_ids[int(candidates[i])], float(cand_scores[i])) for i in order]


def rrf_fuse(
    runs: Sequence[List[Tuple[str, float]]],
    k: int = 10,
    rrf_k: float = 60.0,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[str, float]]:
    """reciprocal rank fusion: score(d) = sum_r w_r / (rrf_k + rank_r(d)).
    rank based so it doesn't care about score-scale mismatch between scorers"""
    if weights is None:
        weights = [1.0] * len(runs)
    fused: Dict[str, float] = {}
    for run, weight in zip(runs, weights):
        for rank, (doc_id, _score) in enumerate(run, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (rrf_k + rank)
    ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:k]
