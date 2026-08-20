"""
submission/_traverse.py — one postings traversal, N scorers.

The rule this module exists to enforce: **decode each query term's postings
exactly once**, then hand the decoded arrays to every active scorer. Running
four rankers must not cost four traversals, or plan.md Section 5.0's
"fuse, don't select" strategy would be unaffordable at query time.

Current candidate generation is a full accumulate over each query term's
postings (a "term-at-a-time" scan). That is already fast -- a ~12-token query on
this collection touches a few hundred thousand postings, decoded with vectorised
NumPy -- and, critically, it is *exact*: it produces the true top-k, so it can
serve as the correctness oracle for the WAND / BlockMax-WAND early-termination
path scheduled for 23 Aug (plan.md Section 7). Optimise against a known-correct
baseline, not instead of one.
"""
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from submission import _scorers
from submission._analysis import analyze
from submission._scorers import CollectionStats

# alias -> (scorer_name, param overrides)
ScorerSpec = Dict[str, Tuple[str, Optional[Dict[str, float]]]]


def collection_stats(index) -> CollectionStats:
    return CollectionStats(
        N=index.N,
        avg_doc_len=index.avg_doc_len,
        total_tokens=index.total_tokens,
    )


def query_terms(index, query: str) -> List[Tuple[str, int]]:
    """Analyse `query` with the *index's own* persisted config and return
    (term, query_term_frequency) pairs.

    Using the index's config rather than the module default is what guarantees a
    query can never be tokenised differently from the corpus it runs against.
    """
    tokens = analyze(query, index.config)
    return list(Counter(tokens).items())


def score_query(
    index,
    query: str,
    specs: ScorerSpec,
    k: int = 10,
) -> Dict[str, List[Tuple[str, float]]]:
    """Score `query` with several scorers over a single postings traversal.

    Returns {alias: [(external_doc_id, score), ...]}, each list sorted by score
    descending and truncated to `k`.
    """
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
            continue  # out-of-vocabulary term contributes nothing
        doc_ids, tfs = index.postings_by_id(tid)
        if doc_ids.size == 0:
            continue
        doc_lens = index.doc_len[doc_ids].astype(np.float64)
        df = int(index.df[tid])
        cf = int(index.cf[tid])
        touched[doc_ids] = True

        # The payoff: postings decoded once above, reused by every scorer here.
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
    """Convenience wrapper for scoring with exactly one scorer."""
    return score_query(index, query, {"_": (scorer_name, params or None)}, k)["_"]


def _top_k(index, candidates: np.ndarray, scores: np.ndarray, k: int) -> List[Tuple[str, float]]:
    """Top-k with a deterministic tie-break.

    Ties break on ascending internal document id (i.e. corpus order), which is
    stable across runs and across processes. The interface contract requires
    determinism, and an arbitrary-but-consistent tie-break is what delivers it.
    """
    if candidates.size == 0 or k <= 0:
        return []
    cand_scores = scores[candidates]
    if candidates.size > k:
        # argpartition is O(n) vs O(n log n) for a full sort; we only need the
        # top k, and the exact order within them is fixed by the lexsort below.
        top = np.argpartition(-cand_scores, k - 1)[:k]
        candidates = candidates[top]
        cand_scores = cand_scores[top]
    order = np.lexsort((candidates, -cand_scores))
    return [(index.doc_ids[int(candidates[i])], float(cand_scores[i])) for i in order]


def rrf_fuse(
    runs: Sequence[List[Tuple[str, float]]],
    k: int = 10,
    rrf_k: float = 60.0,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion (Cormack et al. 2009):

        score(d) = sum_r  w_r / (rrf_k + rank_r(d))

    Rank-based, so it is immune to the score-scale mismatch between BM25 and a
    log-probability language model -- which is exactly why plan.md Section 5.0
    makes it the default fusion method rather than weighted score summation.

    Ties break on document id so the output is deterministic regardless of the
    input runs' internal ordering.
    """
    if weights is None:
        weights = [1.0] * len(runs)
    fused: Dict[str, float] = {}
    for run, weight in zip(runs, weights):
        for rank, (doc_id, _score) in enumerate(run, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (rrf_k + rank)
    ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:k]
