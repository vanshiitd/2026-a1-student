#!/usr/bin/env python
"""
experiments/evaluate.py — fast in-process evaluation for parameter sweeps.

The grading harness spawns two subprocesses and rebuilds nothing between runs,
which is right for grading and far too slow for sweeping hundreds of parameter
settings. This module evaluates configurations in-process against a prebuilt
index, while computing metrics with `harness.metrics` -- the *same* code that
produces the real score, so a sweep number and a harness number are directly
comparable. `verify_against_harness()` asserts that equivalence rather than
assuming it.

The key trick is `QueryPostings`: for a fixed query, the decoded postings are
identical across every (k1, b) setting, so decode once and reuse. Scoring a
configuration then costs pure arithmetic. Postings are cached one query at a
time -- outer loop over queries, inner loop over configurations -- so peak
memory stays bounded by the single largest query rather than the whole dev set.

Statistics live here too (`paired_bootstrap`, `cv_folds`), because the analysis
that decides what to ship is part of the experiment, not an afterthought.
"""
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.metrics import evaluate_run, ndcg_at_k
from harness.trec_io import read_qrels, read_queries
from submission import _scorers
from submission._scorers import CollectionStats
from submission._traverse import query_terms
from submission.indexer import InvertedIndex

REPO = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_DATA = os.path.join(REPO, "data", "full")
DEFAULT_INDEX = os.path.join(REPO, ".index_cache")


# ---------------------------------------------------------------------------
# Index / data loading
# ---------------------------------------------------------------------------

def config_slug(config) -> str:
    """Short stable name for an analysis config, used for index cache dirs."""
    if config is None:
        return "default"
    d = config.to_dict()
    parts = []
    if d.get("stemmer"):
        parts.append(str(d["stemmer"]))
    if d.get("remove_stopwords"):
        parts.append("nostop")
    if d.get("split_alphanum"):
        parts.append("splitan")
    if d.get("min_token_len", 1) != 1:
        parts.append(f"minlen{d['min_token_len']}")
    return "-".join(parts) if parts else "plain"


def get_index(corpus_path: str, index_dir: str = DEFAULT_INDEX, rebuild: bool = False,
              config=None):
    """Build the index once per analysis config and reuse it across sweeps.

    Scoring sweeps vary only *scoring* parameters, which never change the index,
    so rebuilding per configuration would waste ~11s each time. The analysis
    chain does change it, so each config gets its own cache directory keyed by
    `config_slug()` -- otherwise a stemmed sweep would silently read an
    unstemmed index and every conclusion drawn from it would be wrong.
    """
    if config is not None:
        index_dir = f"{index_dir}-{config_slug(config)}"
    meta = os.path.join(index_dir, "meta.json")
    if rebuild or not os.path.exists(meta):
        print(f"building index -> {os.path.basename(index_dir)} ...", flush=True)
        os.makedirs(index_dir, exist_ok=True)
        index = InvertedIndex(config)
        index.build_from_jsonl(corpus_path)
        index.save(index_dir)
        return index
    loaded = InvertedIndex.load(index_dir)
    if config is not None and loaded.config != config:
        raise RuntimeError(
            f"cached index at {index_dir} was built with {loaded.config}, "
            f"not {config}; delete it and rebuild"
        )
    return loaded


def load_topics(data_dir: str = DEFAULT_DATA):
    """Return (queries, qrels) for a dev directory."""
    queries = read_queries(os.path.join(data_dir, "queries_dev.tsv"))
    qrels = read_qrels(os.path.join(data_dir, "qrels_dev.txt"))
    return queries, qrels


# ---------------------------------------------------------------------------
# Cached postings for one query
# ---------------------------------------------------------------------------

@dataclass
class _Term:
    doc_ids: np.ndarray
    tfs: np.ndarray
    doc_lens: np.ndarray
    df: int
    cf: int
    query_tf: int


class QueryPostings:
    """Decoded postings for a single query, reusable across configurations."""

    def __init__(self, index, query: str):
        self.index = index
        self.terms: List[_Term] = []
        self.query_len = 0
        for term, qtf in query_terms(index, query):
            self.query_len += qtf
            tid = index.term_id(term)
            if tid < 0:
                continue
            doc_ids, tfs = index.postings_by_id(tid)
            if doc_ids.size == 0:
                continue
            self.terms.append(_Term(
                doc_ids=doc_ids,
                tfs=tfs,
                doc_lens=index.doc_len[doc_ids].astype(np.float64),
                df=int(index.df[tid]),
                cf=int(index.cf[tid]),
                query_tf=qtf,
            ))

    def rank(self, scorer, params: Dict[str, float], stats: CollectionStats,
             k: int = 10) -> List[Tuple[str, float]]:
        """Rank with one scorer configuration. Mirrors submission/_traverse.py's
        semantics exactly, including the deterministic tie-break."""
        index = self.index
        if not self.terms:
            return []
        scores = np.zeros(index.N, dtype=np.float64)
        touched = np.zeros(index.N, dtype=bool)
        for t in self.terms:
            scores[t.doc_ids] += scorer.term_contribution(
                t.tfs, t.doc_lens, t.df, t.cf, t.query_tf, stats, **params
            )
            touched[t.doc_ids] = True

        candidates = np.flatnonzero(touched)
        if candidates.size == 0:
            return []
        if scorer.doc_prior is not None:
            prior_params = {p: v for p, v in params.items()
                            if p in scorer.doc_prior.__code__.co_varnames}
            scores[candidates] += scorer.doc_prior(
                index.doc_len[candidates].astype(np.float64), self.query_len,
                stats, **prior_params
            )
        cand_scores = scores[candidates]
        if candidates.size > k:
            top = np.argpartition(-cand_scores, k - 1)[:k]
            candidates, cand_scores = candidates[top], cand_scores[top]
        order = np.lexsort((candidates, -cand_scores))
        return [(index.doc_ids[int(candidates[i])], float(cand_scores[i])) for i in order]


# ---------------------------------------------------------------------------
# Sweeping
# ---------------------------------------------------------------------------

def sweep(index, queries, qrels, scorer_name: str,
          configs: Sequence[Dict[str, float]], k: int = 10) -> List[Dict]:
    """Evaluate many parameter settings for one scorer.

    Loops queries on the outside and configurations on the inside so each
    query's postings are decoded exactly once for the whole sweep, and only one
    query's postings are resident at a time.

    Returns one row per configuration, each carrying per-query nDCG@10 so that
    paired significance tests can be run afterwards without re-running anything.
    """
    scorer = _scorers.get(scorer_name)
    stats = CollectionStats(index.N, index.avg_doc_len, index.total_tokens)
    resolved = [_scorers.resolve_params(scorer_name, c) for c in configs]

    per_query: List[Dict[str, float]] = [{} for _ in resolved]
    for qid, text in queries:
        if qid not in qrels:
            continue
        postings = QueryPostings(index, text)
        judged = qrels[qid]
        for i, params in enumerate(resolved):
            ranked = postings.rank(scorer, params, stats, k=k)
            per_query[i][qid] = ndcg_at_k([d for d, _ in ranked], judged, k=10)

    rows = []
    for params, scores in zip(resolved, per_query):
        values = np.array(list(scores.values()), dtype=np.float64)
        rows.append({
            "scorer": scorer_name,
            "params": params,
            "ndcg@10": float(values.mean()) if values.size else 0.0,
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "se": float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0,
            "n_topics": int(values.size),
            "per_query": scores,
        })
    return rows


def run_to_metrics(index, queries, qrels, scorer_name: str,
                   params: Dict[str, float], k: int = 10) -> Dict:
    """Full metric set (nDCG@10, MAP@10, MRR, P@k) for one configuration, via
    the harness's own `evaluate_run` -- used for reporting, not sweeping."""
    scorer = _scorers.get(scorer_name)
    stats = CollectionStats(index.N, index.avg_doc_len, index.total_tokens)
    resolved = _scorers.resolve_params(scorer_name, params)
    run = {qid: QueryPostings(index, text).rank(scorer, resolved, stats, k=k)
           for qid, text in queries}
    return evaluate_run(run, qrels, k=k)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def paired_bootstrap(a: Dict[str, float], b: Dict[str, float],
                     n_resamples: int = 10_000, seed: int = 0) -> Dict[str, float]:
    """Two-sided paired bootstrap on per-query scores.

    Paired, because the two systems are evaluated on the *same* topics and their
    per-query scores are strongly correlated -- an unpaired comparison throws
    that correlation away and is far less sensitive. On 50 topics that
    difference decides whether a real effect is detectable at all.

    Returns the observed mean difference (a - b), the paired standard error, and
    a p-value for H0: no difference.
    """
    qids = sorted(set(a) & set(b))
    if not qids:
        return {"delta": 0.0, "p_value": 1.0, "paired_se": 0.0, "n": 0}
    diffs = np.array([a[q] - b[q] for q in qids], dtype=np.float64)
    observed = float(diffs.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_resamples, diffs.size))
    means = diffs[idx].mean(axis=1)
    # Centre under H0 (mean difference zero), then ask how often the resampled
    # statistic is at least as extreme as what we actually observed.
    centred = means - observed
    p = float((np.abs(centred) >= abs(observed)).mean())

    return {
        "delta": observed,
        "p_value": p,
        "paired_se": float(diffs.std(ddof=1) / np.sqrt(diffs.size)) if diffs.size > 1 else 0.0,
        "n": int(diffs.size),
        "wins": int((diffs > 0).sum()),
        "losses": int((diffs < 0).sum()),
        "ties": int((diffs == 0).sum()),
    }


def cv_folds(qids: Sequence[str], n_folds: int = 5, seed: int = 20260821) -> List[List[str]]:
    """Deterministic topic-level folds.

    The seed is fixed and committed so every experiment in the project uses the
    identical split: comparing two configurations across *different* folds would
    confound the comparison with fold difficulty.
    """
    ordered = sorted(qids)
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(ordered))
    return [shuffled[i::n_folds] for i in range(n_folds)]


def select_plateau(rows: List[Dict], param: str, tolerance_se: float = 1.0) -> Dict:
    """Pick the centre of the best-performing *plateau*, not the argmax.

    Every configuration within `tolerance_se` paired standard errors of the best
    is statistically indistinguishable from it, so taking the single peak is
    fitting noise -- and `max` over noisy estimates is biased upward. Taking the
    midpoint of the widest contiguous near-optimal region instead lands on a
    setting whose neighbours are also good, which is what survives a change of
    topic set.
    """
    if not rows:
        raise ValueError("no rows to select from")
    ordered = sorted(rows, key=lambda r: r["params"][param])
    values = np.array([r["ndcg@10"] for r in ordered])
    best = values.max()
    threshold = best - tolerance_se * float(np.median([r["se"] for r in ordered]))
    near = values >= threshold

    best_run, best_start, cur_start = 0, 0, None
    for i, ok in enumerate(near):
        if ok and cur_start is None:
            cur_start = i
        if (not ok or i == len(near) - 1) and cur_start is not None:
            end = i if not ok else i + 1
            if end - cur_start > best_run:
                best_run, best_start = end - cur_start, cur_start
            cur_start = None

    chosen = ordered[best_start + best_run // 2]
    return {
        "chosen": chosen,
        "argmax": ordered[int(values.argmax())],
        "plateau_size": best_run,
        "plateau_range": (ordered[best_start]["params"][param],
                          ordered[best_start + best_run - 1]["params"][param]),
        "threshold": float(threshold),
    }


def smooth_surface(rows: List[Dict], axes: Sequence[str],
                   scores: Optional[Dict[Tuple, float]] = None,
                   radius: int = 1) -> Dict[Tuple, float]:
    """Mean-filter the parameter surface over its grid neighbourhood.

    This is the coherent multi-dimensional replacement for picking an argmax, or
    for centring each axis independently (which is incoherent: the combination
    of two per-axis centres need not be a good joint setting, and empirically
    was not). Smoothing directly encodes "prefer a setting whose neighbours are
    also good" -- a peak that stands alone is noise, a peak on a broad ridge is
    signal. Taking the argmax of the smoothed surface then yields a point that
    is robust in every swept direction at once.

    `scores` lets the caller supply an alternative objective (e.g. mean nDCG
    over training folds only) keyed by parameter tuple; defaults to each row's
    full-sample mean.
    """
    keyed = {tuple(round(r["params"][a], 6) for a in axes): r for r in rows}
    values = scores or {k: r["ndcg@10"] for k, r in keyed.items()}
    grids = [sorted({k[i] for k in keyed}) for i in range(len(axes))]
    pos = [{v: i for i, v in enumerate(g)} for g in grids]

    out: Dict[Tuple, float] = {}
    for key in keyed:
        idx = [pos[i][key[i]] for i in range(len(axes))]
        total, count = 0.0, 0
        # Rectangular neighbourhood of +/- radius grid steps on every axis.
        offsets = [[]]
        for _ in range(len(axes)):
            offsets = [o + [d] for o in offsets for d in range(-radius, radius + 1)]
        for off in offsets:
            nbr = []
            for i, d in enumerate(off):
                j = idx[i] + d
                if not (0 <= j < len(grids[i])):
                    nbr = None
                    break
                nbr.append(grids[i][j])
            if nbr is None:
                continue
            v = values.get(tuple(nbr))
            if v is not None:
                total += v
                count += 1
        out[key] = total / count if count else values[key]
    return out


def select_smoothed(rows: List[Dict], axes: Sequence[str],
                    scores: Optional[Dict[Tuple, float]] = None,
                    radius: int = 1) -> Dict:
    """Argmax of the smoothed surface -- the plateau-centre selection rule."""
    smoothed = smooth_surface(rows, axes, scores, radius)
    best_key = max(smoothed, key=lambda k: smoothed[k])
    keyed = {tuple(round(r["params"][a], 6) for a in axes): r for r in rows}
    return keyed[best_key]


def verify_against_harness(index, queries, qrels, scorer_name: str,
                           params: Dict[str, float], k: int = 10) -> float:
    """Cross-check: in-process nDCG@10 must match what the real harness reports.

    The sweep path is a reimplementation of the query loop for speed. If it ever
    drifts from submission/_traverse.py, every tuning decision made afterwards
    would be based on numbers the grader will not reproduce. Called from
    tests/test_experiment_harness_agreement.py.
    """
    return run_to_metrics(index, queries, qrels, scorer_name, params, k)["aggregate"]["ndcg@10"]
