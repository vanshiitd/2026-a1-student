"""
tests/test_ranking_components.py -- bm25/vsm/boolean correctness against
hand computed values, not just "does the code agree with itself"

fixture corpus:
    d1: "a b"        [a, b]        len 2
    d2: "a a c"      [a, a, c]     len 3
    d3: "a c c c"    [a, c, c, c]  len 4
    N=3, total=9, avgdl=3
    df: a=3, b=1, c=2   cf: a=4, b=1, c=4
"""
import math
import tempfile

import pytest

from submission import bm25, boolean_vsm
from submission.indexer import InvertedIndex

CORPUS = [("d1", "a b"), ("d2", "a a c"), ("d3", "a c c c")]
N = 3
AVGDL = 3.0


@pytest.fixture
def index():
    ix = InvertedIndex()
    ix.build(CORPUS)
    bm25.build(ix)
    boolean_vsm.build(ix)
    return ix


def test_index_statistics_match_hand_count(index):
    assert index.N == 3
    assert index.total_tokens == 9
    assert index.avg_doc_len == pytest.approx(3.0)
    assert index.document_frequency("a") == 3
    assert index.document_frequency("b") == 1
    assert index.document_frequency("c") == 2
    assert index.document_frequency("missing") == 0
    assert index.collection_frequency("a") == 4
    assert index.collection_frequency("c") == 4


def test_postings_are_sorted_and_correct(index):
    doc_ids, tfs = index.postings("c")
    assert list(doc_ids) == [1, 2]
    assert list(tfs) == [1, 3]
    assert list(doc_ids) == sorted(doc_ids)


# bm25 hand computed
def _bm25_idf(df, n=N):
    return math.log((n - df + 0.5) / (df + 0.5) + 1.0)


def _bm25_term(tf, dl, df, k1, b, n=N, avgdl=AVGDL):
    return _bm25_idf(df, n) * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl))


def test_bm25_single_term_query_matches_hand_computation(index):
    # k1=1.2, b=0.75. d3 (tf=3) beats d2 (tf=1) even though d3 is longer
    results = bm25.score("c", k=10, k1=1.2, b=0.75)
    assert [doc for doc, _ in results] == ["d3", "d2"]

    expected_d3 = _bm25_term(tf=3, dl=4, df=2, k1=1.2, b=0.75)
    expected_d2 = _bm25_term(tf=1, dl=3, df=2, k1=1.2, b=0.75)
    scores = dict(results)
    assert scores["d3"] == pytest.approx(expected_d3, rel=1e-12)
    assert scores["d2"] == pytest.approx(expected_d2, rel=1e-12)


def test_bm25_multi_term_query_sums_term_contributions(index):
    results = dict(bm25.score("b c", k=10, k1=1.2, b=0.75))
    expected = {
        "d1": _bm25_term(tf=1, dl=2, df=1, k1=1.2, b=0.75),
        "d2": _bm25_term(tf=1, dl=3, df=2, k1=1.2, b=0.75),
        "d3": _bm25_term(tf=3, dl=4, df=2, k1=1.2, b=0.75),
    }
    for doc, want in expected.items():
        assert results[doc] == pytest.approx(want, rel=1e-12)
    # b is rare so its idf dominates, d1 wins despite being shortest
    assert max(results, key=results.get) == "d1"


def test_bm25_zero_idf_term_still_scores_nonzero(index):
    # a is in every doc, +1 smoothing keeps idf positive not zero
    results = bm25.score("a", k=10, k1=1.2, b=0.75)
    assert len(results) == 3
    assert all(score > 0 for _doc, score in results)
    assert _bm25_idf(3) == pytest.approx(math.log(1 + 0.5 / 3.5), rel=1e-12)


def test_bm25_k1_and_b_are_real_parameters(index):
    """k1/b must actually be tunable not baked in constants"""
    default = bm25.score("c", k=10, k1=1.2, b=0.75)
    other_k1 = bm25.score("c", k=10, k1=2.5, b=0.75)
    other_b = bm25.score("c", k=10, k1=1.2, b=0.10)
    assert dict(default) != dict(other_k1), "changing k1 did not change scores"
    assert dict(default) != dict(other_b), "changing b did not change scores"


def test_bm25_b_zero_disables_length_normalisation(index):
    scores = dict(bm25.score("c", k=10, k1=1.2, b=0.0))
    expected_d2 = _bm25_idf(2) * (1 * 2.2) / (1 + 1.2)
    assert scores["d2"] == pytest.approx(expected_d2, rel=1e-12)


def test_bm25_higher_k1_reduces_tf_saturation(index):
    def ratio(k1):
        s = dict(bm25.score("c", k=10, k1=k1, b=0.75))
        return s["d3"] / s["d2"]
    assert ratio(2.5) > ratio(0.5)


# vsm hand computed
# idf(a)=ln(3/3)=0, idf(b)=ln(3/1)=1.0986123, idf(c)=ln(3/2)=0.4054651
IDF_B = math.log(3 / 1)
IDF_C = math.log(3 / 2)


def test_vsm_cosine_matches_hand_computation(index):
    # d2/d3 should tie exactly, both only have c and cosine is scale invariant
    q_norm = math.sqrt(IDF_B**2 + IDF_C**2)
    expected_d1 = (IDF_B * IDF_B) / (q_norm * IDF_B)
    expected_d2 = (IDF_C * IDF_C) / (q_norm * IDF_C)
    expected_d3 = (IDF_C * 3 * IDF_C) / (q_norm * 3 * IDF_C)

    results = boolean_vsm.vsm_score("b c", k=10)
    scores = dict(results)
    assert scores["d1"] == pytest.approx(expected_d1, rel=1e-12)
    assert scores["d2"] == pytest.approx(expected_d2, rel=1e-12)
    assert scores["d3"] == pytest.approx(expected_d3, rel=1e-12)
    assert expected_d2 == pytest.approx(expected_d3, rel=1e-12), "fixture should tie d2/d3"
    assert [doc for doc, _ in results] == ["d1", "d2", "d3"]


def test_vsm_cosine_is_bounded_by_one(index):
    for doc, score in boolean_vsm.vsm_score("a b c", k=10):
        assert -1e-9 <= score <= 1.0 + 1e-9, f"{doc} cosine out of range: {score}"


def test_vsm_query_of_only_zero_idf_terms_scores_zero(index):
    assert boolean_vsm.vsm_score("a", k=10) == []


def test_vsm_identical_query_and_document_scores_one(index):
    scores = dict(boolean_vsm.vsm_score("b", k=10))
    assert scores["d1"] == pytest.approx(1.0, rel=1e-12)


# boolean
def test_boolean_and_returns_intersection(index):
    assert boolean_vsm.boolean_search("a c", mode="and") == ["d2", "d3"]


def test_boolean_or_returns_union(index):
    assert boolean_vsm.boolean_search("b c", mode="or") == ["d1", "d2", "d3"]


def test_boolean_and_with_disjoint_terms_is_empty(index):
    assert boolean_vsm.boolean_search("b c", mode="and") == []


def test_boolean_unknown_term_empties_conjunction_but_not_disjunction(index):
    assert boolean_vsm.boolean_search("b zzz", mode="and") == []
    assert boolean_vsm.boolean_search("b zzz", mode="or") == ["d1"]


def test_boolean_repeated_term_is_idempotent(index):
    assert boolean_vsm.boolean_search("c c c", mode="and") == ["d2", "d3"]


def test_boolean_rejects_unknown_mode(index):
    with pytest.raises(ValueError):
        boolean_vsm.boolean_search("a", mode="xor")


# shared contract / edge cases

@pytest.mark.parametrize("fn", [
    lambda: bm25.score("a b c", k=2, k1=1.2, b=0.75),
    lambda: boolean_vsm.vsm_score("a b c", k=2),
])
def test_scorers_respect_k(index, fn):
    assert len(fn()) <= 2


@pytest.mark.parametrize("fn", [
    lambda k: bm25.score("a b c", k=k, k1=1.2, b=0.75),
    lambda k: boolean_vsm.vsm_score("a b c", k=k),
])
def test_scorers_handle_k_zero_and_k_larger_than_corpus(index, fn):
    assert fn(0) == []
    assert len(fn(100)) <= 3


@pytest.mark.parametrize("query", ["", "   ", "!!! ???", "zzz qqq"])
def test_scorers_handle_empty_and_unmatchable_queries(index, query):
    assert bm25.score(query, k=10, k1=1.2, b=0.75) == []
    assert boolean_vsm.vsm_score(query, k=10) == []
    assert boolean_vsm.boolean_search(query, mode="and") == []


@pytest.mark.parametrize("fn", [
    lambda: bm25.score("a b c", k=10, k1=1.2, b=0.75),
    lambda: boolean_vsm.vsm_score("a b c", k=10),
])
def test_scorers_never_return_duplicate_doc_ids(index, fn):
    docs = [doc for doc, _ in fn()]
    assert len(docs) == len(set(docs))


@pytest.mark.parametrize("fn", [
    lambda: bm25.score("a b c", k=10, k1=1.2, b=0.75),
    lambda: boolean_vsm.vsm_score("a b c", k=10),
])
def test_scorers_return_descending_scores(index, fn):
    scores = [s for _doc, s in fn()]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("fn", [
    lambda: bm25.score("a b c", k=10, k1=1.2, b=0.75),
    lambda: boolean_vsm.vsm_score("b c", k=10),
    lambda: boolean_vsm.boolean_search("a c", mode="and"),
])
def test_results_are_deterministic_across_repeated_calls(index, fn):
    assert fn() == fn()


def test_save_load_round_trip_preserves_rankings(index):
    before_bm25 = bm25.score("b c", k=10, k1=1.2, b=0.75)
    before_vsm = boolean_vsm.vsm_score("b c", k=10)
    before_bool = boolean_vsm.boolean_search("a c", mode="and")

    with tempfile.TemporaryDirectory() as index_dir:
        index.save(index_dir)
        reloaded = InvertedIndex.load(index_dir)

    bm25.build(reloaded)
    boolean_vsm.build(reloaded)
    assert bm25.score("b c", k=10, k1=1.2, b=0.75) == before_bm25
    assert boolean_vsm.vsm_score("b c", k=10) == before_vsm
    assert boolean_vsm.boolean_search("a c", mode="and") == before_bool


def test_scorers_raise_a_clear_error_before_build(index):
    bm25._INDEX = None
    boolean_vsm._INDEX = None
    try:
        with pytest.raises(RuntimeError, match="build"):
            bm25.score("a", k=10)
        with pytest.raises(RuntimeError, match="build"):
            boolean_vsm.vsm_score("a", k=10)
    finally:
        bm25.build(index)
        boolean_vsm.build(index)


# adversarial input, harness kills the whole run on any exception so a
# bad query should degrade gracefully not crash

ADVERSARIAL = [
    "",
    "   \t\n  ",
    "!!! ??? ---",
    "你好 مرحبا αβγ",
    "\x00\x01 null bytes",
    "a" * 50_000,
    " ".join(["covid"] * 5_000),
    "zzzqqq wwwvvv",
    "COVID-19 SARS-CoV-2 100%",
]


@pytest.mark.parametrize("query", ADVERSARIAL)
def test_retrieve_survives_adversarial_queries(index, query):
    from submission import retrieve as entry
    entry._INDEX = index
    bm25.build(index)
    results = entry.retrieve(query, 10)
    assert isinstance(results, list)
    assert len(results) <= 10
    docs = [d for d, _ in results]
    assert len(docs) == len(set(docs)), "duplicate doc_id would be a RUNTIME_ERROR"
    scores = [s for _d, s in results]
    assert scores == sorted(scores, reverse=True)
    for doc_id, score in results:
        assert doc_id in {"d1", "d2", "d3"}
        assert isinstance(score, (int, float))


def test_retrieve_still_raises_before_load_index():
    from submission import retrieve as entry
    saved = entry._INDEX
    entry._INDEX = None
    try:
        with pytest.raises(RuntimeError, match="load_index"):
            entry.retrieve("a", 10)
    finally:
        entry._INDEX = saved


def _build_active_strategy_state(entry):
    ix = InvertedIndex()
    ix.build(CORPUS)
    entry._INDEX = ix
    bm25.build(ix)
    return entry._INDEX


def test_retrieve_guard_degrades_one_query_not_the_run(index, monkeypatch, capsys):
    from submission import retrieve as entry
    entry._INDEX = index

    def boom(*_a, **_kw):
        raise ValueError("simulated scorer failure")

    monkeypatch.setattr(entry.bm25, "score", boom)
    assert entry.retrieve("a b c", 10) == []
    assert "simulated scorer failure" in capsys.readouterr().err

    monkeypatch.undo()
    _build_active_strategy_state(entry)
    assert entry.retrieve("a b c", 10), "must recover once the fault clears"
