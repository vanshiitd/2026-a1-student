# A1 — Sparse Retrieval Arena

An inverted-index retrieval engine built from scratch (no Lucene/Elasticsearch/
Pyserini/Whoosh/`rank_bm25`), exposed through the three entrypoints the grading
harness calls.

**Final entry: BM25 with `k1 = 4.5`, `b = 0.60`** over a plain lowercase
alphanumeric analysis chain.

| | dev set (50 topics, 171,332 docs) |
|---|---|
| nDCG@10 | **0.6281** |
| MAP@10 | 0.0141 *(ceiling on this collection is 0.0267 — see report §2)* |
| MRR / P@10 | 0.8804 / 0.6900 |
| Index size | **40.8 MB** |
| Index build | 10.7 s |
| Index load | 0.03 s |
| Query latency | 9.7 ms mean |

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/smoke_test.sh
```

`smoke_test.sh` runs exactly what CI runs: interface conformance, the harness's
metric tests, this submission's own component tests, and a full harness pass on
the toy set.

## Reproducing the index and a full evaluation run

The toy set ships with the repository and needs no download:

```bash
python -m harness.run_harness \
  --corpus data/toy/corpus.jsonl \
  --queries data/toy/queries_dev.tsv \
  --qrels data/toy/qrels_dev.txt \
  --run-out runs/dev_run.trec --report-out runs/dev_report.json
```

For the real collection, fetch it once and point the harness at `data/full/`:

```bash
python scripts/download_full_corpus.py          # ~171K docs via ir_datasets
python -m harness.run_harness \
  --corpus data/full/corpus.jsonl \
  --queries data/full/queries_dev.tsv \
  --qrels data/full/qrels_dev.txt \
  --run-out runs/full_bm25_tuned.trec --report-out runs/full_bm25_tuned.json
```

Both builds are **deterministic** — the same corpus produces byte-identical
index files, and the same query returns an identical ranking every time
(ties break on ascending internal document id).

## What's in `submission/`

| File | Role |
|---|---|
| `retrieve.py` | The three required entrypoints. Deliberately thin; holds the tuned `k1`/`b` as named constants. |
| `indexer.py` | Columnar inverted index, delta+VByte postings, `save()`/`load()`. Optional term positions. |
| `_analysis.py` | The single tokenisation/stopword/stemming chain, shared by indexing and querying. |
| `_codecs.py` | Delta and VByte integer codecs (vectorised). |
| `_scorers.py` | Scorer registry: BM25, BM25+, LM-Dirichlet, PL2, DPH. |
| `_traverse.py` | One postings traversal feeding every scorer; RRF fusion helper. |
| `_proximity.py` | Ordered/unordered window counting for term dependence. |
| `bm25.py` | The required BM25 entrypoint (tunable `k1`, `b`). |
| `boolean_vsm.py` | The required Boolean AND/OR and TF-IDF cosine VSM. |
| `custom_scorer.py` | Sequential Dependence Model over the above. |

### Design notes

**Columnar, not a dict of dicts.** `Dict[str, Dict[str, int]]` costs a Python
object per posting; at 16.3M postings that is several GB of interpreter overhead
and would not fit the 8 GB grading machine. Every quantity is a flat NumPy array.

**Postings are delta + VByte encoded.** Document ids within a postings list are
sorted and dense, so their gaps are small integers — a 4-byte int32 spends 4
bytes on a gap of 3, VByte spends 1. The *whole collection* is encoded in a
single vectorised call rather than once per term. Result: 40.8 MB against an
estimated ~279 MB for a naive JSON dump of the same data.

**Raw document text is deliberately not persisted.** BM25 and VSM need only
term-frequency and length statistics; storing 189 MB of text would cost the
index-size component for no query-time benefit.

**One traversal, N scorers.** A scorer is a pure function of per-posting
statistics, so running several rankers costs barely more than running one.

## Tests

```bash
pytest tests/ -v          # 74 tests
```

- `test_interface_conformance.py`, `test_metrics.py` — shipped with the starter.
- `test_codecs.py` — codec round-trips, including the exact VByte width
  boundaries where an off-by-one would hide.
- `test_ranking_components.py` — BM25 and VSM against **hand-derived** expected
  values (computed in the comments from the formulas, never by pasting what the
  implementation returned), plus determinism, save/load stability, and edge cases.

## `experiments/` — tuning and analysis

Not part of the submission's runtime; nothing under `submission/` imports it.

```bash
python experiments/profile_corpus.py      # collection statistics
python experiments/sweep_bm25.py          # the k1/b grid search
python experiments/cv_select.py           # nested CV over selection rules
python experiments/tune.py --scorer all   # every scorer, honestly evaluated
python experiments/report_assets.py       # report tables and figures
```

`report_assets.py` additionally needs `matplotlib`, which is **not** in
`requirements.txt` on purpose — nothing in the graded path imports it, and
`requirements.txt` drives the grading image build.

## Reading

`plan.md` holds the strategy and `notes/findings.md` the measurement log —
21 numbered findings, including the techniques that were implemented, measured,
and then **rejected on the evidence** (analysis-chain tuning, RRF and score-based
fusion, SDM/proximity, RM3 feedback, PL2/DPH). Their negative results, and the
gap between in-sample and cross-validated gains that produced them, are the
substance of the report.
