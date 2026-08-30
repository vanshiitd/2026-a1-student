# A1 — Sparse Retrieval Arena

An inverted-index retrieval engine built from scratch (no Lucene/Elasticsearch/
Pyserini/Whoosh/`rank_bm25`), exposed through the three entrypoints the grading
harness calls.

Two ranking strategies are fully implemented; `submission/retrieve.py`'s
`ACTIVE_STRATEGY` constant selects which one ships. Both are dev-validated and
independently tested regardless of which is active — switching is a one-line
change, not a redeploy.

**Currently shipped: `"rm3_stemmed"`** — BM25 with pseudo-relevance feedback
(RM3) over a stemmed analysis chain plus a stemmed pseudo-title field. Full
methodology, the parameter search, and every technique tried and rejected
along the way are in the accompanying report, submitted separately.

| | dev set (50 topics, 171,332 docs) |
|---|---|
| nDCG@10 | **0.6837** |
| MAP@10 | 0.0172 *(ceiling on this collection is ~0.027 — see the report)* |
| MRR / P@10 | 0.8583 / 0.7700 |
| Index size | **21.5 MB** |
| Index build | 2.7 s *(4 cores; ~4.4s on 1)* |
| Index load | ~2 s |
| Query latency | ~11–13 ms mean |

Three independent optimisations get RM3 here from an original 51MB / 24s
build / 19ms query, with byte-identical retrieval quality confirmed at every
step (nDCG@10/MAP@10/MRR/P@10 unchanged throughout):

- **Forward index built in memory at load time**, not persisted. A
  document's handful of distinct terms scattered across the whole vocabulary
  compresses far worse than a term's postings across all documents, so
  persisting it cost ~30MB for no query-time benefit. Load time isn't a
  graded efficiency metric, so this is a pure win, not a tradeoff: 51MB -> 21.5MB.
- **A C++ port of NLTK's Porter stemmer** (verified against NLTK across all
  207,034 distinct tokens the corpus produces, not sampled), plus caching a
  dict that was being rebuilt from all 171K doc_ids on every query and
  moving RM3's feedback-term aggregation into a small Cython kernel. Query
  latency ~19ms -> ~11-13ms; a large share of build time moved from Python
  to C++.
- **Tokenisation split across the grading machine's 4 cores.** Gated behind
  a document-count threshold so small corpora take the serial path
  automatically. Postings assembly and compression stay serial by nature
  (Amdahl's law caps the achievable speedup here around 2x, not 4x), but
  combined with the C++ stemmer this brings the full build to **2.7s**.

The alternative, `"shipped"` (plain BM25, `k1=4.5, b=0.60`, plus an unstemmed
pseudo-title field — no feedback pass), is slightly larger (23.96 MB, since
it carries no forward index at all) but now builds in almost exactly the
same time (2.9s) and has much lower query latency (0.65 ms, since it has no
feedback pass to run), and scores 0.6395 nDCG@10 on the same dev topics.
RM3's dev-set advantage is real but has never crossed the p<0.05 significance
bar this project holds every other change to (best honest estimate: p≈0.05–0.08
across four independent tests) and has an
unexplained asymmetric failure mode on a handful of topics. The report covers
the full evidence trail behind shipping the higher-variance option anyway,
and the held-out A/B plan resolving it during the competition round.

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
  --run-out runs/full_run.trec --report-out runs/full_report.json
```

Both builds are **deterministic** — the same corpus produces byte-identical
index files, and the same query returns an identical ranking every time
(ties break on ascending internal document id).

## What's in `submission/`

| File | Role |
|---|---|
| `retrieve.py` | The three required entrypoints. Deliberately thin; holds `ACTIVE_STRATEGY` and both strategies' tuned constants. |
| `indexer.py` | Columnar inverted index, delta+VByte postings, `save()`/`load()`. Optional term positions and forward (doc→terms) index. |
| `_analysis.py` | The single tokenisation/stopword/stemming chain, shared by indexing and querying. |
| `_codecs.py` | Delta and VByte integer codecs (vectorised). |
| `_forward.py` | Doc→terms index, built only when RM3 needs it. |
| `_scorers.py` | Scorer registry: BM25, BM25+, LM-Dirichlet, PL2, DPH. |
| `_traverse.py` | One postings traversal feeding every scorer. |
| `_proximity.py` | Ordered/unordered window counting for term dependence. |
| `bm25.py` | The required BM25 entrypoint (tunable `k1`, `b`), plus pseudo-title field scoring. |
| `boolean_vsm.py` | The required Boolean AND/OR and TF-IDF cosine VSM. |
| `custom_scorer.py` | Sequential Dependence Model over the above (dev-tested, not shipped — see the report). |
| `rm3.py` | Pseudo-relevance feedback (RM3) over the stemmed body + title. Active when `ACTIVE_STRATEGY = "rm3_stemmed"`. |
| `setup.py` | Build definition for the optional compiled extensions below. Run from inside `submission/`: `python setup.py build_ext --inplace`. |
| `_fast.pyx`, `_fastbuild.pyx` | Optional C/C++ extensions (Cython) — fused VByte-decode+BM25 scoring, RM3's feedback-mass aggregation, a C++ tokeniser/builder, and a C++ port of NLTK's Porter stemmer (verified against NLTK across every distinct token the corpus produces). All pure speed: every caller falls back to an equivalent pure-Python/NumPy path if the extension didn't compile, and `tests/test_fast_equivalence.py` asserts bit-identical results. |

### Design notes

**Columnar, not a dict of dicts.** `Dict[str, Dict[str, int]]` costs a Python
object per posting; at 16.3M postings that is several GB of interpreter overhead
and would not fit the 8 GB grading machine. Every quantity is a flat NumPy array.

**Postings are delta + VByte encoded**, plus nibble-packed term frequencies and
zlib compression on disk. Document ids within a postings list are sorted and
dense, so their gaps are small integers — a 4-byte int32 spends 4 bytes on a
gap of 3, VByte spends 1. The *whole collection* is encoded in a single
vectorised call rather than once per term.

**Raw document text is deliberately not persisted.** Every scorer needs only
term-frequency and length statistics; storing the raw corpus would cost the
index-size component for no query-time benefit.

**One traversal, N scorers.** A scorer is a pure function of per-posting
statistics, so running several rankers costs barely more than running one.

**Pseudo-title field.** Titles run directly into abstracts with no delimiter,
so the boundary isn't recoverable — but "early terms are more indicative"
doesn't need an exact one. The first 10 tokens of each document are indexed as
a second field and added at a small weight, both to plain BM25 and to RM3's
scoring.

## Tests

```bash
pytest tests/ -v          # 179 tests
```

- `test_interface_conformance.py`, `test_metrics.py` — shipped with the starter.
- `test_codecs.py` — codec round-trips, including the exact VByte width
  boundaries where an off-by-one would hide.
- `test_ranking_components.py` — BM25 and VSM against **hand-derived** expected
  values, plus determinism, save/load stability, and edge cases.
- `test_fast_equivalence.py` — the C extensions must be bit-identical to the
  pure-Python paths, and the submission must still run correctly if they never
  compiled at all. Also guards `submission/setup.py`'s location and the
  float-safety compile flags.
- `test_forward_index.py`, `test_rm3_strategy.py` — the forward index and the
  RM3 strategy, including a guard that `ACTIVE_STRATEGY` matches what was
  actually intended to ship (update it alongside any deliberate switch).

## `experiments/` — tuning and analysis

Not part of the submission's runtime, and not part of the submission archive
either (`scripts/package_submission.sh` scopes the zip to exactly the
assignment's required file tree). Lives in this repository for anyone
reviewing the methodology; nothing under `submission/` imports it.

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

The accompanying report covers the strategy and the full measurement log —
every technique that was implemented, measured, and then **rejected on the
evidence** (analysis-chain tuning, fusion in several forms, SDM/proximity,
filtered RM3 feedback vocabularies, PL2/DPH). The gap between in-sample and
cross-validated gains that produced most of those rejections, and the
reasoning behind shipping RM3 despite it not clearing the project's own
significance bar, are the substance of it.
