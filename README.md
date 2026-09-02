# A1 — Sparse Retrieval Arena

Inverted index retrieval engine built from scratch (no Lucene/Elasticsearch/
Pyserini/Whoosh/`rank_bm25`), exposed through the three entrypoints the
grading harness calls.

`submission/retrieve.py` ships plain BM25 (`k1=4.5, b=0.60`) + a pseudo
title field, no feedback pass. Full methodology and everything tried and
rejected is in the report.

| | dev set (50 topics, 171,332 docs) |
|---|---|
| nDCG@10 | **0.6395** |
| MAP@10 | 0.0143 |
| MRR / P@10 | 0.8839 / 0.6960 |
| Index size | **22.8 MB** *(23,957,106 bytes)* |
| Index build | 2.6 s *(4 cores)* |
| Index load | ~0.5 s |
| Query latency | ~0.7 ms mean |

Also tried a pseudo relevance feedback (RM3) version, over a stemmed
analysis chain + stemmed title field. Scored higher on dev (0.6837 vs
0.6395) but lost on the actual held out topics on Day 4 (0.1714, near the
bottom of the class). Reverted back to plain BM25 and removed the rm3 code
entirely (`rm3.py`, `_forward.py`, tests) instead of keeping dead code
around. Report has the full story.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/smoke_test.sh
```

`smoke_test.sh` = what CI runs. conformance + metric tests + our own
component tests + a full harness run on the toy set.

## Reproducing the index / a full eval run

toy set ships with the repo, no download needed:

```bash
python -m harness.run_harness \
  --corpus data/toy/corpus.jsonl \
  --queries data/toy/queries_dev.tsv \
  --qrels data/toy/qrels_dev.txt \
  --run-out runs/dev_run.trec --report-out runs/dev_report.json
```

for the real collection:

```bash
python scripts/download_full_corpus.py          # ~171K docs via ir_datasets
python -m harness.run_harness \
  --corpus data/full/corpus.jsonl \
  --queries data/full/queries_dev.tsv \
  --qrels data/full/qrels_dev.txt \
  --run-out runs/full_run.trec --report-out runs/full_report.json
```

builds are deterministic, same corpus = byte identical index, same query =
same ranking every time (ties broken on doc id).

## What's in `submission/`

| File | Role |
|---|---|
| `retrieve.py` | the 3 required entrypoints + tuned constants |
| `indexer.py` | inverted index, delta+vbyte postings, save/load |
| `_analysis.py` | tokeniser/stemmer chain, shared build+query |
| `_codecs.py` | delta + vbyte codecs |
| `_scorers.py` | scorer registry: bm25, bm25+, lm-dirichlet, pl2, dph |
| `_traverse.py` | one postings pass feeding all scorers |
| `_proximity.py` | ordered/unordered window counts for term dependence |
| `bm25.py` | required bm25 entrypoint + title field scoring |
| `boolean_vsm.py` | required boolean AND/OR + tf-idf cosine VSM |
| `custom_scorer.py` | SDM, dev tested not shipped, see report |
| `setup.py` | build def for the cython extensions, run from inside submission/ |
| `_fast.pyx`, `_fastbuild.pyx` | optional c/c++ extensions (fused scoring, c++ tokeniser+builder, c++ porter stemmer verified against nltk). pure speed, falls back to python if not compiled |

### Design notes

**Columnar not dict of dicts.** at 16.3M postings a Dict[str, Dict[str, int]]
would eat several GB just in interpreter overhead, doesn't fit 8gb. flat
numpy arrays instead.

**Delta + vbyte postings** + nibble packed tf + zlib on disk. doc ids in a
postings list are sorted so gaps are small, vbyte spends 1 byte where int32
spends 4. whole collection encoded in one vectorised call not per term.

**No raw text stored.** scorers only need tf + length stats, storing the
corpus would cost index size for nothing.

**One traversal, N scorers.** each scorer is a pure function of per-posting
stats so running several costs barely more than one.

**Pseudo title field.** titles run directly into abstracts with no
delimiter so there's no real boundary, but early terms being more
indicative doesn't need an exact one. first 10 tokens indexed again as a
second field at small weight.

## Tests

```bash
pytest tests/ -v          # 151 tests
```

- `test_interface_conformance.py`, `test_metrics.py` — from the starter
- `test_codecs.py` — round trips, vbyte width boundaries
- `test_ranking_components.py` — bm25/vsm vs hand derived values + edge cases
- `test_fast_equivalence.py` — c extensions must be bit identical to python,
  and everything must still work if they didn't compile

## `experiments/` — tuning and analysis

not part of the submission, not in the archive either (package_submission.sh
scopes to exactly the required tree). kept here for anyone reviewing the
methodology, nothing in submission/ imports it.

```bash
python experiments/profile_corpus.py      # collection statistics
python experiments/sweep_bm25.py          # the k1/b grid search
python experiments/cv_select.py           # nested CV over selection rules
python experiments/tune.py --scorer all   # every scorer, honestly evaluated
python experiments/report_assets.py       # report tables and figures
```

`report_assets.py` needs `matplotlib`, not in requirements.txt on purpose
since nothing graded needs it.

## Reading

report covers the full measurement log, everything tried and rejected on
the evidence (analysis chain tuning, fusion, SDM/proximity, RM3, PL2/DPH),
and the reasoning behind reverting from RM3 back to plain BM25 after the
Day 4 held out result.
