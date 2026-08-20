#!/usr/bin/env bash
# Run the same checks CI runs, locally, before you push.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Interface conformance tests =="
pytest tests/test_interface_conformance.py -v

echo
echo "== Metrics unit tests =="
pytest tests/test_metrics.py -v

echo
echo "== Component correctness tests (codecs, BM25, Boolean/VSM) =="
pytest tests/test_codecs.py tests/test_ranking_components.py -v

echo
echo "== Full harness run on the toy set =="
python -m harness.run_harness \
  --corpus data/toy/corpus.jsonl \
  --queries data/toy/queries_dev.tsv \
  --qrels data/toy/qrels_dev.txt \
  --baseline-run data/toy/reference_bm25_run_dev.trec \
  --run-out runs/dev_run.trec \
  --report-out runs/dev_report.json

echo
echo "All smoke checks passed."
