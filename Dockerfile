# matches the grading machine's toolchain, just for local testing. staff
# don't actually run this file, their image copies in only submission/
FROM python:3.12-slim

WORKDIR /repo

# for the optional cython extension
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# build at image time not inside build_index(), that would count against
# build-time efficiency
RUN if [ -f submission/setup.py ]; then \
        cd submission && python setup.py build_ext --inplace; \
    fi

# default: conformance + smoke test on the toy set, staff override this
# with the real corpus for scoring
CMD ["python", "-m", "harness.run_harness", \
     "--corpus", "data/toy/corpus.jsonl", \
     "--queries", "data/toy/queries_dev.tsv", \
     "--qrels", "data/toy/qrels_dev.txt", \
     "--baseline-run", "data/toy/reference_bm25_run_dev.trec", \
     "--run-out", "runs/dev_run.trec", \
     "--report-out", "runs/dev_report.json"]
