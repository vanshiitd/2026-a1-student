# Pinned, minimal image so the grading harness runs your submission
# identically regardless of local setup (assignment Section 5,
# "Containerisation"). Course staff run every submission through this
# same image at grading time.
FROM python:3.11-slim

WORKDIR /repo

# C/C++ toolchain — present so a submission that compiles part of itself
# as a Cython or pybind11 extension (see docs/SUBMISSION_INTERFACE.md,
# "Compiled extensions") builds correctly here and in course staff's
# grading image (instructor-tools/Dockerfile.grading, kept in lockstep
# with this one). Installed at image-build time only; the grading
# container still runs with --network none at scoring time.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Compile the C extension here, at image-build time, while network access is
# still available and where the cost is not charged to any graded metric. The
# assignment is explicit that build_index() must not do this: anything it does
# counts against the index-build-time efficiency score. If this step were ever
# removed the submission still runs -- submission/bm25.py imports the extension
# behind try/except and falls back to a pure-NumPy path.
RUN python setup.py build_ext --inplace

# Default command: run the interface conformance + smoke-test suite
# against the toy set. Course staff override CMD to point at the real
# corpus/topics/qrels for scoring.
CMD ["python", "-m", "harness.run_harness", \
     "--corpus", "data/toy/corpus.jsonl", \
     "--queries", "data/toy/queries_dev.tsv", \
     "--qrels", "data/toy/qrels_dev.txt", \
     "--baseline-run", "data/toy/reference_bm25_run_dev.trec", \
     "--run-out", "runs/dev_run.trec", \
     "--report-out", "runs/dev_report.json"]
