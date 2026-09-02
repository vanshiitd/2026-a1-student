#!/usr/bin/env python
"""
scripts/query.py — build an index and run ad-hoc queries against it.

Deliberately goes through the SAME three functions the grading harness calls --
build_index(), load_index(), retrieve() -- rather than reaching into the
internals. So whatever this prints is what the graders would get, and if this
works the submission path works.

    # build the index once (writes to .query_index/ by default)
    python scripts/query.py --build

    # one-off query
    python scripts/query.py "what is the origin of covid-19"

    # interactive: type queries, blank line or Ctrl-D to quit
    python scripts/query.py

Useful flags:
    -k 20             how many results
    --text            show the start of each document, not just its id
    --explain         per-term idf/df breakdown of the query
    --corpus PATH     default data/full/corpus.jsonl, --toy for the small set
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

DEFAULT_INDEX = os.path.join(REPO, ".query_index")
FULL = os.path.join(REPO, "data", "full", "corpus.jsonl")
TOY = os.path.join(REPO, "data", "toy", "corpus.jsonl")


def _offsets_path(corpus):
    return os.path.join(REPO, ".query_offsets-" + os.path.basename(
        os.path.dirname(corpus)) + ".json")


def build_offsets(corpus):
    """doc_id -> byte offset, so --text can seek instead of holding 171K docs."""
    path = _offsets_path(corpus)
    if os.path.exists(path) and os.path.getmtime(path) > os.path.getmtime(corpus):
        with open(path) as f:
            return json.load(f)
    print("  indexing document offsets for --text (one pass, cached) ...",
          file=sys.stderr)
    offsets, pos = {}, 0
    with open(corpus, "rb") as f:
        for line in f:
            try:
                offsets[json.loads(line)["doc_id"]] = pos
            except Exception:
                pass
            pos += len(line)
    with open(path, "w") as f:
        json.dump(offsets, f)
    return offsets


def fetch_text(corpus, offsets, doc_id, limit=180):
    off = offsets.get(doc_id)
    if off is None:
        return ""
    with open(corpus, "rb") as f:
        f.seek(off)
        try:
            d = json.loads(f.readline())
        except Exception:
            return ""
    t = (d.get("text") or d.get("contents") or "").replace("\n", " ").strip()
    return t[:limit] + ("..." if len(t) > limit else "")


def explain(query):
    """Per-term df/idf, so you can see which words are actually doing the work."""
    import numpy as np
    from submission import retrieve as R
    from submission._analysis import analyze
    ix = R._INDEX
    if ix is None:
        return
    terms = list(dict.fromkeys(analyze(query, ix.config)))
    lookup = {t: i for i, t in enumerate(ix.terms)}
    print(f"  {'term':<20}{'df':>10}{'idf':>9}   share of query weight")
    rows = []
    for t in terms:
        i = lookup.get(t)
        if i is None:
            rows.append((t, 0, 0.0))
            continue
        df = int(ix.df[i])
        idf = float(np.log(((ix.N - df + 0.5) / (df + 0.5)) + 1.0))
        rows.append((t, df, idf))
    total = sum(r[2] for r in rows) or 1.0
    for t, df, idf in rows:
        bar = "#" * int(round(30 * idf / max(r[2] for r in rows) if rows else 0))
        note = "  (not in vocabulary)" if df == 0 else ""
        print(f"  {t:<20}{df:>10,}{idf:>9.3f}   {100*idf/total:5.1f}% {bar}{note}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="*", help="query text; omit for interactive mode")
    ap.add_argument("--build", action="store_true", help="(re)build the index first")
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--toy", action="store_true", help="use the small toy corpus")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--text", action="store_true", help="show document snippets")
    ap.add_argument("--explain", action="store_true", help="show per-term idf")
    args = ap.parse_args()

    corpus = args.corpus or (TOY if args.toy else FULL)
    if not os.path.exists(corpus):
        sys.exit(f"corpus not found: {corpus}\n"
                 f"For the full corpus run: python scripts/download_full_corpus.py")

    from submission import retrieve as R

    if args.build or not os.path.isdir(args.index):
        if not args.build:
            print(f"no index at {args.index} -- building it now", file=sys.stderr)
        print(f"building from {os.path.relpath(corpus, REPO)} ...",
              file=sys.stderr)
        t0 = time.perf_counter()
        R.build_index(corpus, args.index)
        size = sum(os.path.getsize(os.path.join(dp, f))
                   for dp, _, fs in os.walk(args.index) for f in fs)
        print(f"built in {time.perf_counter()-t0:.2f}s, "
              f"{size/1e6:.1f} MB on disk\n", file=sys.stderr)

    t0 = time.perf_counter()
    R.load_index(args.index)
    load_s = time.perf_counter() - t0

    offsets = build_offsets(corpus) if args.text else {}

    def show(q):
        t = time.perf_counter()
        try:
            results = R.retrieve(q, args.k)
        except Exception as e:
            print(f"  error: {e}\n")
            return
        ms = (time.perf_counter() - t) * 1000
        if args.explain:
            explain(q)
        if not results:
            print(f"  no results  ({ms:.2f} ms)\n")
            return
        width = max(len(d) for d, _ in results)
        for i, (doc_id, score) in enumerate(results, 1):
            print(f"  {i:>3}. {doc_id:<{width}}  {score:8.4f}")
            if args.text:
                snippet = fetch_text(corpus, offsets, doc_id)
                if snippet:
                    print(f"       {snippet}")
        print(f"  ({len(results)} results in {ms:.2f} ms)\n")

    if args.query:
        q = " ".join(args.query)
        print(f"\n{q!r}")
        show(q)
        return

    print(f"\nindex loaded in {load_s:.2f}s. "
          f"Type a query, blank line or Ctrl-D to quit.\n")
    while True:
        try:
            q = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        show(q)


if __name__ == "__main__":
    main()
