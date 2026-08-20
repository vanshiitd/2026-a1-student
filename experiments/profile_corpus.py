#!/usr/bin/env python
"""
experiments/profile_corpus.py — day-1 corpus reality check (plan.md Section 4.1).

Streams the corpus/queries/qrels and writes a markdown profile. Everything here
is measurement, not modelling: the point is to replace the assumptions in
plan.md with numbers from the actual collection before any design is locked in.

Specifically answers:
  - How big is this really (docs, tokens, vocabulary, singleton terms)?
  - What does the document-length distribution look like (drives BM25 `b`)?
  - How deeply judged is it, and what is the *hard ceiling* on MAP@10?
  - How many topics are there (drives the whole variance argument in plan.md 5.0)?

Usage:
    python experiments/profile_corpus.py                      # data/full
    python experiments/profile_corpus.py --data-dir data/toy  # sanity check
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from submission.indexer import tokenize  # same tokenizer the index will use


def _percentiles(sorted_values, points=(0, 1, 5, 25, 50, 75, 95, 99, 100)):
    if not sorted_values:
        return {}
    n = len(sorted_values)
    out = {}
    for p in points:
        idx = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
        out[p] = sorted_values[idx]
    return out


def profile_corpus(corpus_path):
    doc_lens = []
    df = Counter()          # term -> document frequency
    total_tokens = 0
    empty_docs = 0
    n_docs = 0
    raw_bytes = 0

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_bytes += len(line)
            obj = json.loads(line)
            toks = tokenize(obj["text"])
            n_docs += 1
            total_tokens += len(toks)
            doc_lens.append(len(toks))
            if not toks:
                empty_docs += 1
            df.update(set(toks))

    doc_lens.sort()
    singletons = sum(1 for c in df.values() if c == 1)
    return {
        "n_docs": n_docs,
        "total_tokens": total_tokens,
        "avg_doc_len": total_tokens / n_docs if n_docs else 0.0,
        "doc_len_percentiles": _percentiles(doc_lens),
        "empty_docs": empty_docs,
        "vocab_size": len(df),
        "singleton_terms": singletons,
        "singleton_frac": singletons / len(df) if df else 0.0,
        "raw_text_bytes": raw_bytes,
        # Rough postings-count estimate = sum of document frequencies.
        "n_postings": sum(df.values()),
    }


def profile_queries(queries_path):
    lens, rows = [], []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            qid, _, text = line.partition("\t")
            toks = tokenize(text)
            lens.append(len(toks))
            rows.append((qid, text, len(toks)))
    lens.sort()
    return {
        "n_queries": len(rows),
        "query_len_percentiles": _percentiles(lens),
        "avg_query_len": sum(lens) / len(lens) if lens else 0.0,
        "sample": rows[:8],
    }


def profile_qrels(qrels_path):
    per_query_judged = Counter()
    per_query_relevant = Counter()
    grade_dist = Counter()
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, _doc_id, rel = parts[0], parts[1], parts[2], int(parts[3])
            per_query_judged[qid] += 1
            grade_dist[rel] += 1
            if rel > 0:
                per_query_relevant[qid] += 1

    qids = sorted(per_query_judged)
    judged = sorted(per_query_judged.values())
    relevant = sorted(per_query_relevant.get(q, 0) for q in qids)

    # The hard ceiling on MAP@10 (plan.md 4.1). retrieve() returns <= 10 docs,
    # but harness/metrics.py normalises AP by the TRUE relevant count. So even a
    # perfect top-10 scores only min(10, R)/R for a query with R relevant docs.
    ceilings = []
    for q in qids:
        R = per_query_relevant.get(q, 0)
        ceilings.append(min(10, R) / R if R else 0.0)
    map10_ceiling = sum(ceilings) / len(ceilings) if ceilings else 0.0

    return {
        "n_queries_judged": len(qids),
        "total_judgments": sum(per_query_judged.values()),
        "judged_per_query_percentiles": _percentiles(judged),
        "relevant_per_query_percentiles": _percentiles(relevant),
        "avg_judged_per_query": sum(judged) / len(judged) if judged else 0.0,
        "avg_relevant_per_query": sum(relevant) / len(relevant) if relevant else 0.0,
        "grade_distribution": dict(sorted(grade_dist.items())),
        "queries_with_no_relevant": sum(1 for r in relevant if r == 0),
        "map10_ceiling": map10_ceiling,
    }


def _fmt_pct(d):
    return " · ".join(f"p{p}={v}" for p, v in sorted(d.items()))


def _human(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    repo = os.path.join(os.path.dirname(__file__), "..")
    ap.add_argument("--data-dir", default=os.path.join(repo, "data", "full"))
    ap.add_argument("--out", default=os.path.join(repo, "..", "notes", "corpus_profile.md"))
    args = ap.parse_args()

    corpus_path = os.path.join(args.data_dir, "corpus.jsonl")
    queries_path = os.path.join(args.data_dir, "queries_dev.tsv")
    qrels_path = os.path.join(args.data_dir, "qrels_dev.txt")
    for p in (corpus_path, queries_path, qrels_path):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} — run scripts/download_full_corpus.py first")

    print(f"profiling {corpus_path} ...", flush=True)
    c = profile_corpus(corpus_path)
    q = profile_queries(queries_path)
    r = profile_qrels(qrels_path)

    # Naive-index size estimate: what a json/pickle dump of {term: {doc_id: tf}}
    # would cost, i.e. roughly what the class median will look like. Assumes the
    # external doc_id string is repeated once per posting -- which is exactly the
    # redundancy plan.md Section 6 sets out to remove.
    avg_docid_len = 12
    naive_bytes = c["n_postings"] * (avg_docid_len + 6)

    lines = [
        "# Corpus profile",
        "",
        f"_Generated by `experiments/profile_corpus.py` from `{os.path.relpath(args.data_dir, repo)}`._",
        "",
        "## Corpus",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Documents | {c['n_docs']:,} |",
        f"| Total tokens | {c['total_tokens']:,} |",
        f"| Average document length | {c['avg_doc_len']:.1f} tokens |",
        f"| Document length percentiles | {_fmt_pct(c['doc_len_percentiles'])} |",
        f"| Empty documents (0 tokens) | {c['empty_docs']:,} |",
        f"| Vocabulary size | {c['vocab_size']:,} |",
        f"| Singleton terms (df == 1) | {c['singleton_terms']:,} ({c['singleton_frac']:.1%} of vocab) |",
        f"| Total postings (sum of df) | {c['n_postings']:,} |",
        f"| Raw corpus text | {_human(c['raw_text_bytes'])} |",
        "",
        "## Queries",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Queries | {q['n_queries']:,} |",
        f"| Average query length | {q['avg_query_len']:.2f} tokens |",
        f"| Query length percentiles | {_fmt_pct(q['query_len_percentiles'])} |",
        "",
        "Sample queries:",
        "",
        "| qid | text | tokens |",
        "|---|---|---|",
    ]
    for qid, text, n in q["sample"]:
        lines.append(f"| {qid} | {text} | {n} |")

    lines += [
        "",
        "## Qrels",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Queries with judgments | {r['n_queries_judged']:,} |",
        f"| Total judgments | {r['total_judgments']:,} |",
        f"| Average judged per query | {r['avg_judged_per_query']:.1f} |",
        f"| Judged per query percentiles | {_fmt_pct(r['judged_per_query_percentiles'])} |",
        f"| Average *relevant* per query | {r['avg_relevant_per_query']:.1f} |",
        f"| Relevant per query percentiles | {_fmt_pct(r['relevant_per_query_percentiles'])} |",
        f"| Relevance grade distribution | {r['grade_distribution']} |",
        f"| Queries with no relevant doc | {r['queries_with_no_relevant']} |",
        "",
        "## Derived — consequences for the plan",
        "",
        f"- **MAP@10 hard ceiling: {r['map10_ceiling']:.4f}.** No submission in the class can exceed "
        f"this, because `retrieve()` returns at most 10 documents while `harness/metrics.py` "
        f"normalises AP by the true relevant count. Do not read a low MAP@10 as a bug. "
        f"(nDCG@10 has no such ceiling — its IDCG is also computed at depth 10 — so 1.0 stays "
        f"attainable there.)",
        f"- **{r['n_queries_judged']} topics** is the sample size every tuning decision is made "
        f"against. This is the number that drives the fuse-vs-select argument in `plan.md` 5.0; "
        f"the per-query standard deviation still has to be measured once L0 BM25 exists.",
        f"- **Naive index estimate: ~{_human(naive_bytes)}** for a `{{term: {{doc_id: tf}}}}` dump "
        f"(~{c['n_postings']:,} postings x ~{avg_docid_len + 6}B, repeating the doc_id string per "
        f"posting). This approximates the class median; `plan.md` 6 targets <= half of it.",
        f"- **Singleton terms are {c['singleton_frac']:.1%} of the vocabulary** — the dictionary-"
        f"pruning candidate in `plan.md` 6, Tier 1 item 7. Measure the nDCG cost before enabling.",
        "",
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {args.out}")
    print(f"\nMAP@10 ceiling: {r['map10_ceiling']:.4f}   topics: {r['n_queries_judged']}   "
          f"docs: {c['n_docs']:,}   vocab: {c['vocab_size']:,}")


if __name__ == "__main__":
    main()
