#!/usr/bin/env python
"""
experiments/report_assets.py — generate every table and figure the report needs.

Assignment Section 6 requires specific artifacts, and this produces all of them
from the recorded experiments so the report never quotes a hand-copied number:

  * a table comparing Boolean/VSM against BM25 on the dev set
  * a plot of nDCG@10 against the swept parameter (k1 and b), plus the joint
    (k1, b) surface
  * an error analysis of the worst-ranked topics
  * the full ablation table across every technique tried
  * an in-sample vs cross-validated figure, which is the project's main result

DEPENDENCY NOTE: this needs matplotlib, which is deliberately **not** in
requirements.txt. Nothing under submission/ imports it, and requirements.txt
drives the grading image build -- there is no reason to make grading depend on a
plotting library. Install it locally with `pip install matplotlib`.

Usage:
    python experiments/report_assets.py
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from harness.metrics import evaluate_run, ndcg_at_k
from experiments.evaluate import QueryPostings, get_index, load_topics
from submission import _scorers, bm25, boolean_vsm
from submission._scorers import CollectionStats

REPO = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(os.path.dirname(__file__), "results.jsonl")

# Every technique measured, with its in-sample and honest cross-validated gain.
# Sourced from notes/findings.md F9-F23; kept here so the ablation table and the
# figure are generated from one list rather than two hand-copied ones.
ABLATION = [
    ("k1/b grid search",            +0.0685, +0.0657, "0.0002", "SHIPPED"),
    ("Analysis chain (7 variants)", +0.0218, +0.0043, "0.86",   "rejected"),
    ("Cross-chain RRF fusion",      +0.0218, +0.0070, "0.43",   "rejected"),
    ("RM3 feedback",                +0.0190, +0.0048, "0.79",   "rejected"),
    ("SDM / proximity",             +0.0022, +0.0022, "0.23",   "rejected"),
    ("IDF exponent (p=1.25)",       -0.0178, -0.0178, "n/a",    "rejected"),
    ("z-norm score fusion",         -0.0020, -0.0020, "0.50",   "rejected"),
    ("IDF coverage weighting",      -0.0086, -0.0086, "0.36",   "rejected"),
    ("Scorer RRF fusion",           -0.0190, -0.0190, "0.16",   "rejected"),
    ("Near-duplicate removal",      -0.0285, -0.0285, "n/a",    "rejected"),
    ("Drop high-df query terms",    -0.0058, -0.0058, "n/a",    "rejected"),
    ("Require rarest query term",   -0.0840, -0.0840, "n/a",    "rejected"),
]


def load_rows(tag):
    rows, seen = [], set()
    with open(RESULTS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("tag") != tag:
                continue
            key = tuple(sorted(r["params"].items()))
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Required table: Boolean / VSM / BM25
# ---------------------------------------------------------------------------

def retrieval_model_table(index, queries, qrels):
    """Boolean vs VSM vs the probabilistic models, on the full dev set.

    Boolean retrieval returns an unranked set, so it has no meaningful nDCG of
    its own; scoring its first 10 documents in index order is the honest way to
    show what ranking actually buys, and the set sizes explain why Boolean alone
    is unusable at this scale.
    """
    print("evaluating Boolean / VSM / BM25 ...", flush=True)
    boolean_vsm.build(index)
    bm25.build(index)
    rows = []

    for mode in ("and", "or"):
        sizes, run, lat = [], {}, []
        for qid, text in queries:
            if qid not in qrels:
                continue
            t0 = time.perf_counter()
            docs = boolean_vsm.boolean_search(text, mode=mode)
            lat.append(time.perf_counter() - t0)
            sizes.append(len(docs))
            run[qid] = [(d, float(-i)) for i, d in enumerate(docs[:10])]
        ev = evaluate_run(run, qrels, k=10)["aggregate"]
        rows.append((f"Boolean {mode.upper()}", ev["ndcg@10"], ev["map@10"], ev["mrr"],
                     ev["p@10"], np.mean(lat) * 1000,
                     f"{np.mean(sizes):,.0f} docs matched (mean)"))

    for label, fn in (("VSM (TF-IDF cosine)", lambda q: boolean_vsm.vsm_score(q, 10)),
                      ("BM25 (k1=4.5, b=0.60)", lambda q: bm25.score(q, 10, k1=4.5, b=0.60))):
        run, lat = {}, []
        for qid, text in queries:
            if qid not in qrels:
                continue
            t0 = time.perf_counter()
            run[qid] = fn(text)
            lat.append(time.perf_counter() - t0)
        ev = evaluate_run(run, qrels, k=10)["aggregate"]
        rows.append((label, ev["ndcg@10"], ev["map@10"], ev["mrr"], ev["p@10"],
                     np.mean(lat) * 1000, "ranked"))

    # The remaining registered scorers, at their tuned settings.
    with open(os.path.join(os.path.dirname(__file__), "tuned_params.json")) as f:
        tuned = {k: v["params"] for k, v in json.load(f).items()}
    stats = CollectionStats(index.N, index.avg_doc_len, index.total_tokens)
    for name in ("bm25plus", "pl2", "lmd", "dph"):
        if name not in tuned:
            continue
        run, lat = {}, []
        for qid, text in queries:
            if qid not in qrels:
                continue
            t0 = time.perf_counter()
            p = QueryPostings(index, text)
            run[qid] = p.rank(_scorers.get(name),
                              _scorers.resolve_params(name, tuned[name]), stats, k=10)
            lat.append(time.perf_counter() - t0)
        ev = evaluate_run(run, qrels, k=10)["aggregate"]
        label = {"bm25plus": "BM25+", "pl2": "PL2 (DFR)", "lmd": "LM-Dirichlet",
                 "dph": "DPH (DFR)"}[name]
        rows.append((f"{label} {tuned[name]}", ev["ndcg@10"], ev["map@10"], ev["mrr"],
                     ev["p@10"], np.mean(lat) * 1000, "ranked"))

    out = ["| Model | nDCG@10 | MAP@10 | MRR | P@10 | ms/query | note |",
           "|---|---|---|---|---|---|---|"]
    for name, nd, mp, mrr, p10, ms, note in rows:
        out.append(f"| {name} | {nd:.4f} | {mp:.4f} | {mrr:.4f} | {p10:.4f} | {ms:.1f} | {note} |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Required figure: nDCG@10 vs the swept parameter
# ---------------------------------------------------------------------------

def sweep_figures(outdir):
    rows = load_rows("bm25-k1b-extended")
    by = {(r["params"]["k1"], r["params"]["b"]): r["ndcg@10"] for r in rows}
    se = {(r["params"]["k1"], r["params"]["b"]): r["se"] for r in rows}
    k1s = sorted({k for k, _ in by})
    bs = sorted({b for _, b in by})
    best = max(by, key=by.get)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (vals, fixed, label, other, plain) in zip(axes, [
            (k1s, best[1], "$k_1$", "b", "k1"), (bs, best[0], "$b$", "k_1", "b")]):
        if label == "$k_1$":
            y = [by[(v, fixed)] for v in vals]
            e = [se[(v, fixed)] for v in vals]
        else:
            y = [by[(fixed, v)] for v in vals]
            e = [se[(fixed, v)] for v in vals]
        ax.errorbar(vals, y, yerr=e, marker="o", ms=3, lw=1.4, capsize=2,
                    color="#2b6cb0", ecolor="#a0c4e8")
        peak = vals[int(np.argmax(y))]
        ax.axvline(peak, ls="--", c="#c53030", lw=1,
                   label=f"optimum {plain} = {peak:g}")
        ax.axhline(by[(1.2, 0.75)], ls=":", c="#666", lw=1, label="textbook default")
        ax.set_xlabel(label)
        ax.set_ylabel("nDCG@10")
        ax.set_title(f"nDCG@10 vs {label}   (${other}$ = {fixed:g})", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("BM25 parameter sweep, 50 dev topics (error bars = SE of the mean)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_k1_b_sweep.png"), dpi=180)
    plt.close(fig)

    # Joint surface: shows the plateau that motivates smoothed selection.
    grid = np.array([[by[(k, b)] for b in bs] for k in k1s])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis",
                   extent=[min(bs), max(bs), min(k1s), max(k1s)])
    ax.scatter([best[1]], [best[0]], marker="*", s=190, c="white",
               edgecolors="black", zorder=3, label=f"optimum ({best[0]:g}, {best[1]:g})")
    ax.scatter([0.75], [1.2], marker="o", s=60, c="#c53030",
               edgecolors="white", zorder=3, label="textbook (1.2, 0.75)")
    ax.set_xlabel("$b$ (length normalisation)")
    ax.set_ylabel("$k_1$ (tf saturation)")
    ax.set_title("nDCG@10 over the joint $(k_1, b)$ surface", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    fig.colorbar(im, ax=ax, label="nDCG@10")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_k1_b_surface.png"), dpi=180)
    plt.close(fig)

    within = sum(1 for v in by.values() if v >= by[best] - 0.019)
    return best, by[best], by[(1.2, 0.75)], within, len(by)


def ablation_figure(outdir):
    """In-sample vs cross-validated gain — the project's central result."""
    names = [a[0] for a in ABLATION][::-1]
    ins = [a[1] for a in ABLATION][::-1]
    hon = [a[2] for a in ABLATION][::-1]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.barh(y + 0.19, ins, 0.38, label="in-sample (dev set)", color="#f6ad55")
    ax.barh(y - 0.19, hon, 0.38, label="honest (nested CV)", color="#2b6cb0")
    ax.axvline(0, c="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("$\\Delta$ nDCG@10 vs tuned BM25 baseline")
    ax.set_title("Every technique's apparent gain, and what survived cross-validation",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_ablation.png"), dpi=180)
    plt.close(fig)


def error_analysis(index, queries, qrels, n=5):
    bm25.build(index)
    per_q, grades = [], Counter()
    for qid, text in queries:
        if qid not in qrels:
            continue
        r = bm25.score(text, 10, k1=4.5, b=0.60)
        g = [qrels[qid].get(d, 0) for d, _ in r]
        grades.update(g)
        per_q.append((qid, ndcg_at_k([d for d, _ in r], qrels[qid], k=10), g, text,
                      sum(1 for v in qrels[qid].values() if v > 0)))
    per_q.sort(key=lambda x: x[1])

    out = ["| Topic | nDCG@10 | rel. docs | grades of our top-10 | query |",
           "|---|---|---|---|---|"]
    for qid, nd, g, text, nrel in per_q[:n]:
        out.append(f"| {qid} | {nd:.3f} | {nrel} | `{g}` | {text} |")

    ours = sum((2 ** k - 1) * v for k, v in grades.items())
    ideal = 3 * 10 * len(per_q)
    summary = (f"Across all {len(per_q)} topics our top-10 slots hold "
               f"{grades.get(2,0)} grade-2, {grades.get(1,0)} grade-1 and "
               f"**{grades.get(0,0)} entirely non-relevant** documents. Total DCG gain "
               f"{ours} of {ideal} achievable (**{ours/ideal:.1%}**); every topic has at "
               f"least ten grade-2 documents available, so the ideal top-10 is always ten "
               f"grade-2 documents.")
    return "\n".join(out), summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(REPO, "..", "report"))
    args = ap.parse_args()
    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)

    index = get_index(os.path.join(REPO, "data", "full", "corpus.jsonl"))
    queries, qrels = load_topics()

    print("figures ...", flush=True)
    best, best_v, default_v, within, total = sweep_figures(outdir)
    ablation_figure(outdir)
    model_table = retrieval_model_table(index, queries, qrels)
    err_table, err_summary = error_analysis(index, queries, qrels)

    abl = ["| Technique | in-sample Δ | honest CV Δ | p | outcome |", "|---|---|---|---|---|"]
    for name, i, h, p, status in ABLATION:
        abl.append(f"| {name} | {i:+.4f} | **{h:+.4f}** | {p} | {status} |")

    with open(os.path.join(outdir, "tables.md"), "w", encoding="utf-8") as f:
        f.write("# Generated report assets\n\n"
                "_Produced by `experiments/report_assets.py`. Do not hand-edit — regenerate._\n\n"
                "## Table 1 — Retrieval models on the dev set\n\n" + model_table +
                "\n\n## Table 2 — Ablation: apparent vs honest gains\n\n" + "\n".join(abl) +
                f"\n\nBest (k1, b) = ({best[0]:g}, {best[1]:g}) at nDCG@10 {best_v:.4f}; "
                f"textbook (1.2, 0.75) scores {default_v:.4f}. "
                f"**{within} of {total} grid points lie within one paired SE (0.019) of the "
                f"optimum**, which is why the surface is treated as a plateau.\n\n"
                "## Table 3 — Error analysis, five worst topics\n\n" + err_table +
                "\n\n" + err_summary + "\n\n"
                "## Figures\n\n"
                "- `fig1_k1_b_sweep.png` — nDCG@10 vs k1 and vs b (assignment Section 6 requirement)\n"
                "- `fig2_k1_b_surface.png` — the joint surface and its plateau\n"
                "- `fig3_ablation.png` — in-sample vs cross-validated gain per technique\n")
    print(f"\nwrote {outdir}/tables.md and 3 figures")


if __name__ == "__main__":
    main()
