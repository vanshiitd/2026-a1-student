#!/usr/bin/env python
"""
experiments/rm3_refine.py — attack RM3's feedback-term selection directly.

The shipped RM3 estimates each candidate expansion term's weight as

    mass(t) = sum over feedback docs of  doc_weight * tf(t, d) / len(d)

and then takes the top `fb_terms` by that mass. Nothing in that expression
refers to the *collection*. So the terms that win are simply the ones that
occur most often in the feedback documents -- which, in any English corpus,
means function words, and in this corpus also means "covid", "patient",
"study". F30 noted stopword contamination in the harvested vocabulary and an
attempted fix that made things worse; this script does the fix properly rather
than abandoning it.

Five selection rules, all cheap, all principled:

  raw       what ships today: top-k by RM1 mass. The control.
  idf       mass(t) * idf(t). The standard move -- a term earns its place by
            being frequent in the feedback set AND informative in general.
  dfcut     drop any term appearing in more than a fraction of the corpus
            before ranking. A blunt stoplist derived from the data rather
            than an imported word list.
  stop      drop a real stoplist, applied ONLY to feedback selection. Note
            this is not the rejected experiment: F26/F28 removed stopwords
            from the INDEX, which loses them for matching. Here they are
            indexed and matchable as always, and merely barred from being
            *invented* as expansion terms.
  idf+dfcut both filters together.

Then, conditional on the best rule surviving honest evaluation:

  * a (fb_docs, fb_terms, alpha) sweep, re-run under that rule, since the
    optimum may move once the candidate pool stops being full of noise;
  * a k1/b re-tune, because an expanded query has different length and weight
    structure from the user's original and there is no reason the parameters
    tuned for one are optimal for the other;
  * selective RM3 -- apply expansion only where the first pass looks weak,
    which targets F30's asymmetric failures (three topics lose 0.2-0.4) rather
    than its average.

Every headline number is cross-validated and paired-bootstrapped against the
shipped configuration. In-sample bests are reported only as such.

    python experiments/rm3_refine.py
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from harness.metrics import ndcg_at_k  # noqa: E402
from experiments.evaluate import cv_folds, get_index, load_topics, paired_bootstrap  # noqa: E402
from experiments.rm3_stemmed_probe import build_forward_in_memory, field_weighted_rank  # noqa: E402
from experiments.structure_probe import accumulate, build_field, evaluate, window  # noqa: E402
from submission._analysis import AnalysisConfig, analyze  # noqa: E402
from submission.indexer import InvertedIndex  # noqa: E402

K1, B = 4.5, 0.60
TITLE_W, TITLE_LAMBDA = 10, 0.10
K = 10

# F30's selected operating point, the thing to beat.
FB_DOCS, FB_TERMS, ALPHA = 10, 20, 0.6

# A compact, uncontroversial English stoplist. Deliberately NOT NLTK's -- this
# is applied to feedback-term selection only, so it wants the high-frequency
# function words and nothing domain-specific that might carry signal here.
STOPLIST = set("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can cannot could couldn't did didn't
do does doesn't doing don't down during each few for from further had hadn't has
hasn't have haven't having he her here hers herself him himself his how i if in
into is isn't it its itself let's me more most mustn't my myself no nor not of
off on once only or other ought our ours ourselves out over own same shan't she
should shouldn't so some such than that the their theirs them themselves then
there these they this those through to too under until up very was wasn't we
were weren't what when where which while who whom why with won't would wouldn't
you your yours yourself yourselves
""".split())


def idf_vector(index):
    """BM25 idf per term id, as a NumPy array for O(1) lookup."""
    N = index.N
    df = index.df.astype(np.float64)
    return np.log(((N - df + 0.5) / (df + 0.5)) + 1.0)


class RM3:
    """RM3 with a pluggable feedback-term selection rule."""

    def __init__(self, body, title, forward, ext_to_int):
        self.body, self.title = body, title
        self.term_of, self.tfs_all, self.offsets = forward
        self.ext_to_int = ext_to_int
        self.idf = idf_vector(body)
        self.df_frac = body.df.astype(np.float64) / body.N
        # The index is stemmed, so its vocabulary holds stems, not words.
        # Matching a raw word list against it would silently miss almost
        # everything ("having" is stored as "have"), making the stop rule look
        # like a no-op rather than a tested idea. Push the stoplist through the
        # SAME analyser the index was built with.
        stems = set()
        for w in STOPLIST:
            stems.update(analyze(w, body.config))
        self.is_stop = np.array([t in stems for t in body.terms], dtype=bool)
        self._mass_cache = {}

    def _mass(self, query, fb_docs, k1, b):
        key = (query, fb_docs, k1, b)
        hit = self._mass_cache.get(key)
        if hit is not None:
            return hit
        out = self._mass_uncached(query, fb_docs, k1, b)
        self._mass_cache[key] = out
        return out

    def _mass_uncached(self, query, fb_docs, k1, b):
        base = {t: 1.0 for t in dict.fromkeys(analyze(query, self.body.config))}
        if not base:
            return None, None
        tot = sum(base.values())
        base = {t: v / tot for t, v in base.items()}
        docs = self._rank(base, k=fb_docs, depth=fb_docs, k1=k1, b=b)
        if not docs:
            return base, None
        w = np.array([s for _d, s in docs])
        w = w - w.min()
        w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
        rm = defaultdict(float)
        for (doc_id, _s), wd in zip(docs, w):
            di = self.ext_to_int[doc_id]
            lo, hi = self.offsets[di], self.offsets[di + 1]
            dl = max(int(self.body.doc_len[di]), 1)
            for tid, tf in zip(self.term_of[lo:hi], self.tfs_all[lo:hi]):
                rm[int(tid)] += wd * (tf / dl)
        return base, rm

    def _rank(self, terms_weighted, k, depth=None, k1=K1, b=B):
        s = np.zeros(self.body.N)
        touched = np.zeros(self.body.N, dtype=bool)
        for term, w in terms_weighted.items():
            accumulate(self.body, [term], s, touched, k1, b, w)
            accumulate(self.title, [term], s, touched, k1, b, w * TITLE_LAMBDA)
        cand = np.flatnonzero(touched)
        if cand.size == 0:
            return []
        limit = depth or k
        v = s[cand]
        if cand.size > limit:
            top = np.argpartition(-v, limit - 1)[:limit]
            cand, v = cand[top], v[top]
        order = np.lexsort((cand, -v))
        return [(self.body.doc_ids[int(cand[i])], float(v[i])) for i in order]

    def _select(self, rm, rule, fb_terms, dfcut):
        """Apply a selection rule to the RM1 mass and return the top terms."""
        if not rm:
            return []
        tids = np.fromiter(rm.keys(), dtype=np.int64, count=len(rm))
        mass = np.fromiter(rm.values(), dtype=np.float64, count=len(rm))

        keep = np.ones(tids.size, dtype=bool)
        if "dfcut" in rule:
            keep &= self.df_frac[tids] <= dfcut
        if "stop" in rule:
            keep &= ~self.is_stop[tids]
        if not keep.any():
            keep = np.ones(tids.size, dtype=bool)   # never return nothing
        tids, mass = tids[keep], mass[keep]

        if "idf" in rule:
            mass = mass * self.idf[tids]

        if tids.size > fb_terms:
            top = np.argpartition(-mass, fb_terms - 1)[:fb_terms]
            tids, mass = tids[top], mass[top]
        order = np.argsort(-mass)
        return list(zip(tids[order].tolist(), mass[order].tolist()))

    def score(self, query, rule="raw", fb_docs=FB_DOCS, fb_terms=FB_TERMS,
              alpha=ALPHA, dfcut=1.0, k1=K1, b=B, gate=None):
        base, rm = self._mass(query, fb_docs, k1, b)
        if base is None:
            return []
        if rm is None:
            return self._rank(base, k=K, k1=k1, b=b)

        if gate is not None and not gate(self, base, k1, b):
            return self._rank(base, k=K, k1=k1, b=b)   # skip expansion

        top = self._select(rm, rule, fb_terms, dfcut)
        if not top:
            return self._rank(base, k=K, k1=k1, b=b)
        tot = sum(v for _t, v in top) or 1.0
        final = defaultdict(float)
        for t, w in base.items():
            final[t] += alpha * w
        for tid, v in top:
            final[self.body.terms[tid]] += (1 - alpha) * (v / tot)
        return self._rank(dict(final), k=K, k1=k1, b=b)


def run_all(rm3, qs, qrels, **kw):
    """Per-topic nDCG@10 for one configuration."""
    out = {}
    for qid, qtext in qs:
        res = rm3.score(qtext, **kw)
        out[qid] = ndcg_at_k([d for d, _ in res], qrels[qid], K)
    return out


def mean(d):
    return float(np.mean(list(d.values())))


def report(name, scores, baseline, extra=""):
    st = paired_bootstrap(scores, baseline)
    d = mean(scores) - mean(baseline)
    wins = sum(1 for q in scores if scores[q] > baseline[q])
    loss = sum(1 for q in scores if scores[q] < baseline[q])
    tie = len(scores) - wins - loss
    print(f"  {name:<26} {mean(scores):.4f}  {d:+.4f}  p={st['p_value']:.3f}  "
          f"{wins}/{loss}/{tie} {extra}")
    return {"name": name, "ndcg": mean(scores), "delta": d,
            "p": st["p_value"], "w": wins, "l": loss, "t": tie}


def main():
    corpus = os.path.join(REPO, "data", "full", "corpus.jsonl")
    queries, qrels = load_topics()
    qs = [(q, t) for q, t in queries if q in qrels]
    qids = [q for q, _ in qs]

    print("building stemmed body + title + forward index ...", flush=True)
    pcfg = AnalysisConfig(stemmer="porter")
    pbody = get_index(corpus, config=pcfg)
    ptitle = build_field(corpus, pcfg, window(pcfg, 0, TITLE_W))
    fwd = build_forward_in_memory(pbody)
    ext_to_int = {d: i for i, d in enumerate(pbody.doc_ids)}
    rm3 = RM3(pbody, ptitle, fwd, ext_to_int)

    # Reference: the SHIPPED strategy (plain index + title, no RM3).
    plain = InvertedIndex.load(os.path.join(REPO, ".index_cache"))
    cfg = plain.config
    title_plain = build_field(corpus, cfg, window(cfg, 0, TITLE_W))
    shipped = evaluate(plain, [(title_plain, K1, B, TITLE_LAMBDA)], qs, qrels, cfg)
    print(f"\nSHIPPED baseline                 {mean(shipped):.4f}\n")

    results = {}

    # ---------------------------------------------------------------
    # A. Feedback-term selection rules, all at F30's operating point.
    # ---------------------------------------------------------------
    print("A. feedback-term selection  (nDCG  delta  p  win/loss/tie vs SHIPPED)")
    rules = [("raw", 1.0), ("idf", 1.0), ("dfcut", 0.10), ("stop", 1.0),
             ("idf+dfcut", 0.10), ("idf+stop", 1.0), ("idf+dfcut+stop", 0.10)]
    per_rule = {}
    for rule, dfcut in rules:
        sc = run_all(rm3, qs, qrels, rule=rule, dfcut=dfcut)
        per_rule[rule] = sc
        results[f"A:{rule}"] = report(rule, sc, shipped)

    # Also compare each rule against RAW rm3, which is the real control for
    # "did changing the selection rule help?"
    print("\n   same rules, measured against RAW RM3 rather than SHIPPED:")
    for rule, _ in rules:
        if rule == "raw":
            continue
        st = paired_bootstrap(per_rule[rule], per_rule["raw"])
        d = mean(per_rule[rule]) - mean(per_rule["raw"])
        print(f"  {rule:<26} {mean(per_rule[rule]):.4f}  {d:+.4f}  "
              f"p={st['p_value']:.3f}")

    best_rule = max(per_rule, key=lambda r: mean(per_rule[r]))
    best_dfcut = dict(rules)[best_rule]
    print(f"\n   best in-sample rule: {best_rule}  "
          f"({mean(per_rule[best_rule]):.4f}) -- IN-SAMPLE, not yet honest\n")

    # ---------------------------------------------------------------
    # B. df cutoff sweep under the best rule family.
    # ---------------------------------------------------------------
    if "dfcut" in best_rule:
        print("B. df-cutoff sweep (fraction of corpus above which a term is barred)")
        for cut in (0.02, 0.05, 0.10, 0.20, 0.35, 0.50):
            sc = run_all(rm3, qs, qrels, rule=best_rule, dfcut=cut)
            results[f"B:dfcut={cut}"] = report(f"dfcut={cut}", sc, shipped)
        print()

    # ---------------------------------------------------------------
    # C. (fb_docs, fb_terms, alpha) re-sweep under the best rule.
    # ---------------------------------------------------------------
    print(f"C. hyperparameter sweep under rule={best_rule}")
    grid = []
    for fd in (5, 10, 15, 20):
        for ft in (10, 20, 40, 60):
            for al in (0.5, 0.6, 0.7, 0.8):
                grid.append((fd, ft, al))
    rows = []
    for fd, ft, al in grid:
        sc = run_all(rm3, qs, qrels, rule=best_rule, dfcut=best_dfcut,
                     fb_docs=fd, fb_terms=ft, alpha=al)
        rows.append(((fd, ft, al), sc))
    rows.sort(key=lambda r: -mean(r[1]))
    print("   top 8 in-sample:")
    for (fd, ft, al), sc in rows[:8]:
        results[f"C:{fd}-{ft}-{al}"] = report(
            f"fb_docs={fd} terms={ft} a={al}", sc, shipped)
    best_hp, best_hp_scores = rows[0]
    print()

    # ---------------------------------------------------------------
    # D. k1/b re-tune, because expanded queries are not the tuned-for queries.
    # ---------------------------------------------------------------
    fd, ft, al = best_hp
    print(f"D. k1/b re-tune at rule={best_rule}, fb=({fd},{ft},{al})")
    kb_rows = []
    for k1 in (2.5, 3.5, 4.5, 5.5, 7.0):
        for b in (0.3, 0.45, 0.6, 0.75):
            sc = run_all(rm3, qs, qrels, rule=best_rule, dfcut=best_dfcut,
                         fb_docs=fd, fb_terms=ft, alpha=al, k1=k1, b=b)
            kb_rows.append(((k1, b), sc))
    kb_rows.sort(key=lambda r: -mean(r[1]))
    print("   top 5 in-sample:")
    for (k1, b), sc in kb_rows[:5]:
        results[f"D:k1={k1},b={b}"] = report(f"k1={k1} b={b}", sc, shipped)
    print()

    # ---------------------------------------------------------------
    # E. HONEST evaluation: nested CV over the whole selection procedure.
    #    Everything above is in-sample and therefore inflated. This re-runs
    #    the entire choice (rule, dfcut, fb params) inside each fold and
    #    scores only on the held-out fold.
    # ---------------------------------------------------------------
    print("E. nested cross-validation of the FULL selection procedure")
    cache = {}

    def cached(**kw):
        key = tuple(sorted(kw.items()))
        if key not in cache:
            cache[key] = run_all(rm3, qs, qrels, **kw)
        return cache[key]

    candidates = []
    for rule, dc in rules:
        candidates.append({"rule": rule, "dfcut": dc})
    for cut in (0.02, 0.05, 0.20):
        candidates.append({"rule": best_rule, "dfcut": cut})
    for (fd_, ft_, al_), _ in rows[:6]:
        candidates.append({"rule": best_rule, "dfcut": best_dfcut,
                           "fb_docs": fd_, "fb_terms": ft_, "alpha": al_})

    held = {}
    picks = []
    for test in cv_folds(qids, 5):
        train = [q for q in qids if q not in test]
        best_c, best_m = None, -1
        for c in candidates:
            sc = cached(**c)
            m = float(np.mean([sc[q] for q in train]))
            if m > best_m:
                best_c, best_m = c, m
        picks.append(best_c)
        sc = cached(**best_c)
        for q in test:
            held[q] = sc[q]

    st = paired_bootstrap(held, shipped)
    d = mean(held) - mean(shipped)
    wins = sum(1 for q in held if held[q] > shipped[q])
    loss = sum(1 for q in held if held[q] < shipped[q])
    print(f"   honest nested-CV nDCG@10   {mean(held):.4f}")
    print(f"   vs SHIPPED                 {d:+.4f}   p={st['p_value']:.3f}   "
          f"{wins}/{loss}/{len(held)-wins-loss}")
    print(f"   fold picks: {[dict(p) for p in picks]}")

    # And against the CURRENT rm3_stemmed, which is what it must beat to ship.
    raw_rm3 = per_rule["raw"]
    st2 = paired_bootstrap(held, raw_rm3)
    print(f"   vs current RM3 (raw)       {mean(held)-mean(raw_rm3):+.4f}   "
          f"p={st2['p_value']:.3f}")

    results["E:honest_nested_cv"] = {
        "ndcg": mean(held), "delta_vs_shipped": d, "p_vs_shipped": st["p_value"],
        "delta_vs_raw_rm3": mean(held) - mean(raw_rm3), "p_vs_raw_rm3": st2["p_value"],
        "fold_picks": [dict(p) for p in picks],
    }
    results["_baselines"] = {"shipped": mean(shipped), "raw_rm3": mean(raw_rm3)}

    with open(os.path.join(REPO, "experiments", "rm3_refine.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwrote experiments/rm3_refine.json")


if __name__ == "__main__":
    main()
