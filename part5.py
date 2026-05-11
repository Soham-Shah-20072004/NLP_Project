#!/usr/bin/env python3
"""
Part 5: Improving the VSM IR System on the Cranfield dataset.

Implements and evaluates four retrieval systems:
  1. Baseline TF-IDF VSM (from informationRetrieval.py)
  2. BM25 ranking
  3. Latent Semantic Analysis (LSA) via truncated SVD
  4. Query Expansion via WordNet  (from part5_ideas.tex)

Run:
    python part5.py
"""

import json
import math
import time
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import svds
from scipy.stats import wilcoxon

import nltk
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer

from informationRetrieval import InformationRetrieval
from evaluation import Evaluation
from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval

DATASET   = "cranfield"
OUT_DIR   = "output_part5"
K_MAX     = 10          

BM25_K1   = 1.5
BM25_B    = 0.75

LSA_DIMS  = [50, 100, 200, 300]

QE_LAMBDA = 0.5 # synonym weight discount factor

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)

def preprocess(texts):
    """
    Full preprocessing pipeline: segmentation → tokenization →
    inflection reduction (Porter stemming) → stopword removal.

    Returns
    -------
    processed : list[list[list[str]]]
        Fully preprocessed texts (list of docs/queries, each is
        a list of sentences, each sentence is a list of tokens).
    raw_tokens : list[list[list[str]]]
        Tokenised but NOT stemmed (used for WordNet expansion).
    """
    segmenter = SentenceSegmentation()
    tokenizer = Tokenization()
    reducer   = InflectionReduction()
    remover   = StopwordRemoval()

    segs       = [segmenter.punkt(t) for t in texts]
    toks       = [tokenizer.pennTreeBank(s) for s in segs]
    reduced    = [reducer.reduce(t) for t in toks]
    clean      = [remover.fromList(r) for r in reduced]
    return clean, toks


def load_cranfield():
    with open(os.path.join(DATASET, "cran_docs.json")) as f:
        docs_json = json.load(f)
    with open(os.path.join(DATASET, "cran_queries.json")) as f:
        queries_json = json.load(f)
    with open(os.path.join(DATASET, "cran_qrels.json")) as f:
        qrels = json.load(f)

    doc_ids   = [item["id"]           for item in docs_json]
    doc_texts = [item["body"]         for item in docs_json]
    query_ids = [item["query number"] for item in queries_json]
    query_texts = [item["query"]      for item in queries_json]

    return doc_ids, doc_texts, query_ids, query_texts, qrels

# BM25
class BM25Retrieval:
    """BM25 ranking with Okapi BM25 score (Robertson et al.)."""

    def __init__(self, k1=BM25_K1, b=BM25_B):
        self.k1 = k1
        self.b  = b

    def buildIndex(self, docs, doc_ids):
        self.doc_ids = doc_ids
        N = len(docs)
        df = defaultdict(int)
        self.doc_tf  = []
        self.doc_len = []

        for doc in docs:
            terms = [t for sent in doc for t in sent]
            tf = Counter(terms)
            self.doc_tf.append(tf)
            self.doc_len.append(sum(tf.values()))
            for term in tf:
                df[term] += 1

        self.avgdl = sum(self.doc_len) / N if N > 0 else 1.0
        # BM25 IDF (Robertson-Sparck Jones variant, always positive)
        self.idf = {
            term: math.log((N - cnt + 0.5) / (cnt + 0.5) + 1.0)
            for term, cnt in df.items()
        }

    def rank(self, queries):
        results = []
        for query in queries:
            terms = [t for sent in query for t in sent]
            scores = []
            for idx, tf in enumerate(self.doc_tf):
                dl    = self.doc_len[idx]
                score = 0.0
                for term in terms:
                    if term not in self.idf:
                        continue
                    f = tf.get(term, 0)
                    score += self.idf[term] * (
                        f * (self.k1 + 1)
                    ) / (
                        f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                    )
                scores.append((score, self.doc_ids[idx]))
            scores.sort(key=lambda x: x[0], reverse=True)
            results.append([d for _, d in scores])
        return results

# LSA
class LSARetrieval:
    """
    Latent Semantic Analysis via truncated SVD.
    Documents and queries are projected into a k-dimensional
    latent space; retrieval uses cosine similarity there.
    """

    def __init__(self, n_components=100):
        self.k = n_components

    def buildIndex(self, docs, doc_ids):
        self.doc_ids = doc_ids

        # Build vocabulary from all preprocessed docs
        vocab = sorted({t for doc in docs for sent in doc for t in sent})
        self.vocab    = vocab
        self.term2idx = {t: i for i, t in enumerate(vocab)}
        m, n = len(vocab), len(docs)

        # TF-IDF term×doc matrix
        df = defaultdict(int)
        doc_tfs = []
        for doc in docs:
            terms = [t for sent in doc for t in sent]
            tf = Counter(terms)
            doc_tfs.append(tf)
            for term in tf:
                df[term] += 1

        idf = {t: math.log10(n / df[t]) for t in df}

        A = lil_matrix((m, n), dtype=np.float32)
        for j, tf in enumerate(doc_tfs):
            for term, cnt in tf.items():
                i = self.term2idx[term]
                A[i, j] = cnt * idf[term]
        A = A.tocsr()

        k_actual = min(self.k, m - 1, n - 1)
        U, S, Vt = svds(A, k=k_actual)          # U:(m,k) S:(k,) Vt:(k,n)

        # Sort singular values descending
        order     = np.argsort(S)[::-1]
        self.U    = U[:, order]                   # (m, k)
        self.S    = S[order]                      # (k,)
        self.Vt   = Vt[order, :]                  # (k, n)

        # Document vectors in latent space: rows of V * Sigma → (n, k)
        self.doc_vecs  = (np.diag(self.S) @ self.Vt).T
        self.doc_norms = np.linalg.norm(self.doc_vecs, axis=1) + 1e-10

    def rank(self, queries):
        results = []
        m = len(self.vocab)
        for query in queries:
            terms = [t for sent in query for t in sent]
            q_vec = np.zeros(m, dtype=np.float32)
            for t in terms:
                if t in self.term2idx:
                    q_vec[self.term2idx[t]] += 1.0

            # Fold into latent space: q_lat = q^T U Sigma^{-1}
            q_lat  = (q_vec @ self.U) / (self.S + 1e-10)   # (k,)
            q_norm = np.linalg.norm(q_lat) + 1e-10

            cosines = (self.doc_vecs @ q_lat) / (self.doc_norms * q_norm)
            order   = np.argsort(cosines)[::-1]
            results.append([self.doc_ids[i] for i in order])
        return results


# WordNet based query expansion
class QueryExpansionRetrieval:
    """
    TF-IDF VSM with WordNet-based query expansion.
    For each raw query token, synonyms from WordNet are added
    with a discounted weight λ < 1.
    """

    def __init__(self, lam=QE_LAMBDA):
        self.lam     = lam
        self.stemmer = PorterStemmer()
        self.ir      = InformationRetrieval()

    def buildIndex(self, docs, doc_ids):
        self.ir.buildIndex(docs, doc_ids)

    def _expand_weights(self, raw_token_sents):
        """
        Build a term→weight dict for one query using WordNet synonyms.
        raw_token_sents: list of lists of tokens (tokenised, not stemmed).
        """
        weights = defaultdict(float)
        for sent in raw_token_sents:
            for token in sent:
                stemmed = self.stemmer.stem(token.lower())
                weights[stemmed] = max(weights[stemmed], 1.0)
                # Add synonyms from all WordNet synsets
                for syn in wordnet.synsets(token.lower()):
                    for lemma in syn.lemmas():
                        syn_stem = self.stemmer.stem(
                            lemma.name().replace("_", " ").lower()
                        )
                        weights[syn_stem] = max(weights[syn_stem], self.lam)
        return weights

    def rank(self, queries_processed, queries_raw_tokens):
        """
        Parameters
        ----------
        queries_processed   : preprocessed (stemmed+filtered) queries
        queries_raw_tokens  : tokenised-only queries (for WordNet lookup)
        """
        idf         = self.ir.idf
        doc_vectors = self.ir.doc_vectors
        doc_norms   = self.ir.doc_norms
        doc_ids     = self.ir.doc_ids
        results     = []

        for q_raw in queries_raw_tokens:
            expanded = self._expand_weights(q_raw)

            # Build TF-IDF query vector over expanded terms
            q_vec  = {}
            q_norm_sq = 0.0
            for term, w in expanded.items():
                if term in idf:
                    val = w * idf[term]
                    q_vec[term] = val
                    q_norm_sq  += val * val
            q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0 else 0.0

            scores = []
            for idx, dv in enumerate(doc_vectors):
                dn = doc_norms[idx]
                if q_norm == 0 or dn == 0:
                    score = 0.0
                else:
                    dot   = sum(q_vec.get(t, 0) * dv.get(t, 0) for t in q_vec)
                    score = dot / (q_norm * dn)
                scores.append((score, doc_ids[idx]))

            scores.sort(key=lambda x: x[0], reverse=True)
            results.append([d for _, d in scores])
        return results


### evaluation functions ### 

def per_query_ap(ranked_results, query_ids, qrels, k, ev):
    """Returns list of per-query AP@k values (one float per query with ≥1 relevant doc)."""
    aps = []
    for i, qid in enumerate(query_ids):
        true_ids = ev._get_true_doc_IDs(qid, qrels)
        if not true_ids:
            continue
        aps.append(ev.queryAveragePrecision(ranked_results[i], qid, true_ids, k))
    return aps


def evaluate_at_all_k(ranked_results, query_ids, qrels, ev, k_max=K_MAX):
    """Returns dict metric → list of mean values at k = 1 … k_max."""
    metrics = {m: [] for m in ["map", "mrr", "precision", "recall", "fscore", "ndcg"]}
    for k in range(1, k_max + 1):
        metrics["map"].append(ev.meanAveragePrecision(ranked_results, query_ids, qrels, k))
        metrics["mrr"].append(ev.meanReciprocalRank(ranked_results, query_ids, qrels, k))
        metrics["precision"].append(ev.meanPrecision(ranked_results, query_ids, qrels, k))
        metrics["recall"].append(ev.meanRecall(ranked_results, query_ids, qrels, k))
        metrics["fscore"].append(ev.meanFscore(ranked_results, query_ids, qrels, k))
        metrics["ndcg"].append(ev.meanNDCG(ranked_results, query_ids, qrels, k))
    return metrics


def count_zero_result(ranked_results, query_ids, qrels, ev):
    """Queries for which no relevant doc appears in any retrieved position."""
    zero = 0
    for i, qid in enumerate(query_ids):
        true_ids = ev._get_true_doc_IDs(qid, qrels)
        if not true_ids:
            continue
        rr = ev.queryReciprocalRank(ranked_results[i], qid, true_ids, len(ranked_results[i]))
        if rr == 0.0:
            zero += 1
    return zero


def wilcoxon_pvalue(ap_a, ap_b):
    """Two-sided Wilcoxon signed-rank test on paired AP vectors."""
    diffs = [a - b for a, b in zip(ap_a, ap_b)]
    if all(d == 0 for d in diffs):
        return 1.0
    _, p = wilcoxon(ap_a, ap_b)
    return p


### Plotting functions ###

def plot_system_metrics(metrics_dict, title, fname):
    """
    For a single system, plot all six metrics as a function of k.
    """
    ks = list(range(1, K_MAX + 1))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    names = {
        "precision": "Precision@k",
        "recall":    "Recall@k",
        "fscore":    "F0.5@k",
        "map":       "MAP@k",
        "ndcg":      "nDCG@k",
        "mrr":       "MRR@k",
    }
    for ax, (key, label) in zip(axes, names.items()):
        ax.plot(ks, metrics_dict[key], marker="o", linewidth=1.5)
        ax.set_title(label)
        ax.set_xlabel("k")
        ax.set_xticks(ks)
        ax.grid(True, alpha=0.4)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "figures", fname)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_metric_comparison(all_metrics, metric, ylabel, title, fname):
    """
    Compare a single metric across multiple systems.
    all_metrics : dict  system_name → metrics_dict
    """
    ks = list(range(1, K_MAX + 1))
    plt.figure(figsize=(8, 5))
    for name, m in all_metrics.items():
        plt.plot(ks, m[metric], marker="o", label=name, linewidth=1.5)
    plt.xlabel("k")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(ks)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "figures", fname)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_lsa_sweep(lsa_maps, chosen_k):
    """Bar chart of MAP@10 vs LSA dimension k."""
    ks   = list(lsa_maps.keys())
    maps = list(lsa_maps.values())
    plt.figure(figsize=(6, 4))
    bars = plt.bar([str(k) for k in ks], maps, color="steelblue")
    idx  = ks.index(chosen_k)
    bars[idx].set_color("tomato")
    plt.xlabel("Number of LSA dimensions")
    plt.ylabel("MAP@10")
    plt.title("LSA: MAP@10 vs number of latent dimensions")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "figures", "lsa_k_sweep.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


## should remove in the final version!
def print_table(rows, headers, title=""):
    col_widths = [max(len(h), max(len(str(r[i])) for r in rows))
                  for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"
    if title:
        print(f"\n{'─'*len(sep)}")
        print(f"  {title}")
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))
    print(sep)


def fmt(x, decimals=4):
    return f"{x:.{decimals}f}"


def main():
    # load data
    print("\nLoading and preprocessing Cranfield dataset …")
    doc_ids, doc_texts, query_ids, query_texts, qrels = load_cranfield()
    print(f"  Documents : {len(doc_ids)}")
    print(f"  Queries   : {len(query_ids)}")
    print(f"  Qrels     : {len(qrels)}")

    # preprocessing
    t0 = time.time()
    docs_processed, _ = preprocess(doc_texts)
    queries_processed, queries_raw = preprocess(query_texts)
    preprocess_time = time.time() - t0
    print(f"  Preprocessing done in {preprocess_time:.1f}s")

    ev = Evaluation()
    print("\nBaseline TF-IDF VSM …")
    vsm = InformationRetrieval()
    t0 = time.time()
    vsm.buildIndex(docs_processed, doc_ids)
    vsm_ranked = vsm.rank(queries_processed)
    vsm_time   = time.time() - t0
    vsm_metrics = evaluate_at_all_k(vsm_ranked, query_ids, qrels, ev)
    vsm_ap10    = per_query_ap(vsm_ranked, query_ids, qrels, K_MAX, ev)
    vsm_zero    = count_zero_result(vsm_ranked, query_ids, qrels, ev)
    print(f"  MAP@10={fmt(vsm_metrics['map'][-1])}  MRR@10={fmt(vsm_metrics['mrr'][-1])}"
          f"  nDCG@10={fmt(vsm_metrics['ndcg'][-1])}  time={vsm_time:.2f}s")
    plot_system_metrics(vsm_metrics, "Baseline TF-IDF VSM", "vsm_metrics.png")

    # BM25
    print("\nBM25 (k1={}, b={}) …".format(BM25_K1, BM25_B))
    bm25 = BM25Retrieval()
    t0 = time.time()
    bm25.buildIndex(docs_processed, doc_ids)
    bm25_ranked  = bm25.rank(queries_processed)
    bm25_time    = time.time() - t0
    bm25_metrics = evaluate_at_all_k(bm25_ranked, query_ids, qrels, ev)
    bm25_ap10    = per_query_ap(bm25_ranked, query_ids, qrels, K_MAX, ev)
    bm25_zero    = count_zero_result(bm25_ranked, query_ids, qrels, ev)
    bm25_p       = wilcoxon_pvalue(bm25_ap10, vsm_ap10)
    print(f"  MAP@10={fmt(bm25_metrics['map'][-1])}  MRR@10={fmt(bm25_metrics['mrr'][-1])}"
          f"  nDCG@10={fmt(bm25_metrics['ndcg'][-1])}  time={bm25_time:.2f}s")
    print(f"  Wilcoxon vs VSM  p={bm25_p:.4f}  "
          f"{'SIGNIFICANT' if bm25_p < 0.05 else 'not significant'} at α=0.05")
    plot_system_metrics(bm25_metrics, "BM25", "bm25_metrics.png")

    # LSA - check for different k
    print("\nLSA – sweeping k ∈ {} …".format(LSA_DIMS))
    lsa_map10 = {}
    lsa_results_all = {}
    for k_dim in LSA_DIMS:
        lsa = LSARetrieval(n_components=k_dim)
        t0  = time.time()
        lsa.buildIndex(docs_processed, doc_ids)
        ranked = lsa.rank(queries_processed)
        elapsed = time.time() - t0
        m10 = ev.meanAveragePrecision(ranked, query_ids, qrels, K_MAX)
        lsa_map10[k_dim]    = m10
        lsa_results_all[k_dim] = (lsa, ranked, elapsed)
        print(f"  k={k_dim:>3d}  MAP@10={fmt(m10)}  time={elapsed:.1f}s")

    best_k = max(lsa_map10, key=lsa_map10.get)
    print(f"  → Best k = {best_k}  (MAP@10 = {fmt(lsa_map10[best_k])})")
    plot_lsa_sweep(lsa_map10, best_k)

    lsa_best, lsa_ranked, lsa_time = lsa_results_all[best_k]
    lsa_metrics = evaluate_at_all_k(lsa_ranked, query_ids, qrels, ev)
    lsa_ap10    = per_query_ap(lsa_ranked, query_ids, qrels, K_MAX, ev)
    lsa_zero    = count_zero_result(lsa_ranked, query_ids, qrels, ev)
    lsa_p       = wilcoxon_pvalue(lsa_ap10, vsm_ap10)
    print(f"  MAP@10={fmt(lsa_metrics['map'][-1])}  MRR@10={fmt(lsa_metrics['mrr'][-1])}"
          f"  nDCG@10={fmt(lsa_metrics['ndcg'][-1])}")
    print(f"  Wilcoxon vs VSM  p={lsa_p:.4f}  "
          f"{'SIGNIFICANT' if lsa_p < 0.05 else 'not significant'} at α=0.05")
    plot_system_metrics(lsa_metrics, f"LSA (k={best_k})", "lsa_metrics.png")

    # WordNet Query Expansion
    print("\nWordNet Query Expansion (λ={}) …".format(QE_LAMBDA))
    qe = QueryExpansionRetrieval()
    t0 = time.time()
    qe.buildIndex(docs_processed, doc_ids)
    qe_ranked  = qe.rank(queries_processed, queries_raw)
    qe_time    = time.time() - t0
    qe_metrics = evaluate_at_all_k(qe_ranked, query_ids, qrels, ev)
    qe_ap10    = per_query_ap(qe_ranked, query_ids, qrels, K_MAX, ev)
    qe_zero    = count_zero_result(qe_ranked, query_ids, qrels, ev)
    qe_p       = wilcoxon_pvalue(qe_ap10, vsm_ap10)
    print(f"  MAP@10={fmt(qe_metrics['map'][-1])}  MRR@10={fmt(qe_metrics['mrr'][-1])}"
          f"  nDCG@10={fmt(qe_metrics['ndcg'][-1])}  time={qe_time:.2f}s")
    print(f"  Zero-result queries: VSM={vsm_zero}  QE={qe_zero}")
    print(f"  Wilcoxon vs VSM  p={qe_p:.4f}  "
          f"{'SIGNIFICANT' if qe_p < 0.05 else 'not significant'} at α=0.05")
    plot_system_metrics(qe_metrics, "WordNet Query Expansion", "qe_metrics.png")

    
    # plot comparison
    all_metrics = {
        "VSM (baseline)":   vsm_metrics,
        "BM25":             bm25_metrics,
        f"LSA (k={best_k})": lsa_metrics,
        "Query Expansion":  qe_metrics,
    }
    plot_metric_comparison(all_metrics, "map",  "MAP@k",   "All systems: MAP@k",   "all_map.png")
    plot_metric_comparison(all_metrics, "ndcg", "nDCG@k",  "All systems: nDCG@k",  "all_ndcg.png")
    plot_metric_comparison(all_metrics, "precision", "P@k", "All systems: P@k",    "all_precision.png")
    plot_metric_comparison(all_metrics, "recall",    "R@k", "All systems: R@k",    "all_recall.png")

    print()
    rows = [
        ["VSM (baseline)",       fmt(vsm_metrics["map"][-1]),
                                  fmt(vsm_metrics["mrr"][-1]),
                                  fmt(vsm_metrics["precision"][-1]),
                                  fmt(vsm_metrics["recall"][-1]),
                                  fmt(vsm_metrics["ndcg"][-1]),
                                  str(vsm_zero),   f"{vsm_time:.2f}"],
        ["BM25",                  fmt(bm25_metrics["map"][-1]),
                                  fmt(bm25_metrics["mrr"][-1]),
                                  fmt(bm25_metrics["precision"][-1]),
                                  fmt(bm25_metrics["recall"][-1]),
                                  fmt(bm25_metrics["ndcg"][-1]),
                                  str(bm25_zero),  f"{bm25_time:.2f}"],
        [f"LSA k={best_k}",       fmt(lsa_metrics["map"][-1]),
                                  fmt(lsa_metrics["mrr"][-1]),
                                  fmt(lsa_metrics["precision"][-1]),
                                  fmt(lsa_metrics["recall"][-1]),
                                  fmt(lsa_metrics["ndcg"][-1]),
                                  str(lsa_zero),   f"{lsa_time:.2f}"],
        ["Query Expansion",       fmt(qe_metrics["map"][-1]),
                                  fmt(qe_metrics["mrr"][-1]),
                                  fmt(qe_metrics["precision"][-1]),
                                  fmt(qe_metrics["recall"][-1]),
                                  fmt(qe_metrics["ndcg"][-1]),
                                  str(qe_zero),    f"{qe_time:.2f}"],
   ]
    headers = ["System", "MAP@10", "MRR@10", "P@10", "R@10", "nDCG@10", "Zero", "Time(s)"]
    print_table(rows, headers, title="Summary: All Systems on Cranfield")

    # Wilcoxon p-values table
    pval_rows = [
        ["BM25 vs VSM", f"{bm25_p:.4f}", "YES" if bm25_p < 0.05 else "no"],
        [f"LSA k={best_k} vs VSM", f"{lsa_p:.4f}", "YES" if lsa_p < 0.05 else "no"],
        ["QE vs VSM", f"{qe_p:.4f}",   "YES" if qe_p < 0.05 else "no"],
    ]
    print_table(pval_rows, ["Comparison", "p-value", "Sig. (α=0.05)"],
                title="Wilcoxon Signed-Rank Tests (AP@10)")

    # ── Save full results as JSON ──────────────────────────
    results_json = {
        "vsm":  {**{m: vsm_metrics[m]  for m in vsm_metrics},
                 "zero_results": vsm_zero,  "time": vsm_time},
        "bm25": {**{m: bm25_metrics[m] for m in bm25_metrics},
                 "zero_results": bm25_zero, "time": bm25_time, "p_vs_vsm": bm25_p},
        "lsa":  {**{m: lsa_metrics[m]  for m in lsa_metrics},
                 "zero_results": lsa_zero,  "time": lsa_time,
                 "best_k": best_k, "p_vs_vsm": lsa_p,
                 "k_sweep": {k: v for k, v in lsa_map10.items()}},
        "qe":   {**{m: qe_metrics[m]   for m in qe_metrics},
                 "zero_results": qe_zero,   "time": qe_time,  "p_vs_vsm": qe_p},
    }
    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nFull results saved to {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
