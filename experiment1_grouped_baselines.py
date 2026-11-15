#!/usr/bin/env python3
"""
Experiment (1): how much relation-inference headroom does this task really
have? Group test edges by how many manufacturers (Web-keyed, corrected)
share the edge's product string -- {1, 2-5, 6-19, high_frequency(>=20)} -- and
report simple, training-free baselines (product popularity degree; text
cosine similarity reusing Graph_A's own TF-IDF-SVD node features) on the
SAME frozen split used for every other method in this round.

High-frequency threshold (>=20 manufacturers) is a DECLARED bucketing rule,
NOT a semantic audit -- we have not systematically verified that high-sharing
strings are generic/category words rather than genuinely common specific
products (a handful of manual spot-checks suggested a mix; this needs a
proper semantic audit before being used to justify excluding/down-weighting
these edges). Bucketing uses FULL-graph product-sharing counts (computed
over the whole CSV); baseline SCORES below use TRAIN-split-only degree --
these are deliberately different quantities, logged explicitly per bucket
to make the distinction auditable (see the degree-convention diagnostic
printed below).
"""
import csv
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import roc_auc_score

GENERIC_THRESHOLD = 20


def canon(w):
    w = str(w).strip().lower()
    for p in ("https://", "http://"):
        if w.startswith(p):
            w = w[len(p):]
    return w.rstrip("/")


def load_product_sharing_counts():
    prod_to_manus = defaultdict(set)
    with open("Cleaned_Master_CSV_v3.csv", newline='', encoding='utf-8-sig', errors='replace') as f:
        reader = csv.DictReader(f)
        cols = [f"Product_{i}" for i in range(1, 151)]
        for r in reader:
            m = canon(r.get("Web", ""))
            for c in cols:
                p = r.get(c)
                if p and p.strip() and p.strip().lower() != "yes":
                    prod_to_manus[p.strip()].add(m)
    return {p: len(ms) for p, ms in prod_to_manus.items()}


def bucket_for_count(c):
    if c <= 1:
        return "1_singleton"
    if c <= 5:
        return "2-5"
    if c < GENERIC_THRESHOLD:
        return "6-19"
    return f"high_frequency_{GENERIC_THRESHOLD}+"


def main():
    split = torch.load("frozen_split_GraphA_seed42.pt", map_location="cpu", weights_only=False)
    data = torch.load("hetero_graphs/Graph_A.pt", map_location="cpu", weights_only=False)
    train_data, val_data, test_data = split["train"], split["val"], split["test"]

    rel = ("manufacturer", "has_product", "product")

    # recover product id -> string mapping in the SAME order used to build Graph_A
    # (build_edges sorts the edge set, then takes sorted unique product strings)
    edge_set = set()
    with open("Cleaned_Master_CSV_v3.csv", newline='', encoding='utf-8-sig', errors='replace') as f:
        reader = csv.DictReader(f)
        cols = [f"Product_{i}" for i in range(1, 151)]
        for r in reader:
            m = canon(r.get("Web", ""))
            for c in cols:
                p = r.get(c)
                if p and p.strip() and p.strip().lower() != "yes":
                    edge_set.add((m, p.strip()))
    products_sorted = sorted({p for _, p in edge_set})
    assert len(products_sorted) == data['product'].x.size(0), \
        f"product count mismatch: {len(products_sorted)} vs {data['product'].x.size(0)}"
    idx2prod = products_sorted

    sharing_counts = load_product_sharing_counts()
    prod_idx_to_bucket = np.array([bucket_for_count(sharing_counts.get(p, 0)) for p in idx2prod], dtype=object)

    # ---- training-set product popularity (degree) ----------------------
    train_pos_mask = train_data[rel].edge_label == 1
    train_pos_idx = train_data[rel].edge_label_index[:, train_pos_mask]
    prod_degree = np.zeros(data['product'].x.size(0), dtype=np.int64)
    for p in train_pos_idx[1].numpy():
        prod_degree[p] += 1

    # ---- text-similarity baseline: reuse Graph_A's own TF-IDF-SVD features --
    manu_x = data['manufacturer'].x.numpy()
    prod_x = data['product'].x.numpy()
    manu_x_n = manu_x / (np.linalg.norm(manu_x, axis=1, keepdims=True) + 1e-9)
    prod_x_n = prod_x / (np.linalg.norm(prod_x, axis=1, keepdims=True) + 1e-9)

    # ---- evaluate on the frozen test split, grouped by bucket -----------
    idx = test_data[rel].edge_label_index.numpy()
    labels = test_data[rel].edge_label.numpy()
    m_idx, p_idx = idx[0], idx[1]

    score_degree = prod_degree[p_idx].astype(np.float64)
    score_textsim = np.sum(manu_x_n[m_idx] * prod_x_n[p_idx], axis=1)

    buckets = prod_idx_to_bucket[p_idx]

    print("NOTE: buckets are assigned by FULL-graph product-sharing count; "
          "score_degree is computed from TRAIN-split-only degree -- these are "
          "deliberately different quantities (bucketing describes the task's "
          "structure; train-degree is what a popularity baseline could actually "
          "see at train time). Diagnostic, per bucket: fraction of positives/"
          "negatives with train-degree==0:")
    for b in sorted(set(buckets)):
        mask = buckets == b
        lb, deg = labels[mask], prod_degree[p_idx[mask]]
        pos_zero = (deg[lb == 1] == 0).mean() if (lb == 1).any() else float('nan')
        neg_zero = (deg[lb == 0] == 0).mean() if (lb == 0).any() else float('nan')
        print(f"  {b:<16} pos train-deg==0 frac={pos_zero:.4f}  neg train-deg==0 frac={neg_zero:.4f}")

    print(f"\ntest set size: {len(labels)}  (high_frequency threshold = {GENERIC_THRESHOLD} manufacturers)")
    print(f"{'bucket':<16} {'n':>7} {'pos_rate':>9} {'AUC_degree':>11} {'AUC_textsim':>12}")
    results = []
    for b in sorted(set(buckets)):
        mask = buckets == b
        n = mask.sum()
        lb = labels[mask]
        if len(set(lb)) < 2:
            auc_d, auc_t = float('nan'), float('nan')
        else:
            auc_d = roc_auc_score(lb, score_degree[mask])
            auc_t = roc_auc_score(lb, score_textsim[mask])
        print(f"{b:<16} {n:>7} {lb.mean():>9.4f} {auc_d:>11.4f} {auc_t:>12.4f}")
        results.append((b, int(n), float(lb.mean()), auc_d, auc_t))

    auc_d_all = roc_auc_score(labels, score_degree)
    auc_t_all = roc_auc_score(labels, score_textsim)
    print(f"{'ALL':<16} {len(labels):>7} {labels.mean():>9.4f} {auc_d_all:>11.4f} {auc_t_all:>12.4f}")

    with open("experiment1_baseline_results.md", "w", encoding="utf-8") as f:
        f.write("# Experiment 1: grouped evaluation + simple baselines (Graph_A, frozen split, seed=42)\n\n")
        f.write(f"Generic-term threshold: products shared by >= {GENERIC_THRESHOLD} manufacturers "
                f"(declared audit rule, not a statistical cutoff -- see script docstring).\n\n")
        f.write("| Bucket | n | pos rate | AUC (product-degree baseline) | AUC (text-similarity baseline) |\n")
        f.write("|---|---|---|---|---|\n")
        for b, n, pr, ad, at in results:
            f.write(f"| {b} | {n} | {pr:.4f} | {ad:.4f} | {at:.4f} |\n")
        f.write(f"| **ALL** | {len(labels)} | {labels.mean():.4f} | {auc_d_all:.4f} | {auc_t_all:.4f} |\n")

    print("\nwrote experiment1_baseline_results.md")


if __name__ == "__main__":
    main()
