#!/usr/bin/env python3
"""
Corrected text-similarity baseline for Task A.

BUG FOUND AND FIXED THIS ROUND: the `textsim` baseline in
experiment1_grouped_baselines.py / experiment1_degree_matched_robustness.py
computes cosine similarity between `data['manufacturer'].x` and
`data['product'].x` -- but these are NOT a shared embedding space. Checked
directly against Graph-A-Builder.py: the manufacturer feature is built from
its OWN TfidfVectorizer (fit on Name + NAICS_Description_1) then reduced by
its OWN TruncatedSVD fit on the full manufacturer feature matrix (TF-IDF +
one-hots + numeric); the product feature is built from a COMPLETELY SEPARATE
TfidfVectorizer (fit on raw product strings) then a COMPLETELY SEPARATE
TruncatedSVD fit only on that. Both end up 64-dim by coincidence of the
reduction target, not because they share a basis -- dimension i of one space
has no semantic correspondence to dimension i of the other. Cosine similarity
between them is therefore not a meaningful text-similarity signal; the old
`textsim` numbers in experiment1_baseline_results.md /
experiment1_degree_matched_results.md are WRONG and should not be cited.

FIX (this script): fit ONE shared TfidfVectorizer on the union of
manufacturer text (Name + NAICS_Description_1, the same two fields used to
build the manufacturer feature) and all product strings, so both sides are
projected into the identical vocabulary/coordinate space. No SVD (per
reviewer suggestion -- not needed for a simple baseline, and avoids
re-introducing a second place where the two sides could silently diverge).
Evaluated under BOTH negative-sampling protocols already used in Section 5.1
(random negatives from the frozen split; degree-matched negatives using the
IDENTICAL matching logic -- same train-degree definition, same 3 resampling
seeds -- duplicated from experiment1_degree_matched_robustness.py so this
baseline is scored on the exact same pairs as the original one was).
"""
import csv
import random
import numpy as np
import torch
import scipy.sparse as sp
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score


def canon(w):
    w = str(w).strip().lower()
    for p in ("https://", "http://"):
        if w.startswith(p):
            w = w[len(p):]
    return w.rstrip("/")


def main():
    split = torch.load("frozen_split_GraphA_seed42.pt", map_location="cpu", weights_only=False)
    data = torch.load("hetero_graphs/Graph_A.pt", map_location="cpu", weights_only=False)
    train_data, test_data = split["train"], split["test"]
    rel = ("manufacturer", "has_product", "product")

    # ---- recover manufacturer order (m2i) and text (Name + NAICS_Description_1),
    #      and product order (p2i) and text -- same source fields as Graph-A-Builder.py
    edge_set_str = set()
    rows_by_manu = {}
    with open("Cleaned_Master_CSV_v3.csv", newline='', encoding='utf-8-sig', errors='replace') as f:
        reader = csv.DictReader(f)
        cols = [f"Product_{i}" for i in range(1, 151)]
        for r in reader:
            m = canon(r.get("Web", ""))
            rows_by_manu[m] = r
            for c in cols:
                p = r.get(c)
                if p and p.strip() and p.strip().lower() != "yes":
                    edge_set_str.add((m, p.strip()))
    manus_sorted = sorted({m for m, _ in edge_set_str})
    products_sorted = sorted({p for _, p in edge_set_str})
    assert len(manus_sorted) == data['manufacturer'].x.size(0)
    assert len(products_sorted) == data['product'].x.size(0)

    manu_text = [f"{rows_by_manu[m].get('Name','') or ''} {rows_by_manu[m].get('NAICS_Description_1','') or ''}"
                 for m in manus_sorted]
    prod_text = list(products_sorted)

    # ---- ONE shared TF-IDF space, fit on manufacturer text + product text together ----
    shared_vec = TfidfVectorizer(stop_words='english')
    shared_vec.fit(manu_text + prod_text)
    manu_tfidf = shared_vec.transform(manu_text)   # sparse [n_manu, V]
    prod_tfidf = shared_vec.transform(prod_text)    # sparse [n_prod, V]
    print(f"shared TF-IDF vocab size: {len(shared_vec.vocabulary_)}")

    def l2norm_sparse(m):
        norms = np.sqrt(m.multiply(m).sum(axis=1)).A1
        norms[norms == 0] = 1.0
        return sp.diags(1.0 / norms) @ m

    manu_n = l2norm_sparse(manu_tfidf).tocsr()
    prod_n = l2norm_sparse(prod_tfidf).tocsr()

    def score_textsim_shared(m_idx, p_idx):
        # row-wise cosine (already L2-normalized): dot product of corresponding rows
        a = manu_n[m_idx]
        b = prod_n[p_idx]
        return np.asarray(a.multiply(b).sum(axis=1)).ravel()

    # ---- random-negative test set (same frozen split as Section 5.1) ----
    idx = test_data[rel].edge_label_index.numpy()
    labels = test_data[rel].edge_label.numpy()
    m_idx, p_idx = idx[0], idx[1]

    prod_to_manus_full = defaultdict(set)
    for m, p in edge_set_str:
        prod_to_manus_full[p].add(m)
    full_count = np.array([len(prod_to_manus_full[p]) for p in products_sorted])
    singleton_mask_full = full_count[p_idx] <= 1

    score_rand = score_textsim_shared(m_idx, p_idx)
    auc_rand_singleton = roc_auc_score(labels[singleton_mask_full], score_rand[singleton_mask_full])
    auc_rand_overall = roc_auc_score(labels, score_rand)
    print(f"\n[random negatives] textsim_shared AUC: singleton={auc_rand_singleton:.4f}  overall={auc_rand_overall:.4f}")

    # ---- degree-matched negatives: IDENTICAL matching logic to
    #      experiment1_degree_matched_robustness.py (duplicated for a self-contained
    #      audit script), so this baseline is scored on the exact same pairs ----
    true_edges_by_manu = defaultdict(set)
    m2i = {m: i for i, m in enumerate(manus_sorted)}
    p2i = {p: i for i, p in enumerate(products_sorted)}
    for m, p in edge_set_str:
        true_edges_by_manu[m2i[m]].add(p2i[p])

    train_pos_mask = train_data[rel].edge_label == 1
    train_pos_idx = train_data[rel].edge_label_index[:, train_pos_mask]
    n_prod = data['product'].x.size(0)
    train_degree = np.zeros(n_prod, dtype=np.int64)
    for p in train_pos_idx[1].numpy():
        train_degree[p] += 1
    degree_to_products = defaultdict(list)
    for p in range(n_prod):
        degree_to_products[int(train_degree[p])].append(p)

    pos_mask = labels == 1
    pos_m, pos_p = idx[0][pos_mask], idx[1][pos_mask]
    singleton_pos_mask = full_count[pos_p] <= 1

    def build_matched_negatives(seed):
        rng = random.Random(seed)
        shuffled, cursors = {}, {}
        for d, lst in degree_to_products.items():
            s = list(lst); rng.shuffle(s)
            shuffled[d] = s; cursors[d] = 0
        order = list(range(len(pos_m))); rng.shuffle(order)
        neg_p = np.full(len(pos_m), -1, dtype=np.int64)
        used_pairs = set()
        n_exact, n_fallback, n_unmatched = 0, 0, 0
        gap = np.zeros(len(pos_m), dtype=np.int64)
        for i in order:
            m, p, d = int(pos_m[i]), int(pos_p[i]), int(train_degree[pos_p[i]])
            chosen = None
            for widen in range(0, 6):
                dds = [d] if widen == 0 else [d - widen, d + widen]
                for dd in dds:
                    if dd < 0 or dd not in shuffled:
                        continue
                    lst = shuffled[dd]; n = len(lst)
                    if n == 0:
                        continue
                    start = cursors[dd]
                    for step in range(n):
                        c = lst[(start + step) % n]
                        if c == p or c in true_edges_by_manu[m] or (m, c) in used_pairs:
                            continue
                        chosen = c; cursors[dd] = (start + step + 1) % n
                        break
                    if chosen is not None:
                        break
                if chosen is not None:
                    break
            if chosen is None:
                n_unmatched += 1
                continue
            neg_p[i] = chosen
            gap[i] = abs(train_degree[chosen] - d)
            used_pairs.add((m, chosen))
            if gap[i] == 0:
                n_exact += 1
            else:
                n_fallback += 1
        return neg_p, gap, n_exact, n_fallback, n_unmatched

    SEEDS = [1, 2, 3]
    print("\n[degree-matched negatives] textsim_shared AUC (matching logic identical to "
          "experiment1_degree_matched_robustness.py):")
    for seed in SEEDS:
        neg_p, gap, n_exact, n_fallback, n_unmatched = build_matched_negatives(seed)
        valid = neg_p >= 0
        m_pos, p_pos = pos_m[valid], pos_p[valid]
        m_neg, p_neg = pos_m[valid], neg_p[valid]
        m_all = np.concatenate([m_pos, m_neg])
        p_all = np.concatenate([p_pos, p_neg])
        y_all = np.concatenate([np.ones(valid.sum()), np.zeros(valid.sum())])
        sing_mask = np.concatenate([singleton_pos_mask[valid], singleton_pos_mask[valid]])
        s_textsim = score_textsim_shared(m_all, p_all)
        auc_overall = roc_auc_score(y_all, s_textsim)
        auc_singleton = roc_auc_score(y_all[sing_mask], s_textsim[sing_mask])
        exact_rate = n_exact / len(pos_m)
        print(f"  seed={seed}  exact_match_rate={exact_rate:.4f}  unmatched={n_unmatched}  "
              f"textsim_shared_AUC overall={auc_overall:.4f}  singleton={auc_singleton:.4f}")

    with open("experiment1_textsim_shared_space_results.md", "w", encoding="utf-8") as f:
        f.write("# Corrected (shared-space) text-similarity baseline, Task A\n\n")
        f.write("Supersedes the `textsim` column in experiment1_baseline_results.md / "
                "experiment1_degree_matched_results.md, which computed cosine similarity "
                "between two INDEPENDENTLY-fit SVD spaces (manufacturer vs. product) -- "
                "not a valid similarity measure. This version fits ONE shared TF-IDF "
                "vocabulary over manufacturer text (Name + NAICS_Description_1) and product "
                "strings together, no SVD.\n\n")
        f.write(f"Shared TF-IDF vocab size: {len(shared_vec.vocabulary_)}\n\n")
        f.write("| Protocol | Bucket | AUC (textsim_shared, corrected) |\n")
        f.write("|---|---|---|\n")
        f.write(f"| random negatives | singleton | {auc_rand_singleton:.4f} |\n")
        f.write(f"| random negatives | overall | {auc_rand_overall:.4f} |\n")
        for seed in SEEDS:
            neg_p, gap, n_exact, n_fallback, n_unmatched = build_matched_negatives(seed)
            valid = neg_p >= 0
            m_pos, p_pos = pos_m[valid], pos_p[valid]
            m_neg, p_neg = pos_m[valid], neg_p[valid]
            m_all = np.concatenate([m_pos, m_neg]); p_all = np.concatenate([p_pos, p_neg])
            y_all = np.concatenate([np.ones(valid.sum()), np.zeros(valid.sum())])
            sing_mask = np.concatenate([singleton_pos_mask[valid], singleton_pos_mask[valid]])
            s_textsim = score_textsim_shared(m_all, p_all)
            f.write(f"| degree-matched (seed={seed}) | overall | {roc_auc_score(y_all, s_textsim):.4f} |\n")
            f.write(f"| degree-matched (seed={seed}) | singleton | {roc_auc_score(y_all[sing_mask], s_textsim[sing_mask]):.4f} |\n")

    print("\nwrote experiment1_textsim_shared_space_results.md")


if __name__ == "__main__":
    main()
