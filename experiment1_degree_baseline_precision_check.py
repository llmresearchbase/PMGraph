#!/usr/bin/env python3
"""
Precision check on the degree-matched `degree` baseline's AUC, which the
paper's Section 5.1 described as "0.500 (by construction)". Reviewer flagged
that match rate is 97.19% EXACT + a widened-fallback remainder (not 100%),
so "exactly 0.5" needs checking at full precision, not assumed from a
4-decimal printout. Reuses the identical train-degree definition and
matching logic already used for SAGE/degree/textsim (no retraining; this is
a pure re-scoring of the degree baseline at higher precision).
"""
import csv
import random
import numpy as np
import torch
from collections import defaultdict
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

    edge_set_str = set()
    with open("Cleaned_Master_CSV_v3.csv", newline='', encoding='utf-8-sig', errors='replace') as f:
        reader = csv.DictReader(f)
        cols = [f"Product_{i}" for i in range(1, 151)]
        for r in reader:
            m = canon(r.get("Web", ""))
            for c in cols:
                p = r.get(c)
                if p and p.strip() and p.strip().lower() != "yes":
                    edge_set_str.add((m, p.strip()))
    manus_sorted = sorted({m for m, _ in edge_set_str})
    products_sorted = sorted({p for _, p in edge_set_str})
    m2i = {m: i for i, m in enumerate(manus_sorted)}
    p2i = {p: i for i, p in enumerate(products_sorted)}
    true_edges_by_manu = defaultdict(set)
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

    idx = test_data[rel].edge_label_index.numpy()
    labels = test_data[rel].edge_label.numpy()
    pos_mask = labels == 1
    pos_m, pos_p = idx[0][pos_mask], idx[1][pos_mask]

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

    print("Matching uses TRAIN-degree (not full-graph degree); the degree baseline's "
          "score is ALSO train_degree -- same definition on both sides.\n")
    for seed in [1, 2, 3]:
        neg_p, gap, n_exact, n_fallback, n_unmatched = build_matched_negatives(seed)
        valid = neg_p >= 0
        p_pos, p_neg = pos_p[valid], neg_p[valid]
        y = np.concatenate([np.ones(valid.sum()), np.zeros(valid.sum())])
        s = np.concatenate([train_degree[p_pos], train_degree[p_neg]]).astype(np.float64)
        auc = roc_auc_score(y, s)
        n_total = len(pos_m)
        fallback_gaps = gap[gap > 0]
        print(f"seed={seed}: exact={n_exact} ({n_exact/n_total:.4%})  fallback={n_fallback} "
              f"({n_fallback/n_total:.4%}, mean realized degree-gap={fallback_gaps.mean():.3f})  "
              f"unmatched={n_unmatched} ({n_unmatched/n_total:.4%})")
        print(f"  degree_AUC (overall) = {auc:.6f}  (deviation from 0.5: {auc-0.5:+.6f})")


if __name__ == "__main__":
    main()
