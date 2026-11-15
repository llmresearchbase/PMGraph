#!/usr/bin/env python3
"""
Robustness check: does SAGE's singleton-bucket advantage (AUC=0.7346 under
uniform-random negatives) survive when negatives are instead sampled to
match each positive's TRAIN-degree? This removes the train-degree==0-vs-1
confound diagnosed earlier (positives' train-degree is ~always 0 in the
singleton bucket; uniform-random negatives land on train-degree==1 products
84.6% of the time there, purely because degree-0 products are a minority of
the full product space).

No retraining: loads the already-trained, frozen SAGE checkpoint
(experiment1_sage_checkpoint.pt) and only reconstructs the NEGATIVE half of
the test set. Positives are untouched (same manufacturer, same product,
same label=1 rows as the original frozen test split).

Matching rule (declared, not tuned):
  - For each positive (m, p) with train_degree(p) == d, sample one negative
    product p' with train_degree(p') == d (exact match), p' != p, and
    (m, p') not a recorded true edge ANYWHERE in the full graph.
  - "Not a recorded true edge" is the only exclusion applied -- everything
    else is an UNOBSERVED-relation negative, not a confirmed no-capability
    negative (the manufacturer may well be capable; we simply have no
    record of it).
  - If no exact-degree candidate remains (after excluding true edges and
    already-used-up candidates), widen symmetrically to d+-1, d+-2, ... up
    to d+-5; if still nothing, mark unmatched and report the count instead
    of silently substituting a non-matching negative.
Match rate, unmatched count, and realized degree gap are reported.
"""
import csv
import random
import importlib.util
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score

spec = importlib.util.spec_from_file_location("lp_abcd", "LP-ABCD.py")
lp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp)


def canon(w):
    w = str(w).strip().lower()
    for p in ("https://", "http://"):
        if w.startswith(p):
            w = w[len(p):]
    return w.rstrip("/")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = torch.load("frozen_split_GraphA_seed42.pt", map_location=device, weights_only=False)
    data = torch.load("hetero_graphs/Graph_A.pt", map_location=device, weights_only=False)
    train_data, test_data = split["train"], split["test"]
    rel = ("manufacturer", "has_product", "product")

    # ---- rebuild the FULL true-edge set (manufacturer_idx, product_idx), for
    # the "exclude already-recorded positive relations" rule -----------------
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
    assert len(manus_sorted) == data['manufacturer'].x.size(0)
    assert len(products_sorted) == data['product'].x.size(0)
    true_edges_by_manu = defaultdict(set)
    for m, p in edge_set_str:
        true_edges_by_manu[m2i[m]].add(p2i[p])

    # ---- train-only degree per product -------------------------------------
    train_pos_mask = train_data[rel].edge_label == 1
    train_pos_idx = train_data[rel].edge_label_index[:, train_pos_mask]
    n_prod = data['product'].x.size(0)
    train_degree = np.zeros(n_prod, dtype=np.int64)
    for p in train_pos_idx[1].cpu().numpy():
        train_degree[p] += 1
    degree_to_products = defaultdict(list)
    for p in range(n_prod):
        degree_to_products[int(train_degree[p])].append(p)

    # ---- original test positives (kept fixed throughout) --------------------
    idx = test_data[rel].edge_label_index.cpu().numpy()
    labels = test_data[rel].edge_label.cpu().numpy()
    pos_mask = labels == 1
    pos_m, pos_p = idx[0][pos_mask], idx[1][pos_mask]
    print(f"total positive test edges: {len(pos_m)}")

    prod_to_manus_full = defaultdict(set)
    for m, p in edge_set_str:
        prod_to_manus_full[p].add(m)
    full_count = np.array([len(prod_to_manus_full[p]) for p in products_sorted])
    singleton_pos_mask = full_count[pos_p] <= 1
    print(f"of which singleton-bucket positives: {singleton_pos_mask.sum()}")

    def build_matched_negatives(seed):
        rng = random.Random(seed)
        # pre-shuffle each degree bucket once; scan with a rotating cursor
        # instead of reshuffling per-query (degree=0 bucket alone is ~thousands
        # of candidates and singleton positives number in the tens of thousands)
        shuffled = {}
        cursors = {}
        for d, lst in degree_to_products.items():
            s = list(lst)
            rng.shuffle(s)
            shuffled[d] = s
            cursors[d] = 0

        # process positives in random order so no single manufacturer/degree
        # systematically exhausts the front of a shuffled bucket first
        order = list(range(len(pos_m)))
        rng.shuffle(order)

        neg_p = np.full(len(pos_m), -1, dtype=np.int64)
        realized_gap = np.zeros(len(pos_m), dtype=np.int64)
        n_exact, n_fallback, n_unmatched = 0, 0, 0
        used_pairs = set()

        for i in order:
            m, p, d = int(pos_m[i]), int(pos_p[i]), int(train_degree[pos_p[i]])
            chosen = None
            for widen in range(0, 6):
                dds = [d] if widen == 0 else [d - widen, d + widen]
                for dd in dds:
                    if dd < 0 or dd not in shuffled:
                        continue
                    lst = shuffled[dd]
                    n = len(lst)
                    if n == 0:
                        continue
                    start = cursors[dd]
                    for step in range(n):
                        c = lst[(start + step) % n]
                        if c == p or c in true_edges_by_manu[m] or (m, c) in used_pairs:
                            continue
                        chosen = c
                        cursors[dd] = (start + step + 1) % n
                        break
                    if chosen is not None:
                        break
                if chosen is not None:
                    break
            if chosen is None:
                n_unmatched += 1
                continue
            neg_p[i] = chosen
            realized_gap[i] = abs(train_degree[chosen] - d)
            used_pairs.add((m, chosen))
            if realized_gap[i] == 0:
                n_exact += 1
            else:
                n_fallback += 1
        return neg_p, realized_gap, n_exact, n_fallback, n_unmatched

    # ---- load the FROZEN, already-trained SAGE checkpoint (no retraining) ---
    ckpt = torch.load("experiment1_sage_checkpoint.pt", map_location=device, weights_only=False)
    model = lp.HeteroSAGE(ckpt["in_manuf"], ckpt["in_prod"], ckpt["hidden"], ckpt["rels"], ckpt["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    def score_sage(m_arr, p_arr):
        with torch.no_grad():
            h = model(test_data.x_dict, test_data.edge_index_dict)
            m_t = torch.as_tensor(m_arr, device=device)
            p_t = torch.as_tensor(p_arr, device=device)
            logits = (h['manufacturer'][m_t] * h['product'][p_t]).sum(dim=-1)
            return torch.sigmoid(logits).cpu().numpy()

    def score_degree(p_arr):
        return train_degree[p_arr].astype(np.float64)

    manu_x = data['manufacturer'].x.cpu().numpy()
    prod_x = data['product'].x.cpu().numpy()
    manu_x_n = manu_x / (np.linalg.norm(manu_x, axis=1, keepdims=True) + 1e-9)
    prod_x_n = prod_x / (np.linalg.norm(prod_x, axis=1, keepdims=True) + 1e-9)

    def score_textsim(m_arr, p_arr):
        return np.sum(manu_x_n[m_arr] * prod_x_n[p_arr], axis=1)

    SEEDS = [1, 2, 3]
    report_lines = []
    report_lines.append("# Experiment 1: degree-matched negative-sampling robustness check\n")
    report_lines.append("Same trained SAGE checkpoint, NOT retrained. Only the negative half of the "
                         "test set is reconstructed to match each positive's train-degree.\n")

    # original (random-negative) full test arrays, for a fair same-positive-set comparison
    orig_m_all, orig_p_all = idx[0], idx[1]
    orig_y_all = labels
    orig_singleton_edge_mask = full_count[orig_p_all] <= 1
    s_sage_orig_full = score_sage(orig_m_all, orig_p_all)
    s_deg_orig_full = score_degree(orig_p_all)
    s_txt_orig_full = score_textsim(orig_m_all, orig_p_all)

    for seed in SEEDS:
        neg_p, gap, n_exact, n_fallback, n_unmatched = build_matched_negatives(seed)
        valid = neg_p >= 0
        m_pos, p_pos = pos_m[valid], pos_p[valid]
        m_neg, p_neg = pos_m[valid], neg_p[valid]  # same manufacturer as its paired positive

        m_all = np.concatenate([m_pos, m_neg])
        p_all = np.concatenate([p_pos, p_neg])
        y_all = np.concatenate([np.ones(valid.sum()), np.zeros(valid.sum())])
        # for negatives, "singleton-bucket membership" is inherited from the paired positive (matching rule)
        sing_mask_overall = np.concatenate([singleton_pos_mask[valid], singleton_pos_mask[valid]])

        s_sage = score_sage(m_all, p_all)
        s_deg = score_degree(p_all)
        s_txt = score_textsim(m_all, p_all)

        exact_rate = n_exact / len(pos_m)
        total_match_rate = (n_exact + n_fallback) / len(pos_m)

        # fair comparison: original random-negative evaluation, restricted to the SAME
        # retained positive identities (exclude the same unmatched positives), negatives
        # unchanged (all original negatives, not reconstructed)
        valid_pos_set = set(zip(pos_m[valid].tolist(), pos_p[valid].tolist()))
        orig_pos_mask_full = orig_y_all == 1
        orig_keep_pos = np.array([
            (m, p) in valid_pos_set if lbl == 1 else True
            for m, p, lbl in zip(orig_m_all, orig_p_all, orig_y_all)
        ])

        def report(mask_new, mask_orig, label):
            yl = y_all[mask_new]
            n_pos, n_neg = int((yl == 1).sum()), int((yl == 0).sum())
            auc_sage = roc_auc_score(yl, s_sage[mask_new])
            pr_sage = average_precision_score(yl, s_sage[mask_new])
            auc_deg = roc_auc_score(yl, s_deg[mask_new])
            auc_txt = roc_auc_score(yl, s_txt[mask_new])

            yl_o = orig_y_all[mask_orig]
            n_pos_o, n_neg_o = int((yl_o == 1).sum()), int((yl_o == 0).sum())
            auc_sage_o = roc_auc_score(yl_o, s_sage_orig_full[mask_orig])
            pr_sage_o = average_precision_score(yl_o, s_sage_orig_full[mask_orig])
            auc_deg_o = roc_auc_score(yl_o, s_deg_orig_full[mask_orig])
            auc_txt_o = roc_auc_score(yl_o, s_txt_orig_full[mask_orig])

            line1 = (f"seed={seed} {label:<10} [degree-matched]  n_pos={n_pos} n_neg={n_neg} "
                     f"SAGE_AUC={auc_sage:.4f} SAGE_PRAUC={pr_sage:.4f} "
                     f"degree_AUC={auc_deg:.4f} textsim_AUC={auc_txt:.4f}")
            line2 = (f"seed={seed} {label:<10} [random-neg, same {n_pos}+{len(pos_m)-valid.sum()}-excluded "
                     f"positives]  n_pos={n_pos_o} n_neg={n_neg_o} "
                     f"SAGE_AUC={auc_sage_o:.4f} SAGE_PRAUC={pr_sage_o:.4f} "
                     f"degree_AUC={auc_deg_o:.4f} textsim_AUC={auc_txt_o:.4f}")
            print(line1); print(line2)
            report_lines.append(line1 + "  \n")
            report_lines.append(line2 + "  \n")

        print(f"\n--- seed={seed} --- exact={n_exact} ({exact_rate:.4f}) fallback={n_fallback} "
              f"unmatched={n_unmatched}  total_match_rate(incl.fallback)={total_match_rate:.4f}  "
              f"mean realized gap on fallback cases={gap[gap>0].mean() if (gap>0).any() else 0:.3f}")
        report_lines.append(f"\nseed={seed}: exact={n_exact} ({exact_rate:.4f})  fallback={n_fallback}  "
                             f"unmatched={n_unmatched}  total_match_rate(incl.fallback)={total_match_rate:.4f}\n\n")

        mask_all_orig = orig_keep_pos | (orig_y_all == 0)
        mask_sing_new = sing_mask_overall
        mask_sing_orig = mask_all_orig & (orig_singleton_edge_mask)
        report(np.ones(len(y_all), dtype=bool), mask_all_orig, "overall")
        report(mask_sing_new, mask_sing_orig, "singleton")

    with open("experiment1_degree_matched_results.md", "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    print("\nwrote experiment1_degree_matched_results.md")


if __name__ == "__main__":
    main()
