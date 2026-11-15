#!/usr/bin/env python3
"""
Experiment 1 addendum (user-directed, priority 1 of the "what does this
benchmark actually discriminate" round): a competitive NO-GRAPH baseline for
Task A. Degree and shared-space-textsim are both fixed, untrained scores --
they cannot show whether SAGE's advantage is really about graph structure,
or just about having ANY trained model. This closes that gap: a pairwise
MLP over the SAME raw manufacturer/product features SAGE starts from
(before any message passing), concatenated and scored by a small trained
head. No edge_index, no HeteroConv, nothing graph-structural at all.

Same frozen split, same train/val/test edges, same FocalLoss, same
early-stopping-on-val-AUC discipline as experiment1_sage_grouped.py, EXCEPT:
the LR is selected from a validation-only grid (SAGE's LR was reused from a
prior search, not re-searched here; this is a new model class and deserves
its own selection, not a borrowed value).

Evaluated under BOTH negative-sampling protocols already established for
this task:
  1. Random negatives -- the frozen split's own test edges.
  2. Degree-matched negatives -- same matching procedure as
     experiment1_degree_matched_robustness.py (3 resampling seeds), applied
     to this model's own scores instead of the archived SAGE checkpoint.
"""
import csv
import random
import importlib.util
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

spec = importlib.util.spec_from_file_location("lp_abcd", "LP-ABCD.py")
lp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp)

GENERIC_THRESHOLD = 20
SEED = 42
HIDDEN, DROPOUT = 128, 0.3
LR_GRID = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
MAX_EPOCHS, PATIENCE = 1000, 20
DEGREE_MATCH_SEEDS = [1, 2, 3]


def canon(w):
    w = str(w).strip().lower()
    for p in ("https://", "http://"):
        if w.startswith(p):
            w = w[len(p):]
    return w.rstrip("/")


def bucket_for_count(c):
    if c <= 1:
        return "1_singleton"
    if c <= 5:
        return "2-5"
    if c < GENERIC_THRESHOLD:
        return "6-19"
    return f"high_frequency_{GENERIC_THRESHOLD}+"


class PairwiseMLP(nn.Module):
    """No graph access whatsoever: scores a (manufacturer, product) pair from
    their own raw features alone, concatenated. Never sees edge_index."""
    def __init__(self, in_manuf, in_prod, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_manuf + in_prod, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, manu_x, prod_x, m_idx, p_idx):
        pair = torch.cat([manu_x[m_idx], prod_x[p_idx]], dim=-1)
        return self.net(pair).squeeze(-1)


def train_one(manu_x, prod_x, train_data, val_data, lr, seed, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rel = ("manufacturer", "has_product", "product")
    model = PairwiseMLP(manu_x.shape[1], prod_x.shape[1], HIDDEN, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = lp.FocalLoss(alpha=0.25, gamma=2.0)

    train_idx = train_data[rel].edge_label_index
    train_labels = train_data[rel].edge_label.float()
    val_idx = val_data[rel].edge_label_index
    val_labels = val_data[rel].edge_label.float()

    best_auc, best_state, no_imp, stop_epoch = -float("inf"), None, 0, MAX_EPOCHS
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(manu_x, prod_x, train_idx[0], train_idx[1])
        loss = criterion(logits, train_labels)
        loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(manu_x, prod_x, val_idx[0], val_idx[1])
            val_auc = roc_auc_score(val_labels.cpu(), torch.sigmoid(val_logits).cpu())
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                stop_epoch = epoch
                break
    model.load_state_dict(best_state)
    return model, best_auc, stop_epoch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = torch.load("frozen_split_GraphA_seed42.pt", map_location=device, weights_only=False)
    data = torch.load("hetero_graphs/Graph_A.pt", map_location=device, weights_only=False)
    train_data, val_data, test_data = split["train"], split["val"], split["test"]
    rel = ("manufacturer", "has_product", "product")
    manu_x, prod_x = data["manufacturer"].x.to(device), data["product"].x.to(device)

    # ---- LR selection on validation AUC only --------------------------------
    print("=== LR grid, selected on validation AUC ===")
    best = (None, -1, None, None)
    for lr in LR_GRID:
        model, val_auc, stop_ep = train_one(manu_x, prod_x, train_data, val_data, lr, SEED, device)
        print(f"  lr={lr:.1e}  val_AUC={val_auc:.4f}  stop_epoch={stop_ep}")
        if val_auc > best[1]:
            best = (lr, val_auc, model, stop_ep)
    lr, val_auc, model, stop_epoch = best
    print(f"selected lr={lr:.1e}  val_AUC={val_auc:.4f}  stop_epoch={stop_epoch}")
    model.eval()

    # ---- full product-sharing-count buckets (same definition as elsewhere) --
    prod_to_manus = defaultdict(set)
    edge_set = set()
    with open("Cleaned_Master_CSV_v3.csv", newline='', encoding='utf-8-sig', errors='replace') as f:
        reader = csv.DictReader(f)
        cols = [f"Product_{i}" for i in range(1, 151)]
        for r in reader:
            m = canon(r.get("Web", ""))
            for c in cols:
                p = r.get(c)
                if p and p.strip() and p.strip().lower() != "yes":
                    prod_to_manus[p.strip()].add(m)
                    edge_set.add((m, p.strip()))
    products_sorted = sorted({p for _, p in edge_set})
    full_count = np.array([len(prod_to_manus[p]) for p in products_sorted])

    def score(m_arr, p_arr):
        with torch.no_grad():
            m_t = torch.as_tensor(m_arr, device=device)
            p_t = torch.as_tensor(p_arr, device=device)
            logits = model(manu_x, prod_x, m_t, p_t)
            return torch.sigmoid(logits).cpu().numpy()

    # ---- protocol 1: random negatives, the frozen split's own test edges ----
    idx = test_data[rel].edge_label_index.cpu().numpy()
    labels = test_data[rel].edge_label.cpu().numpy()
    probs = score(idx[0], idx[1])
    overall_auc_random = roc_auc_score(labels, probs)
    overall_prauc_random = average_precision_score(labels, probs)
    singleton_mask = full_count[idx[1]] <= 1
    singleton_auc_random = roc_auc_score(labels[singleton_mask], probs[singleton_mask])

    print(f"\n=== Random negatives (frozen test split) ===")
    print(f"  overall AUC={overall_auc_random:.4f}  PR-AUC={overall_prauc_random:.4f}")
    print(f"  singleton AUC={singleton_auc_random:.4f}")

    # ---- protocol 2: degree-matched negatives, 3 resampling seeds -----------
    edge_set_str = edge_set
    manus_sorted = sorted({m for m, _ in edge_set_str})
    m2i = {m: i for i, m in enumerate(manus_sorted)}
    p2i = {p: i for i, p in enumerate(products_sorted)}
    true_edges_by_manu = defaultdict(set)
    for m, p in edge_set_str:
        true_edges_by_manu[m2i[m]].add(p2i[p])

    train_pos_mask = train_data[rel].edge_label == 1
    train_pos_idx = train_data[rel].edge_label_index[:, train_pos_mask]
    n_prod = data["product"].x.size(0)
    train_degree = np.zeros(n_prod, dtype=np.int64)
    for p in train_pos_idx[1].cpu().numpy():
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
                continue
            neg_p[i] = chosen
            used_pairs.add((m, chosen))
        return neg_p

    overall_aucs, singleton_aucs = [], []
    for seed in DEGREE_MATCH_SEEDS:
        neg_p = build_matched_negatives(seed)
        valid = neg_p >= 0
        m_pos, p_pos = pos_m[valid], pos_p[valid]
        m_neg, p_neg = pos_m[valid], neg_p[valid]
        m_all = np.concatenate([m_pos, m_neg])
        p_all = np.concatenate([p_pos, p_neg])
        y_all = np.concatenate([np.ones(valid.sum()), np.zeros(valid.sum())])
        sing_mask = np.concatenate([singleton_pos_mask[valid], singleton_pos_mask[valid]])
        probs_dm = score(m_all, p_all)
        overall_aucs.append(roc_auc_score(y_all, probs_dm))
        singleton_aucs.append(roc_auc_score(y_all[sing_mask], probs_dm[sing_mask]))

    overall_auc_dm = float(np.mean(overall_aucs))
    singleton_auc_dm = float(np.mean(singleton_aucs))
    print(f"\n=== Degree-matched negatives (mean of {len(DEGREE_MATCH_SEEDS)} resampling seeds) ===")
    print(f"  overall AUC={overall_auc_dm:.4f} (per-seed: {[f'{x:.4f}' for x in overall_aucs]})")
    print(f"  singleton AUC={singleton_auc_dm:.4f} (per-seed: {[f'{x:.4f}' for x in singleton_aucs]})")

    # ---- population-matched comparison, matching Table 2's EXISTING SAGE=0.6012 row:
    # same 11,214 positives retained after seed=1 degree-matching exclusions, scored
    # under RANDOM negatives (all 11,259 original negatives kept). Re-derives the SAME
    # 45-positive exclusion set via the identical matching code at the identical seed,
    # not a different/approximate restriction. -------------------------------------
    neg_p_seed1 = build_matched_negatives(1)
    valid_seed1 = neg_p_seed1 >= 0
    n_matched = int(valid_seed1.sum())
    print(f"\n=== Population-matched re-score (seed=1 matching identity, n_pos={n_matched}) ===")
    matched_pos_set = set(zip(pos_m[valid_seed1].tolist(), pos_p[valid_seed1].tolist()))
    keep_mask = np.array([
        (m, p) in matched_pos_set if lbl == 1 else True
        for m, p, lbl in zip(idx[0], idx[1], labels)
    ])
    probs_matched = probs[keep_mask]
    labels_matched = labels[keep_mask]
    overall_auc_matched = roc_auc_score(labels_matched, probs_matched)
    sing_mask_matched = full_count[idx[1][keep_mask]] <= 1
    singleton_auc_matched = roc_auc_score(labels_matched[sing_mask_matched], probs_matched[sing_mask_matched])
    print(f"  n_pos={int((labels_matched==1).sum())}  n_neg={int((labels_matched==0).sum())}")
    print(f"  overall AUC (population-matched)={overall_auc_matched:.4f}")
    print(f"  singleton AUC (population-matched)={singleton_auc_matched:.4f}")

    # same restriction applied to degree and the CORRECTED shared-space textsim
    # baseline (fit ONE shared TfidfVectorizer on manufacturer + product text
    # together -- NOT data['manufacturer'].x / data['product'].x, which are
    # separately-fit SVD spaces with no shared basis; that invalid comparison
    # was already found and fixed in experiment1_textsim_shared_space.py and
    # must not be reintroduced here) -------------------------------------------
    from sklearn.feature_extraction.text import TfidfVectorizer
    import scipy.sparse as sp
    rows_by_manu = {}
    with open("Cleaned_Master_CSV_v3.csv", newline='', encoding='utf-8-sig', errors='replace') as f:
        import csv as _csv
        reader = _csv.DictReader(f)
        for r in reader:
            rows_by_manu[canon(r.get("Web", ""))] = r
    manus_sorted_full = sorted({m for m, _ in edge_set_str})
    manu_text = [f"{rows_by_manu[m].get('Name','') or ''} {rows_by_manu[m].get('NAICS_Description_1','') or ''}"
                 for m in manus_sorted_full]
    prod_text = list(products_sorted)
    shared_vec = TfidfVectorizer(stop_words='english')
    shared_vec.fit(manu_text + prod_text)
    manu_tfidf = shared_vec.transform(manu_text)
    prod_tfidf = shared_vec.transform(prod_text)

    def l2norm_sparse(m):
        norms = np.sqrt(m.multiply(m).sum(axis=1)).A1
        norms[norms == 0] = 1.0
        return sp.diags(1.0 / norms) @ m

    manu_n = l2norm_sparse(manu_tfidf).tocsr()
    prod_n = l2norm_sparse(prod_tfidf).tocsr()

    def score_textsim_shared(m_arr, p_arr):
        a, b = manu_n[m_arr], prod_n[p_arr]
        return np.asarray(a.multiply(b).sum(axis=1)).ravel()

    deg_scores_all = train_degree[idx[1]].astype(np.float64)
    txt_scores_all = score_textsim_shared(idx[0], idx[1])
    deg_auc_matched = roc_auc_score(labels_matched, deg_scores_all[keep_mask])
    txt_auc_matched = roc_auc_score(labels_matched, txt_scores_all[keep_mask])
    print(f"  degree AUC (population-matched)={deg_auc_matched:.4f}")
    print(f"  shared-space textsim AUC (population-matched, corrected method)={txt_auc_matched:.4f}")

    # SAGE on the exact same matched population, for a true like-for-like singleton
    # comparison (not asserted -- actually recomputed from the archived checkpoint)
    sage_ckpt = torch.load("experiment1_sage_checkpoint.pt", map_location=device, weights_only=False)
    sage_model = lp.HeteroSAGE(sage_ckpt["in_manuf"], sage_ckpt["in_prod"], sage_ckpt["hidden"],
                                sage_ckpt["rels"], sage_ckpt["dropout"]).to(device)
    sage_model.load_state_dict(sage_ckpt["state_dict"])
    sage_model.eval()
    with torch.no_grad():
        h = sage_model(test_data.x_dict, test_data.edge_index_dict)
        sage_logits_all = (h["manufacturer"][torch.as_tensor(idx[0], device=device)]
                            * h["product"][torch.as_tensor(idx[1], device=device)]).sum(dim=-1)
        sage_probs_all = torch.sigmoid(sage_logits_all).cpu().numpy()
    sage_auc_matched_overall = roc_auc_score(labels_matched, sage_probs_all[keep_mask])
    sage_auc_matched_singleton = roc_auc_score(labels_matched[sing_mask_matched],
                                                sage_probs_all[keep_mask][sing_mask_matched])
    print(f"  SAGE overall AUC (population-matched, recomputed)={sage_auc_matched_overall:.4f}")
    print(f"  SAGE singleton AUC (population-matched, recomputed)={sage_auc_matched_singleton:.4f}")

    with open("experiment1_pairwise_mlp_results.md", "w", encoding="utf-8") as f:
        f.write("# Task A: pairwise no-graph MLP baseline\n\n")
        f.write(
            "Scores each (manufacturer, product) test pair from a concatenation of their "
            "own raw 64-dim features ALONE -- no edge_index, no message passing, nothing "
            "graph-structural. Same frozen split, same FocalLoss, same early-stopping-on-"
            f"val-AUC discipline as the SAGE baseline. LR selected from a validation-only "
            f"grid {LR_GRID} (selected lr={lr:.1e}, val_AUC={val_auc:.4f}, stop_epoch={stop_epoch}) "
            "-- a fresh selection for this model class, not SAGE's reused value.\n\n"
        )
        f.write("| Bucket | Protocol | Pairwise MLP (no graph) AUC |\n|---|---|---|\n")
        f.write(f"| Overall | Random (full test set, n_pos=11259) | {overall_auc_random:.4f} |\n")
        f.write(f"| Overall | Degree-matched (mean, 3 seeds) | {overall_auc_dm:.4f} |\n")
        f.write(f"| Singleton | Random (full test set) | {singleton_auc_random:.4f} |\n")
        f.write(f"| Singleton | Degree-matched (mean, 3 seeds) | {singleton_auc_dm:.4f} |\n")
        f.write(f"\nOverall PR-AUC (random negatives, full test set): {overall_prauc_random:.4f}\n")
        f.write(f"\nPer-seed degree-matched AUCs -- overall: {overall_aucs}; singleton: {singleton_aucs}\n")
        f.write(f"\n## Population-matched re-score for Table 2 (same {n_matched}-positive population as "
                f"the archived SAGE=0.6012 row, seed=1 degree-matching identity exclusion, random negatives)\n\n")
        f.write("| Bucket | Protocol | Pairwise MLP | SAGE (archived) | Degree | Shared-space text (corrected method) |\n")
        f.write("|---|---|---|---|---|---|\n")
        f.write(f"| Overall | Random (population-matched) | {overall_auc_matched:.4f} | "
                f"{sage_auc_matched_overall:.4f} (recomputed; archived table value 0.6012) | "
                f"{deg_auc_matched:.4f} | {txt_auc_matched:.4f} |\n")
        f.write(f"| Singleton | Random (population-matched) | {singleton_auc_matched:.4f} | "
                f"{sage_auc_matched_singleton:.4f} (recomputed; archived full-population value 0.7346) | -- | -- |\n")
        f.write(f"\n**Single run (seed={SEED} only) for the pairwise MLP -- not averaged over multiple "
                f"training seeds. Report as a single-run result, not as statistically robust across seeds.**\n")

    print("\nwrote experiment1_pairwise_mlp_results.md")


if __name__ == "__main__":
    main()
