#!/usr/bin/env python3
"""
Train one SAGE model on the corrected Graph_A using the FROZEN split
(frozen_split_GraphA_seed42.pt -- same literal split object used by the
baselines in experiment1_grouped_baselines.py), then report overall AND
per-bucket (product-sharing-count group) AUC/PR-AUC with pos/neg counts,
for direct comparison against the product-degree and text-similarity
baselines. One model only, per the agreed scope -- not a full re-sweep.
"""
import csv
import random
import importlib.util
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score

spec = importlib.util.spec_from_file_location("lp_abcd", "LP-ABCD.py")
lp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp)

GENERIC_THRESHOLD = 20
SEED = 42
HIDDEN, DROPOUT, LOSS_TYPE = 128, 0.3, "focal"
LR = 5e-3  # reuse the previously-selected sage LR for Graph_A (not re-searched)
MAX_EPOCHS, PATIENCE = 1000, 20


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


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = torch.load("frozen_split_GraphA_seed42.pt", map_location=device, weights_only=False)
    data = torch.load("hetero_graphs/Graph_A.pt", map_location=device, weights_only=False)
    train_data, val_data, test_data = split["train"], split["val"], split["test"]

    rel = ("manufacturer", "has_product", "product")
    rels = {"fwd": rel, "rev": ("product", "rev_has_product", "manufacturer")}
    in_manuf, in_prod = data["manufacturer"].x.size(1), data["product"].x.size(1)

    train_labels = train_data[rel].edge_label.float()
    pos, neg = train_labels.sum().item(), train_labels.numel() - train_labels.sum().item()
    criterion = lp.FocalLoss(alpha=0.25, gamma=2.0) if LOSS_TYPE == "focal" \
        else nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))

    model = lp.HeteroSAGE(in_manuf, in_prod, HIDDEN, rels, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_auc, best_state, no_imp, stop_epoch = -float("inf"), None, 0, MAX_EPOCHS
    for epoch in range(1, MAX_EPOCHS + 1):
        lp.train_step(train_data, model, optimizer, criterion, device)
        _, val_auc, *_ = lp.eval_step(val_data, model, criterion, device)
        if val_auc > best_auc:
            best_auc, best_state, no_imp = val_auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                stop_epoch = epoch
                print(f"Early stop @ {epoch}; no val-AUC gain for {PATIENCE} ep.")
                break
    model.load_state_dict(best_state)
    print(f"stop_epoch={stop_epoch}  best val_auc={best_auc:.4f}")
    torch.save({"state_dict": best_state, "hidden": HIDDEN, "dropout": DROPOUT,
                "in_manuf": in_manuf, "in_prod": in_prod, "rels": rels},
               "experiment1_sage_checkpoint.pt")
    print("saved experiment1_sage_checkpoint.pt")

    _, test_auc, test_labels, test_probs, _ = lp.eval_step(test_data, model, criterion, device)
    test_labels, test_probs = test_labels.numpy(), test_probs.numpy()
    print(f"\nOverall SAGE test AUC={test_auc:.4f}  PR-AUC={average_precision_score(test_labels, test_probs):.4f}")

    # ---- bucket by FULL-graph product-sharing count (same as experiment1) --
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

    idx = test_data[rel].edge_label_index.cpu().numpy()
    p_idx = idx[1]
    buckets = np.array([bucket_for_count(c) for c in full_count[p_idx]], dtype=object)

    print(f"\n{'bucket':<20} {'n':>7} {'n_pos':>7} {'n_neg':>7} {'AUC':>8} {'PR-AUC':>8}")
    rows = []
    for b in sorted(set(buckets)):
        mask = buckets == b
        lb, pr = test_labels[mask], test_probs[mask]
        n_pos, n_neg = int((lb == 1).sum()), int((lb == 0).sum())
        if len(set(lb)) < 2:
            auc, prauc = float('nan'), float('nan')
        else:
            auc, prauc = roc_auc_score(lb, pr), average_precision_score(lb, pr)
        print(f"{b:<20} {mask.sum():>7} {n_pos:>7} {n_neg:>7} {auc:>8.4f} {prauc:>8.4f}")
        rows.append((b, int(mask.sum()), n_pos, n_neg, auc, prauc))

    with open("experiment1_sage_results.md", "w", encoding="utf-8") as f:
        f.write("# Experiment 1: SAGE on corrected Graph_A, frozen split (seed=42)\n\n")
        f.write(f"Overall test AUC={test_auc:.4f}  PR-AUC={average_precision_score(test_labels, test_probs):.4f}  "
                f"(lr={LR}, hidden={HIDDEN}, dropout={DROPOUT}, loss={LOSS_TYPE}, stop_epoch={stop_epoch})\n\n")
        f.write("| Bucket | n | n_pos | n_neg | AUC | PR-AUC |\n|---|---|---|---|---|---|\n")
        for b, n, npos, nneg, auc, prauc in rows:
            f.write(f"| {b} | {n} | {npos} | {nneg} | {auc:.4f} | {prauc:.4f} |\n")
    print("\nwrote experiment1_sage_results.md")


if __name__ == "__main__":
    main()
