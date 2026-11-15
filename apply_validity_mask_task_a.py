#!/usr/bin/env python3
"""
User-directed check: the paper's main Task A results were computed on the
UNMASKED graph (manufacturer_validity_mask.pt excludes 3 non-manufacturer
nodes -- indices 0, 5516, 6236 -- and 3 contaminated edges from manufacturer
1712, found in the Section 3.5 known-issue screen). Few entries are
affected, so this does a filtered RE-SCORE of the already-archived SAGE
checkpoint (no retraining, no graph rebuild/reindexing) rather than a full
rerun: evaluate the same frozen checkpoint on the test set with any pair
touching an excluded manufacturer or an excluded edge removed from the
scoring population, and report the delta against the archived (unmasked)
numbers.
"""
import importlib.util
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

spec = importlib.util.spec_from_file_location("lp_abcd", "LP-ABCD.py")
lp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp)

GENERIC_THRESHOLD = 20
EXCLUDED_MANU_IDX = {0, 5516, 6236}
EXCLUDED_EDGE_PAIRS = {(1712, 58055), (1712, 58058), (1712, 58059)}


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load("experiment1_sage_checkpoint.pt", map_location=device, weights_only=False)
    split = torch.load("frozen_split_GraphA_seed42.pt", map_location=device, weights_only=False)
    test_data = split["test"]
    rel = ("manufacturer", "has_product", "product")

    model = lp.HeteroSAGE(ckpt["in_manuf"], ckpt["in_prod"], ckpt["hidden"], ckpt["rels"], ckpt["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    criterion = lp.FocalLoss(alpha=0.25, gamma=2.0)

    _, test_auc, test_labels, test_probs, _ = lp.eval_step(test_data, model, criterion, device)
    test_labels, test_probs = test_labels.numpy(), test_probs.numpy()

    idx = test_data[rel].edge_label_index.cpu().numpy()
    m_idx, p_idx = idx[0], idx[1]

    prod_to_manus = defaultdict(set)
    edge_set = set()
    with open("Cleaned_Master_CSV_v3.csv", newline='', encoding='utf-8-sig', errors='replace') as f:
        import csv
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
    buckets = np.array([bucket_for_count(c) for c in full_count[p_idx]], dtype=object)

    pairs = list(zip(m_idx.tolist(), p_idx.tolist()))
    excluded_mask = np.array([
        (m in EXCLUDED_MANU_IDX) or ((m, p) in EXCLUDED_EDGE_PAIRS) for m, p in pairs
    ])
    print(f"test pairs touching the validity mask: {excluded_mask.sum()} / {len(excluded_mask)}")

    keep = ~excluded_mask
    overall_auc_unmasked = roc_auc_score(test_labels, test_probs)
    overall_prauc_unmasked = average_precision_score(test_labels, test_probs)
    overall_auc_masked = roc_auc_score(test_labels[keep], test_probs[keep])
    overall_prauc_masked = average_precision_score(test_labels[keep], test_probs[keep])

    singleton_mask = buckets == "1_singleton"
    sing_auc_unmasked = roc_auc_score(test_labels[singleton_mask], test_probs[singleton_mask])
    sing_keep = singleton_mask & keep
    sing_auc_masked = roc_auc_score(test_labels[sing_keep], test_probs[sing_keep])

    lines = []
    lines.append("# Task A: validity-mask filtered re-score of the archived SAGE checkpoint\n")
    lines.append(
        "No retraining, no graph rebuild/reindexing -- the archived checkpoint "
        "(experiment1_sage_checkpoint.pt) is re-scored with any test pair touching "
        "an excluded manufacturer (indices 0, 5516, 6236) or an excluded edge "
        "((1712,58055)/(1712,58058)/(1712,58059), per manufacturer_validity_mask.pt) "
        "removed from the scoring population.\n"
    )
    lines.append(f"\nTest pairs touching the validity mask: {excluded_mask.sum()} / {len(excluded_mask)} "
                  f"({100*excluded_mask.sum()/len(excluded_mask):.3f}%)\n")
    lines.append("\n| Metric | Unmasked (archived) | Masked (filtered) | Delta |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Overall AUC | {overall_auc_unmasked:.4f} | {overall_auc_masked:.4f} | "
                  f"{overall_auc_masked-overall_auc_unmasked:+.4f} |")
    lines.append(f"| Overall PR-AUC | {overall_prauc_unmasked:.4f} | {overall_prauc_masked:.4f} | "
                  f"{overall_prauc_masked-overall_prauc_unmasked:+.4f} |")
    lines.append(f"| Singleton AUC | {sing_auc_unmasked:.4f} | {sing_auc_masked:.4f} | "
                  f"{sing_auc_masked-sing_auc_unmasked:+.4f} |")

    out = "\n".join(lines) + "\n"
    print("\n" + out)
    with open("apply_validity_mask_task_a_results.md", "w", encoding="utf-8") as f:
        f.write(out)
    print("wrote apply_validity_mask_task_a_results.md")


if __name__ == "__main__":
    main()
