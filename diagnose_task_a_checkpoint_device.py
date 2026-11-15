#!/usr/bin/env python3
"""
Step 2 of the Task A reproducibility investigation (user-directed): load the
FIXED, already-trained checkpoint (experiment1_sage_checkpoint.pt, from the
archived run -- NOT retrained here) and run EVAL ONLY on both CPU and GPU.
If the two devices agree closely, the training-vs-archive divergence found
by the third party is a TRAINING-time phenomenon, not an inference/eval-time
one. If they disagree substantially, the problem is in inference/eval
itself and training is not the (sole) culprit.
"""
import csv
import importlib.util
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

spec = importlib.util.spec_from_file_location("lp_abcd", "LP-ABCD.py")
lp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp)

GENERIC_THRESHOLD = 20


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


def evaluate_on_device(device_str, ckpt_path):
    device = torch.device(device_str)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    split = torch.load("frozen_split_GraphA_seed42.pt", map_location=device, weights_only=False)
    data = torch.load("hetero_graphs/Graph_A.pt", map_location=device, weights_only=False)
    test_data = split["test"]
    rel = ("manufacturer", "has_product", "product")

    model = lp.HeteroSAGE(ckpt["in_manuf"], ckpt["in_prod"], ckpt["hidden"], ckpt["rels"], ckpt["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    criterion = lp.FocalLoss(alpha=0.25, gamma=2.0)

    _, test_auc, test_labels, test_probs, _ = lp.eval_step(test_data, model, criterion, device)
    test_labels, test_probs = test_labels.numpy(), test_probs.numpy()
    overall_prauc = average_precision_score(test_labels, test_probs)

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

    singleton_mask = buckets == "1_singleton"
    singleton_auc = roc_auc_score(test_labels[singleton_mask], test_probs[singleton_mask])

    return {
        "device": device_str, "overall_auc": float(test_auc), "overall_prauc": float(overall_prauc),
        "singleton_auc": float(singleton_auc),
        "probs_checksum": float(test_probs.sum()),
    }


def main():
    print("cuda available:", torch.cuda.is_available())
    ckpt_path = "experiment1_sage_checkpoint.pt"  # the Oct-10 archived checkpoint, NOT retrained

    results = {}
    for dev in (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]):
        r = evaluate_on_device(dev, ckpt_path)
        results[dev] = r
        print(f"\n=== device={dev} ===")
        print(f"  overall_auc={r['overall_auc']:.4f}  overall_prauc={r['overall_prauc']:.4f}  "
              f"singleton_auc={r['singleton_auc']:.4f}  probs_checksum={r['probs_checksum']:.6f}")

    print(f"\n=== reference (archived, experiment1_sage_results.md) ===")
    print("  overall_auc=0.5998 (approx, see file)  singleton_auc=0.7346")

    if "cuda" in results and "cpu" in results:
        d_auc = abs(results["cuda"]["overall_auc"] - results["cpu"]["overall_auc"])
        d_sing = abs(results["cuda"]["singleton_auc"] - results["cpu"]["singleton_auc"])
        print(f"\n=== CUDA vs CPU, SAME fixed checkpoint, inference-only ===")
        print(f"  |overall_auc diff| = {d_auc:.6f}")
        print(f"  |singleton_auc diff| = {d_sing:.6f}")
        print("  Interpretation: if these are near-zero, the divergence found by the "
              "third party is a TRAINING-time phenomenon, not inference/eval. If these "
              "are large, inference/eval itself is device-sensitive.")


if __name__ == "__main__":
    main()
