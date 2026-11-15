#!/usr/bin/env python3
"""
Overfitting/leakage diagnostic for material_recovery_mlp_baseline.py.
Reuses its exact train_one_mlp/select_lr_mlp (same code, same seeds, same LR
grid -- deterministic, so this reproduces the reported numbers exactly) and
additionally reports TRAIN-manufacturer Recall@K at the selected checkpoint,
plus the stop_epoch each run actually used, to check whether early stopping
on validation left a large train/test gap.
"""
import importlib.util
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("mb", "material_recovery_mlp_baseline.py")
mb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mb)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edges = torch.load("material_edges.pt", map_location="cpu", weights_only=False)
    feat = torch.load("manufacturer_features_no_material.pt", map_location=device, weights_only=False)

    pos_by_manu_train = {}
    for m, v in zip(edges["train_manu_idx"].tolist(), edges["train_mat_idx"].tolist()):
        pos_by_manu_train.setdefault(m, set()).add(v)
    pos_by_manu_val = {}
    for m, v in zip(edges["val_manu_idx"].tolist(), edges["val_mat_idx"].tolist()):
        pos_by_manu_val.setdefault(m, set()).add(v)
    pos_by_manu_test = {}
    for m, v in zip(edges["test_manu_idx"].tolist(), edges["test_mat_idx"].tolist()):
        pos_by_manu_test.setdefault(m, set()).add(v)

    for seed in mb.SEEDS:
        print(f"\n=== seed={seed} ===")
        best = (None, -1, None, None)
        for lr in mb.mrp.LR_GRID:
            model, head, r10, stop_ep = mb.train_one_mlp(feat, device, edges, lr, seed)
            print(f"  lr={lr:.1e}  val_R@10={r10:.4f}  stop_epoch={stop_ep}")
            if r10 > best[1]:
                best = (lr, r10, model, head, stop_ep)
        lr, val_r10, model, head, stop_ep = best
        print(f"  selected lr={lr:.1e} val_R@10={val_r10:.4f} stop_epoch={stop_ep}")

        model.eval(); head.eval()
        with torch.no_grad():
            manu_h = model(feat)
            for name, pos_by_manu in [("TRAIN", pos_by_manu_train), ("VAL", pos_by_manu_val), ("TEST", pos_by_manu_test)]:
                idx = sorted(pos_by_manu.keys())
                scores = head.score_all(manu_h[torch.tensor(idx, device=device)]).cpu().numpy()
                scores_by_row = {m: scores[i] for i, m in enumerate(idx)}
                r = mb.mrp.recall_at_k(idx, pos_by_manu, scores_by_row, mb.mrp.EVAL_KS)
                print(f"    {name:5s} n={len(idx):5d}  Recall@5={r[5]:.4f}  Recall@10={r[10]:.4f}")


if __name__ == "__main__":
    main()
