#!/usr/bin/env python3
"""
Completes the 2x3 {MLP, GNN} x {attrs-only, attrs+real, attrs+zeroed} table.
Answers: is Design A's weak-but-consistent image signal specific to the GNN
(graph structure needed to exploit image content), or does it show up for a
plain feature model too (image content itself carries signal, independent
of graph propagation)?

Reuses FeatureMLP (material_recovery_mlp_baseline.py, no message passing at
all) with the SAME 97-dim concatenated input Design A's GNN uses: 64-dim
attribute feature + 1-dim has_image flag + 32-dim SVD image slice -- the
existing concat-style fusion, not a new design, per explicit instruction not
to introduce a new fusion architecture for this comparison. Same task/split/
vocab/negative-sampling/eval/val-only-LR-selection/3-seed protocol as every
other condition in this table.
"""
import importlib.util
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("mb", "material_recovery_mlp_baseline.py")
mb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mb)

SEEDS = mb.SEEDS


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edges = torch.load("material_edges.pt", map_location="cpu", weights_only=False)
    cov = torch.load("manufacturer_coverage_flags.pt", map_location="cpu", weights_only=False)
    has_image = cov["has_image"].numpy().astype(bool)

    no_mat_feat = torch.load("manufacturer_features_no_material.pt", map_location=device, weights_only=False)
    # RELEASE PACKAGE NOTE: see material_recovery_pilot.py's equivalent note -- uses
    # only the verified-safe 32-dim image slice from the combined release asset.
    img_asset = torch.load("manufacturer_image_release_asset.pt", map_location=device, weights_only=False)
    img_part_real = img_asset["embedding_32dim_svd"].to(device)
    has_image_t = cov["has_image"].to(device).unsqueeze(1)

    feat_real = torch.cat([no_mat_feat, has_image_t, img_part_real], dim=1)
    feat_zeroed = torch.cat([no_mat_feat, has_image_t, torch.zeros_like(img_part_real)], dim=1)
    conditions = {"mlp_real_image": feat_real, "mlp_zeroed_image": feat_zeroed}

    pos_by_manu_test = {}
    for m, v in zip(edges["test_manu_idx"].tolist(), edges["test_mat_idx"].tolist()):
        pos_by_manu_test.setdefault(m, set()).add(v)
    manu_idx_all = sorted(pos_by_manu_test.keys())
    groups = {
        "all": manu_idx_all,
        "has_image": [m for m in manu_idx_all if has_image[m]],
        "no_image": [m for m in manu_idx_all if not has_image[m]],
    }

    all_results = {}
    for seed in SEEDS:
        for cond_name, feat in conditions.items():
            print(f"\n=== seed={seed} condition={cond_name} ===")
            lr, val_r10, model, head = mb.select_lr_mlp(feat, device, edges, seed)
            print(f"  selected lr={lr:.1e}  val_Recall@10={val_r10:.4f}")
            model.eval(); head.eval()
            with torch.no_grad():
                manu_h = model(feat)
                all_scores = head.score_all(manu_h[torch.tensor(manu_idx_all, device=device)]).cpu().numpy()
            scores_by_row = {m: all_scores[i] for i, m in enumerate(manu_idx_all)}
            entry = {"lr": lr, "val_r10": val_r10, "groups": {}}
            for g, idxs in groups.items():
                r = mb.mrp.recall_at_k(idxs, pos_by_manu_test, scores_by_row, mb.mrp.EVAL_KS)
                entry["groups"][g] = {"recall5": r[5], "recall10": r[10]}
                print(f"  [{g}] Recall@5={r[5]:.4f} Recall@10={r[10]:.4f}")
            all_results[(seed, cond_name)] = entry

    print("\n=== paired differences (per seed, has_image group) ===")
    diff_r5, diff_r10 = [], []
    for seed in SEEDS:
        real = all_results[(seed, "mlp_real_image")]["groups"]["has_image"]
        zero = all_results[(seed, "mlp_zeroed_image")]["groups"]["has_image"]
        d5, d10 = real["recall5"] - zero["recall5"], real["recall10"] - zero["recall10"]
        diff_r5.append(d5); diff_r10.append(d10)
        print(f"  seed={seed}  real-zeroed: dR5={d5:+.4f}  dR10={d10:+.4f}")
    diff_r5, diff_r10 = np.array(diff_r5), np.array(diff_r10)
    print(f"\n  mean+-std: dR5={diff_r5.mean():+.4f}+-{diff_r5.std(ddof=1):.4f}  "
          f"dR10={diff_r10.mean():+.4f}+-{diff_r10.std(ddof=1):.4f}")

    with open("material_recovery_mlp_image_baseline_results.md", "w", encoding="utf-8") as f:
        f.write("# Material-relation recovery: MLP + image (completes the 2x3 table), 3 seeds\n\n")
        f.write("Same concat-style input Design A's GNN uses (64-dim attrs + 1-dim has_image\n")
        f.write("flag + 32-dim SVD image slice), fed into the no-message-passing FeatureMLP\n")
        f.write("instead of HeteroSAGE. No new fusion design introduced.\n\n")
        f.write("| seed | condition | LR | val R@10 | group | Recall@5 | Recall@10 |\n|---|---|---|---|---|---|---|\n")
        for (seed, cond), entry in all_results.items():
            for g, r in entry["groups"].items():
                f.write(f"| {seed} | {cond} | {entry['lr']:.1e} | {entry['val_r10']:.4f} | {g} | "
                        f"{r['recall5']:.4f} | {r['recall10']:.4f} |\n")
        f.write("\n## Real - zeroed, has_image group, per seed\n\n")
        for seed, d5, d10 in zip(SEEDS, diff_r5, diff_r10):
            f.write(f"- seed={seed}: dR5={d5:+.4f}  dR10={d10:+.4f}\n")
        f.write(f"\nmean +- sample std (ddof=1): dR5={diff_r5.mean():+.4f}+-{diff_r5.std(ddof=1):.4f}  "
                f"dR10={diff_r10.mean():+.4f}+-{diff_r10.std(ddof=1):.4f}\n")

    print("\nwrote material_recovery_mlp_image_baseline_results.md")


if __name__ == "__main__":
    main()
