#!/usr/bin/env python3
"""
GNN (Design A) counterpart to material_recovery_image_diff_ci.py: manufacturer-
level bootstrap 95% CI for the REAL-image vs ZEROED-image Recall@5/@10
difference, has_image group. Same "no new tuning" standard as the MLP script
and material_recovery_hit_at_1.py: reuses the EXACT already-selected LR per
seed/condition from material_recovery_multiseed_results.md's own table (42:
real=1e-2/zeroed=1e-2; 43: real=1e-2/zeroed=1e-2; 44: real=1e-3/zeroed=5e-3),
no grid search.

Unlike the MLP track, the GNN involves conv kernels -- GPU/cuDNN training is
NOT bit-reproducible across runs even at a fixed seed (already established in
this project, Section 5.3's independent-rerun disclosure). This rerun's point
estimates are therefore NOT expected to exactly match the archived table; this
is reported plainly, not treated as a discrepancy to chase.
"""
import importlib.util
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("mrp", "material_recovery_pilot.py")
mrp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mrp)

REAL_LR = {42: 1e-2, 43: 1e-2, 44: 1e-3}
ZEROED_LR = {42: 1e-2, 43: 1e-2, 44: 5e-3}
SEEDS = [42, 43, 44]
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 777


def per_manu_recall(manu_idx_list, pos_by_manu, scores_by_row, ks):
    out = {k: [] for k in ks}
    manus_scored = []
    for m in manu_idx_list:
        pos = pos_by_manu.get(m)
        if not pos:
            continue
        scores = scores_by_row[m]
        order = np.argsort(-scores)
        for k in ks:
            topk = set(order[:k].tolist())
            out[k].append(len(topk & pos) / len(pos))
        manus_scored.append(m)
    return manus_scored, {k: np.array(v) for k, v in out.items()}


def bootstrap_ci(diffs, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    rng = np.random.RandomState(seed)
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = diffs[idx].mean()
    return float(boot_means.mean()), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load("hetero_graphs/Graph_A.pt", map_location=device, weights_only=False)
    edges = torch.load("material_edges.pt", map_location="cpu", weights_only=False)
    cov = torch.load("manufacturer_coverage_flags.pt", map_location="cpu", weights_only=False)
    has_image = cov["has_image"].numpy().astype(bool)

    no_mat_feat = torch.load("manufacturer_features_no_material.pt", map_location=device, weights_only=False)
    img_real = torch.load("manufacturer_features_real.pt", map_location=device, weights_only=False)
    has_image_t = cov["has_image"].to(device).unsqueeze(1)
    img_part_real = img_real[:, 65:]
    feat_real = torch.cat([no_mat_feat, has_image_t, img_part_real], dim=1)
    feat_zeroed = torch.cat([no_mat_feat, has_image_t, torch.zeros_like(img_part_real)], dim=1)

    pos_by_manu_test = {}
    for m, v in zip(edges["test_manu_idx"].tolist(), edges["test_mat_idx"].tolist()):
        pos_by_manu_test.setdefault(m, set()).add(v)
    manu_idx_all = sorted(pos_by_manu_test.keys())
    has_image_manus = [m for m in manu_idx_all if has_image[m]]
    ks = mrp.EVAL_KS

    def train_and_score(feat, lr, seed):
        model, head, d, r10, stop_ep = mrp.train_one(feat, device, data, edges, lr, seed)
        print(f"    seed={seed} lr={lr:.1e} val_R@10={r10:.4f} stop_epoch={stop_ep}")
        model.eval(); head.eval()
        with torch.no_grad():
            h = model(d.x_dict, d.edge_index_dict)
            manu_h = h["manufacturer"]
            all_scores = head.score_all(manu_h[torch.tensor(manu_idx_all, device=device)]).cpu().numpy()
        return {m: all_scores[i] for i, m in enumerate(manu_idx_all)}

    report_lines = []
    report_lines.append("# GNN (Design A): manufacturer-level bootstrap CI for REAL-ZEROED image difference\n\n")
    report_lines.append("Reused already-selected per-seed LRs (no grid search). GNN training is NOT\n")
    report_lines.append("bit-reproducible across runs (GPU/cuDNN, already established in Section 5.3) --\n")
    report_lines.append("point estimates here are expected to differ somewhat from the archived table;\n")
    report_lines.append("this is disclosed, not treated as an error.\n\n")
    report_lines.append(f"Bootstrap: manufacturer-level resampling with replacement, n={N_BOOTSTRAP}, seed={BOOTSTRAP_SEED}.\n\n")
    report_lines.append("| seed | metric | point diff (real-zeroed) | bootstrap mean | 95% CI low | 95% CI high | excludes 0? |\n")
    report_lines.append("|---|---|---|---|---|---|---|\n")

    per_seed_point = {5: [], 10: []}
    for seed in SEEDS:
        print(f"\n=== seed={seed} ===")
        scores_real = train_and_score(feat_real, REAL_LR[seed], seed)
        scores_zeroed = train_and_score(feat_zeroed, ZEROED_LR[seed], seed)

        manus_r, recall_real = per_manu_recall(has_image_manus, pos_by_manu_test, scores_real, ks)
        manus_z, recall_zeroed = per_manu_recall(has_image_manus, pos_by_manu_test, scores_zeroed, ks)
        assert manus_r == manus_z, "manufacturer order mismatch between real/zeroed scoring"

        for k in ks:
            diffs = recall_real[k] - recall_zeroed[k]
            point = float(diffs.mean())
            boot_mean, lo, hi = bootstrap_ci(diffs)
            per_seed_point[k].append(point)
            excl = "yes" if (lo > 0 or hi < 0) else "no"
            print(f"  Recall@{k}: point_diff={point:+.4f} boot_mean={boot_mean:+.4f} "
                  f"95%CI=({lo:+.4f},{hi:+.4f}) excludes_0={excl} n_manu={len(diffs)}")
            report_lines.append(f"| {seed} | Recall@{k} | {point:+.4f} | {boot_mean:+.4f} | {lo:+.4f} | {hi:+.4f} | {excl} |\n")

    report_lines.append("\n## Cross-seed mean +- sample std (ddof=1) -- this rerun's own value, for comparison against the archived table\n\n")
    for k in ks:
        vals = np.array(per_seed_point[k])
        line = f"Recall@{k} real-zeroed, has_image: {vals.mean():+.4f} +- {vals.std(ddof=1):.4f} (this rerun; archived table: see material_recovery_multiseed_results.md)"
        print(line)
        report_lines.append(f"- {line}\n")

    with open("material_recovery_gnn_image_diff_ci_results.md", "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    print("\nwrote material_recovery_gnn_image_diff_ci_results.md")


if __name__ == "__main__":
    main()
