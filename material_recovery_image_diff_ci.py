#!/usr/bin/env python3
"""
Manufacturer-level bootstrap 95% CI for the REAL-image vs ZEROED-image
Recall@5/@10 difference (MLP track, has_image group) -- answers "how much
test-sample uncertainty is there around the ~1pp image increment," a
DIFFERENT question from the already-reported cross-seed mean+-std (which
answers "how much does the number move if we retrain with a different
seed"). The two are reported side by side, not merged into one number.

No raw per-manufacturer score matrix was saved by the original runs (same
situation as material_recovery_hit_at_1.py), so this re-executes training
once per seed/condition with the EXACT already-selected LR (no grid search,
no new hyperparameter choice) -- not a literal zero-rerun, but no new
tuning either. MLP training has no conv kernels, so it does not carry the
GPU/cuDNN non-determinism the GNN scripts do -- it reproduces exactly on
the SAME machine/environment that produced the archived numbers (confirmed
repeatedly in this project). An independent third-party rerun on a
DIFFERENT machine found this does NOT extend across environments: ordinary
CPU floating-point non-associativity (no BLAS/thread-count pinning) can tip
the per-seed LR grid-search onto a different selected LR, giving a close
but not byte-identical result (~0.1-0.15pp off, same qualitative finding) --
see the paper's Section 7.1 third-party verification report.
"""
import importlib.util
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("mb", "material_recovery_mlp_baseline.py")
mb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mb)

REAL_IMAGE_LR = {42: 5e-3, 43: 1e-2, 44: 5e-3}
ZEROED_IMAGE_LR = {42: 1e-2, 43: 1e-2, 44: 1e-2}
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

    ks = mb.mrp.EVAL_KS

    def train_and_score(feat, lr, seed):
        model, head, val_r10, stop_ep = mb.train_one_mlp(feat, device, edges, lr, seed)
        model.eval(); head.eval()
        with torch.no_grad():
            manu_h = model(feat)
            all_scores = head.score_all(manu_h[torch.tensor(manu_idx_all, device=device)]).cpu().numpy()
        return {m: all_scores[i] for i, m in enumerate(manu_idx_all)}

    report_lines = []
    report_lines.append("# Manufacturer-level bootstrap CI for REAL-ZEROED image difference (MLP, has_image)\n\n")
    report_lines.append("Test-sample uncertainty (this file) is a DIFFERENT dimension from the already-\n")
    report_lines.append("reported cross-seed mean+-std (training-seed randomness) -- reported side by side,\n")
    report_lines.append("not combined into one number.\n\n")
    report_lines.append(f"Bootstrap: manufacturer-level resampling with replacement, n={N_BOOTSTRAP}, seed={BOOTSTRAP_SEED}.\n\n")
    report_lines.append("| seed | metric | point diff (real-zeroed) | bootstrap mean | 95% CI low | 95% CI high | excludes 0? |\n")
    report_lines.append("|---|---|---|---|---|---|---|\n")

    per_seed_point = {5: [], 10: []}
    for seed in SEEDS:
        scores_real = train_and_score(feat_real, REAL_IMAGE_LR[seed], seed)
        scores_zeroed = train_and_score(feat_zeroed, ZEROED_IMAGE_LR[seed], seed)

        manus_r, recall_real = per_manu_recall(has_image_manus, pos_by_manu_test, scores_real, ks)
        manus_z, recall_zeroed = per_manu_recall(has_image_manus, pos_by_manu_test, scores_zeroed, ks)
        assert manus_r == manus_z, "manufacturer order mismatch between real/zeroed scoring"

        print(f"\n=== seed={seed} ===")
        for k in ks:
            diffs = recall_real[k] - recall_zeroed[k]
            point = float(diffs.mean())
            boot_mean, lo, hi = bootstrap_ci(diffs)
            per_seed_point[k].append(point)
            excl = "yes" if (lo > 0 or hi < 0) else "no"
            print(f"  Recall@{k}: point_diff={point:+.4f} boot_mean={boot_mean:+.4f} "
                  f"95%CI=({lo:+.4f},{hi:+.4f}) excludes_0={excl} n_manu={len(diffs)}")
            report_lines.append(f"| {seed} | Recall@{k} | {point:+.4f} | {boot_mean:+.4f} | {lo:+.4f} | {hi:+.4f} | {excl} |\n")

    report_lines.append("\n## Cross-seed mean +- sample std (ddof=1) -- training-randomness dimension, already reported in the paper\n\n")
    for k in ks:
        vals = np.array(per_seed_point[k])
        line = f"Recall@{k} real-zeroed, has_image: {vals.mean():+.4f} +- {vals.std(ddof=1):.4f} (this script's recomputation)"
        print(line)
        report_lines.append(f"- {line}\n")

    with open("material_recovery_image_diff_ci_results.md", "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    print("\nwrote material_recovery_image_diff_ci_results.md")


if __name__ == "__main__":
    main()
