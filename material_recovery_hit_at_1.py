#!/usr/bin/env python3
"""
Adds Hit@1 (= Precision@1: is the rank-1 prediction one of the manufacturer's
RECORDED positives?) alongside the already-reported Recall@5/@10, for the
MLP track (attrs-only / +real / +zeroed). No raw per-manufacturer score
matrix was saved by the original runs, so this re-executes training once per
seed/condition -- but with the EXACT already-selected LR (no grid search, no
new hyperparameter choice), which is why this is "no new tuning," not a
literal zero-rerun. R5/R10 are recomputed too, as a built-in consistency
check against the already-reported numbers (the MLP has no conv kernels, so
same seed+LR should reproduce them exactly).

Caveat carried into the paper verbatim: a miss at rank-1 may reflect the
field's own under-recording (per the 17.4%/24.6% inconsistency rates), not
necessarily a true capability-prediction error.
"""
import importlib.util
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("mb", "material_recovery_mlp_baseline.py")
mb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mb)

# Exact LRs already selected by the original (grid-searched) runs -- reused,
# not re-searched.
ATTRS_ONLY_LR = {42: 1e-3, 43: 5e-3, 44: 5e-3}
REAL_IMAGE_LR = {42: 5e-3, 43: 1e-2, 44: 5e-3}
ZEROED_IMAGE_LR = {42: 1e-2, 43: 1e-2, 44: 1e-2}
SEEDS = [42, 43, 44]


def train_fixed_lr(feat, device, edges, seed, lr):
    model, head, val_r10, stop_ep = mb.train_one_mlp(feat, device, edges, lr, seed)
    return model, head, val_r10, stop_ep


def hit_at_1(manu_idx_list, pos_by_manu, scores_by_row):
    hits = []
    for m in manu_idx_list:
        top1 = int(np.argmax(scores_by_row[m]))
        hits.append(1 if top1 in pos_by_manu[m] else 0)
    return float(np.mean(hits)), hits


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edges = torch.load("material_edges.pt", map_location="cpu", weights_only=False)
    cov = torch.load("manufacturer_coverage_flags.pt", map_location="cpu", weights_only=False)
    has_image = cov["has_image"].numpy().astype(bool)
    no_mat_feat = torch.load("manufacturer_features_no_material.pt", map_location=device, weights_only=False)
    img_real = torch.load("manufacturer_features_real.pt", map_location=device, weights_only=False)
    has_image_t = cov["has_image"].to(device).unsqueeze(1)
    img_part_real = img_real[:, 65:]

    feat_attrs = no_mat_feat
    feat_real = torch.cat([no_mat_feat, has_image_t, img_part_real], dim=1)
    feat_zeroed = torch.cat([no_mat_feat, has_image_t, torch.zeros_like(img_part_real)], dim=1)

    conditions = {
        "attrs_only": (feat_attrs, ATTRS_ONLY_LR),
        "real_image": (feat_real, REAL_IMAGE_LR),
        "zeroed_image": (feat_zeroed, ZEROED_IMAGE_LR),
    }

    pos_by_manu_test = {}
    for m, v in zip(edges["test_manu_idx"].tolist(), edges["test_mat_idx"].tolist()):
        pos_by_manu_test.setdefault(m, set()).add(v)
    manu_idx_all = sorted(pos_by_manu_test.keys())
    groups = {"all": manu_idx_all,
              "has_image": [m for m in manu_idx_all if has_image[m]],
              "no_image": [m for m in manu_idx_all if not has_image[m]]}

    all_results = {}
    for cond_name, (feat, lr_by_seed) in conditions.items():
        for seed in SEEDS:
            lr = lr_by_seed[seed]
            print(f"\n=== {cond_name} seed={seed} lr={lr:.1e} (reused, not re-searched) ===")
            model, head, val_r10, stop_ep = train_fixed_lr(feat, device, edges, seed, lr)
            print(f"  val_R@10={val_r10:.4f} stop_epoch={stop_ep}")
            model.eval(); head.eval()
            with torch.no_grad():
                manu_h = model(feat)
                all_scores = head.score_all(manu_h[torch.tensor(manu_idx_all, device=device)]).cpu().numpy()
            scores_by_row = {m: all_scores[i] for i, m in enumerate(manu_idx_all)}
            entry = {"lr": lr, "val_r10": val_r10, "groups": {}}
            for g, idxs in groups.items():
                r = mb.mrp.recall_at_k(idxs, pos_by_manu_test, scores_by_row, mb.mrp.EVAL_KS)
                h1, _ = hit_at_1(idxs, pos_by_manu_test, scores_by_row)
                entry["groups"][g] = {"recall5": r[5], "recall10": r[10], "hit1": h1}
                print(f"  [{g}] Recall@5={r[5]:.4f} Recall@10={r[10]:.4f} Hit@1={h1:.4f}")
            all_results[(cond_name, seed)] = entry

    print("\n=== mean +- sample std (ddof=1), has_image group ===")
    summary_lines = []
    for cond_name in conditions:
        for metric in ["recall5", "recall10", "hit1"]:
            vals = np.array([all_results[(cond_name, s)]["groups"]["has_image"][metric] for s in SEEDS])
            line = f"{cond_name} has_image {metric}: {vals.mean():.4f} +- {vals.std(ddof=1):.4f}"
            print(" ", line)
            summary_lines.append(line)

    print("\n=== paired Hit@1 differences, has_image group, per seed ===")
    for comp_name, (a, b) in [("real_minus_attrs", ("real_image", "attrs_only")),
                               ("real_minus_zeroed", ("real_image", "zeroed_image"))]:
        diffs = []
        for s in SEEDS:
            d = (all_results[(a, s)]["groups"]["has_image"]["hit1"] -
                 all_results[(b, s)]["groups"]["has_image"]["hit1"])
            diffs.append(d)
            print(f"  {comp_name} seed={s}: dHit@1={d:+.4f}")
        diffs = np.array(diffs)
        print(f"  {comp_name} mean+-std: {diffs.mean():+.4f}+-{diffs.std(ddof=1):.4f}")

    with open("material_recovery_hit_at_1_results.md", "w", encoding="utf-8") as f:
        f.write("# Hit@1 (post-hoc analysis, added after seeing test results -- see caveat)\n\n")
        f.write("**Post-hoc disclosure**: this metric was computed after the R@5/@10 test results\n")
        f.write("were already seen. It supplements, does not replace, those primary results.\n")
        f.write("Re-executes training with the ALREADY-SELECTED LR per seed (no grid search, no\n")
        f.write("new hyperparameter choice) because the original runs never saved a raw\n")
        f.write("per-manufacturer score matrix. R5/R10 are recomputed as a consistency check.\n\n")
        f.write("**Caveat**: a miss at rank-1 may reflect the field's own under-recording\n")
        f.write("(17.4%/24.6% inconsistency rates, Section 4.2), not necessarily a true\n")
        f.write("capability-prediction error.\n\n")
        f.write("| condition | seed | LR (reused) | group | Recall@5 | Recall@10 | Hit@1 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for (cond, seed), entry in all_results.items():
            for g, r in entry["groups"].items():
                f.write(f"| {cond} | {seed} | {entry['lr']:.1e} | {g} | {r['recall5']:.4f} | "
                        f"{r['recall10']:.4f} | {r['hit1']:.4f} |\n")
        f.write("\n## Mean +- sample std (ddof=1), has_image group\n\n")
        for line in summary_lines:
            f.write(f"- {line}\n")

    print("\nwrote material_recovery_hit_at_1_results.md")


if __name__ == "__main__":
    main()
