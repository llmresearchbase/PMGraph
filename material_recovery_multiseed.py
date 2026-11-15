#!/usr/bin/env python3
"""
3-seed extension of material_recovery_pilot.py (negative-sampling bug already
fixed there). SAME categories, split, fusion architecture (concat + single
projection), and validation-LR-selection rule as the single-seed run --
only the training seed varies. Reports per-seed paired differences
(REAL - baseline, REAL - ZEROED) and their mean +- std across seeds.
Frequency baseline is now computed per group (all/has_image/no_image), not
just "all", so it's a fair reference for the has_image-subset comparison.
"""
import importlib.util
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("mrp", "material_recovery_pilot.py")
mrp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mrp)

SEEDS = [42, 43, 44]


def select_lr_seeded(feat, device, data, edges, seed):
    best = (None, -1, None, None, None)
    for lr in mrp.LR_GRID:
        model, head, d, r10, stop_ep = mrp.train_one(feat, device, data, edges, lr, seed)
        print(f"    seed={seed} lr={lr:.1e}  val_Recall@10={r10:.4f}  stop_epoch={stop_ep}")
        if r10 > best[1]:
            best = (lr, r10, model, head, d)
    return best


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load("hetero_graphs/Graph_A.pt", map_location=device, weights_only=False)
    edges = torch.load("material_edges.pt", map_location="cpu", weights_only=False)
    cov = torch.load("manufacturer_coverage_flags.pt", map_location="cpu", weights_only=False)
    has_image = cov["has_image"].numpy().astype(bool)

    no_mat_feat = torch.load("manufacturer_features_no_material.pt", map_location=device, weights_only=False)
    # RELEASE PACKAGE NOTE: see material_recovery_pilot.py's equivalent note -- this
    # loads only the verified-safe 32-dim image slice, not the leaky 97-dim dev file.
    img_asset = torch.load("manufacturer_image_release_asset.pt", map_location=device, weights_only=False)
    img_part_real = img_asset["embedding_32dim_svd"].to(device)
    has_image_t = cov["has_image"].to(device).unsqueeze(1)
    feat_baseline = no_mat_feat
    feat_real = torch.cat([no_mat_feat, has_image_t, img_part_real], dim=1)
    feat_zeroed = torch.cat([no_mat_feat, has_image_t, torch.zeros_like(img_part_real)], dim=1)
    conditions = {"baseline_no_image": feat_baseline, "real": feat_real, "zeroed": feat_zeroed}

    pos_by_manu_test = {}
    for m, v in zip(edges["test_manu_idx"].tolist(), edges["test_mat_idx"].tolist()):
        pos_by_manu_test.setdefault(m, set()).add(v)
    manu_idx_all = sorted(pos_by_manu_test.keys())
    groups = {
        "all": manu_idx_all,
        "has_image": [m for m in manu_idx_all if has_image[m]],
        "no_image": [m for m in manu_idx_all if not has_image[m]],
    }

    # ---- frequency baseline, per group (static, seed-independent) ----
    n_vocab = len(edges["vocab"])
    freq = np.zeros(n_vocab)
    for v in edges["train_mat_idx"].tolist():
        freq[v] += 1
    print("\n=== frequency baseline, per group ===")
    freq_results = {}
    for g, idxs in groups.items():
        scores_by_row = {m: freq for m in idxs}
        r = mrp.recall_at_k(idxs, pos_by_manu_test, scores_by_row, mrp.EVAL_KS)
        print(f"  [{g}] n={len(idxs)}  Recall@5={r[5]:.4f}  Recall@10={r[10]:.4f}")
        freq_results[g] = {"n": len(idxs), "recall5": r[5], "recall10": r[10]}

    # ---- per-seed, per-condition evaluation ----
    all_results = {}  # (seed, cond) -> {group: {recall5, recall10}, lr, val_r10}
    for seed in SEEDS:
        for cond_name, feat in conditions.items():
            print(f"\n=== seed={seed} condition={cond_name} ===")
            lr, val_r10, model, head, d = select_lr_seeded(feat, device, data, edges, seed)
            print(f"  selected lr={lr:.1e}  val_Recall@10={val_r10:.4f}")
            model.eval(); head.eval()
            with torch.no_grad():
                h = model(d.x_dict, d.edge_index_dict)
                manu_h = h["manufacturer"]
                all_scores = head.score_all(manu_h[torch.tensor(manu_idx_all, device=device)]).cpu().numpy()
            scores_by_row = {m: all_scores[i] for i, m in enumerate(manu_idx_all)}
            entry = {"lr": lr, "val_r10": val_r10, "groups": {}}
            for g, idxs in groups.items():
                r = mrp.recall_at_k(idxs, pos_by_manu_test, scores_by_row, mrp.EVAL_KS)
                entry["groups"][g] = {"recall5": r[5], "recall10": r[10]}
                print(f"  [{g}] Recall@5={r[5]:.4f} Recall@10={r[10]:.4f}")
            all_results[(seed, cond_name)] = entry

    # ---- paired differences per seed, then mean+-std across seeds ----
    print("\n=== paired differences (per seed) ===")
    diffs = {"real_minus_baseline": {g: {"r5": [], "r10": []} for g in groups},
             "real_minus_zeroed": {g: {"r5": [], "r10": []} for g in groups}}
    for seed in SEEDS:
        real = all_results[(seed, "real")]
        base = all_results[(seed, "baseline_no_image")]
        zero = all_results[(seed, "zeroed")]
        for g in groups:
            d_rb5 = real["groups"][g]["recall5"] - base["groups"][g]["recall5"]
            d_rb10 = real["groups"][g]["recall10"] - base["groups"][g]["recall10"]
            d_rz5 = real["groups"][g]["recall5"] - zero["groups"][g]["recall5"]
            d_rz10 = real["groups"][g]["recall10"] - zero["groups"][g]["recall10"]
            diffs["real_minus_baseline"][g]["r5"].append(d_rb5)
            diffs["real_minus_baseline"][g]["r10"].append(d_rb10)
            diffs["real_minus_zeroed"][g]["r5"].append(d_rz5)
            diffs["real_minus_zeroed"][g]["r10"].append(d_rz10)
            print(f"  seed={seed} [{g}] REAL-baseline: dR5={d_rb5:+.4f} dR10={d_rb10:+.4f}  "
                  f"REAL-ZEROED: dR5={d_rz5:+.4f} dR10={d_rz10:+.4f}")

    print("\n=== mean +- std across 3 seeds ===")
    summary_lines = []
    for comparison, gdict in diffs.items():
        for g, metrics in gdict.items():
            r5 = np.array(metrics["r5"]); r10 = np.array(metrics["r10"])
            # sample std (ddof=1) -- the conventional choice for mean+-std across a small
            # number of seeds; numpy's default .std() is population std (ddof=0) and
            # under-reports variance for small n, which caused a documented archive discrepancy
            line = (f"{comparison} [{g}]: dR5={r5.mean():+.4f}+-{r5.std(ddof=1):.4f}  "
                    f"dR10={r10.mean():+.4f}+-{r10.std(ddof=1):.4f}")
            print(" ", line)
            summary_lines.append(line)

    with open("material_recovery_multiseed_results.md", "w", encoding="utf-8") as f:
        f.write("# Material-relation recovery: 3-seed (42/43/44) paired comparison\n\n")
        f.write("Same categories/split/fusion/val-LR-selection as the single-seed run; "
                "only training seed varies.\n\n")
        f.write("## Frequency baseline, per group\n\n")
        f.write("| group | n | Recall@5 | Recall@10 |\n|---|---|---|---|\n")
        for g, r in freq_results.items():
            f.write(f"| {g} | {r['n']} | {r['recall5']:.4f} | {r['recall10']:.4f} |\n")
        f.write("\n## Per-seed, per-condition results\n\n")
        f.write("| seed | condition | LR | val R@10 | group | Recall@5 | Recall@10 |\n|---|---|---|---|---|---|---|\n")
        for (seed, cond), entry in all_results.items():
            for g, r in entry["groups"].items():
                f.write(f"| {seed} | {cond} | {entry['lr']:.1e} | {entry['val_r10']:.4f} | {g} | "
                        f"{r['recall5']:.4f} | {r['recall10']:.4f} |\n")
        f.write("\n## Paired differences, mean +- std across seeds\n\n")
        for line in summary_lines:
            f.write(f"- {line}\n")

    print("\nwrote material_recovery_multiseed_results.md")


if __name__ == "__main__":
    main()
