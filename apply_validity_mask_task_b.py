#!/usr/bin/env python3
"""
User-directed check: Task B's main attributes-only MLP baseline
(material_recovery_mlp_baseline.py) was trained on the UNMASKED graph.
manufacturer_validity_mask.pt's 3 excluded manufacturer indices (0, 5516,
6236) map to domains apaa.wildapricot.org, www.krffinancialllc.weebly.com,
www.meridianmanor.blogspot.com. Checked directly against
material_task_split.json: only index 0 (apaa.wildapricot.org) is in ANY
Task B split, and it's in TRAIN only -- confirming the paper's existing
claim that the other two aren't in the scored population and only one
contributes a (noisy) training example. This reruns the identical 3-seed
protocol with that one manufacturer's training edges removed, to show the
masked-training delta empirically rather than assert it's negligible.
"""
import importlib.util
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("mb", "material_recovery_mlp_baseline.py")
mb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mb)

EXCLUDED_MANU_IDX = 0  # apaa.wildapricot.org -- the only one of the 3 validity-mask
                        # exclusions that falls in any Task B split (train only)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edges = torch.load("material_edges.pt", map_location="cpu", weights_only=False)
    cov = torch.load("manufacturer_coverage_flags.pt", map_location="cpu", weights_only=False)
    has_image = cov["has_image"].numpy().astype(bool)
    feat = torch.load("manufacturer_features_no_material.pt", map_location=device, weights_only=False)

    assert edges["manus_sorted"][EXCLUDED_MANU_IDX] == "apaa.wildapricot.org"

    train_m = edges["train_manu_idx"]
    train_v = edges["train_mat_idx"]
    keep = train_m != EXCLUDED_MANU_IDX
    n_dropped = (~keep).sum().item()
    print(f"dropping {n_dropped} training edge(s) belonging to manufacturer index {EXCLUDED_MANU_IDX} "
          f"({edges['manus_sorted'][EXCLUDED_MANU_IDX]})")
    edges_masked = dict(edges)
    edges_masked["train_manu_idx"] = train_m[keep]
    edges_masked["train_mat_idx"] = train_v[keep]

    pos_by_manu_test = {}
    for m, v in zip(edges["test_manu_idx"].tolist(), edges["test_mat_idx"].tolist()):
        pos_by_manu_test.setdefault(m, set()).add(v)
    manu_idx_all = sorted(pos_by_manu_test.keys())
    groups = {
        "all": manu_idx_all,
        "has_image": [m for m in manu_idx_all if has_image[m]],
        "no_image": [m for m in manu_idx_all if not has_image[m]],
    }
    assert EXCLUDED_MANU_IDX not in manu_idx_all, "excluded manufacturer unexpectedly in the test-scored population"

    all_results = {}
    for seed in mb.SEEDS:
        print(f"\n=== seed={seed} condition=feature_mlp_no_graph, validity-masked training ===")
        lr, val_r10, model, head = mb.select_lr_mlp(feat, device, edges_masked, seed)
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
        all_results[seed] = entry

    print("\n=== mean +- sample std (ddof=1) across 3 seeds, validity-masked training ===")
    masked_summary = {}
    for g in groups:
        r5s = np.array([all_results[s]["groups"][g]["recall5"] for s in mb.SEEDS])
        r10s = np.array([all_results[s]["groups"][g]["recall10"] for s in mb.SEEDS])
        masked_summary[g] = (r5s.mean(), r5s.std(ddof=1), r10s.mean(), r10s.std(ddof=1))
        print(f"  [{g}] Recall@5={r5s.mean():.4f}+-{r5s.std(ddof=1):.4f}  "
              f"Recall@10={r10s.mean():.4f}+-{r10s.std(ddof=1):.4f}")

    # archived (unmasked) reference numbers, read directly from the archived results file
    import re
    archived = {}
    with open("material_recovery_mlp_baseline_results.md", encoding="utf-8") as f:
        text = f.read()
    for g in groups:
        m = re.search(
            rf"feature_mlp_no_graph \[{g}\]: Recall@5=([\d.]+)\+-([\d.]+)\s+Recall@10=([\d.]+)\+-([\d.]+)",
            text,
        )
        archived[g] = tuple(float(x) for x in m.groups())

    lines = []
    lines.append("# Task B: validity-mask filtered retraining of the attributes-only MLP baseline\n")
    lines.append(
        f"Reruns `material_recovery_mlp_baseline.py`'s identical 3-seed protocol "
        f"(seeds 42/43/44, same LR grid/val selection/eval) with manufacturer index "
        f"{EXCLUDED_MANU_IDX} (`{edges['manus_sorted'][EXCLUDED_MANU_IDX]}`, one of the 3 "
        f"validity-mask exclusions, and the only one present in any Task B split -- in TRAIN "
        f"only) removed from training. The other 2 excluded manufacturers are already outside "
        f"every Task B split, confirming the paper's existing claim.\n"
    )
    lines.append("\n| Group | Recall@5 (archived, unmasked) | Recall@5 (masked) | Delta | "
                  "Recall@10 (archived) | Recall@10 (masked) | Delta |")
    lines.append("|---|---|---|---|---|---|---|")
    for g in groups:
        a_r5, a_r5sd, a_r10, a_r10sd = archived[g]
        m_r5, m_r5sd, m_r10, m_r10sd = masked_summary[g]
        lines.append(
            f"| {g} | {a_r5:.4f}±{a_r5sd:.4f} | {m_r5:.4f}±{m_r5sd:.4f} | {m_r5-a_r5:+.4f} | "
            f"{a_r10:.4f}±{a_r10sd:.4f} | {m_r10:.4f}±{m_r10sd:.4f} | {m_r10-a_r10:+.4f} |"
        )

    out = "\n".join(lines) + "\n"
    print("\n" + out)
    with open("apply_validity_mask_task_b_results.md", "w", encoding="utf-8") as f:
        f.write(out)
    print("wrote apply_validity_mask_task_b_results.md")


if __name__ == "__main__":
    main()
