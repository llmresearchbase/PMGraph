#!/usr/bin/env python3
"""
No-message-passing MLP baseline for the material-relation-recovery task.
Completes the evidence chain the paper needs: frequency prior -> feature MLP
(this script) -> GNN (material_recovery_pilot.py's baseline_no_image) -> GNN+image
(real/zeroed). Isolates how much value comes from the manufacturer's own
64-dim attribute feature ALONE, with zero graph structure, so the GNN's
contribution (message passing over the product-relation graph) can be read
as an increment over this, not assumed.

Same 56-category frozen vocab, same manufacturer-level split, same
negative-sampling-with-exclusion rule, same Recall@5/@10 eval, same
val-only LR selection from the same 5-value grid, same 3 seeds (42/43/44) --
every axis held identical to material_recovery_pilot.py/multiseed.py except
the manufacturer encoder itself, which here is a plain 2-layer MLP over the
64-dim feature with NO access to the manufacturer-product graph at all.
"""
import importlib.util
import numpy as np
import torch
import torch.nn as nn

spec = importlib.util.spec_from_file_location("mrp", "material_recovery_pilot.py")
mrp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mrp)

HIDDEN, DROPOUT = 128, 0.3
SEEDS = [42, 43, 44]


class FeatureMLP(nn.Module):
    """No graph access whatsoever -- a manufacturer's embedding depends only
    on its own 64-dim attribute feature, never on which products it lists."""
    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x):
        return self.net(x)


def train_one_mlp(feat, device, edges, lr, seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    model = FeatureMLP(feat.shape[1], HIDDEN, DROPOUT).to(device)
    head = mrp.MaterialHead(HIDDEN, len(edges["vocab"])).to(device)
    criterion = mrp.lp.FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)

    train_m = edges["train_manu_idx"].to(device)
    train_v = edges["train_mat_idx"].to(device)
    n_vocab = len(edges["vocab"])
    n_manu = edges["n_manu"]

    pos_by_manu_val = {}
    for m, v in zip(edges["val_manu_idx"].tolist(), edges["val_mat_idx"].tolist()):
        pos_by_manu_val.setdefault(m, set()).add(v)

    train_pos_mask = torch.zeros(n_manu, n_vocab, dtype=torch.bool, device=device)
    for m, v in zip(edges["train_manu_idx"].tolist(), edges["train_mat_idx"].tolist()):
        train_pos_mask[m, v] = True

    def sample_clean_negatives(neg_m):
        neg_v = torch.randint(0, n_vocab, (neg_m.size(0),), device=device)
        for _ in range(10):
            bad = train_pos_mask[neg_m, neg_v]
            if not bad.any():
                break
            neg_v = torch.where(bad, torch.randint(0, n_vocab, (neg_m.size(0),), device=device), neg_v)
        return neg_v

    def run_epoch_train():
        model.train(); head.train()
        optimizer.zero_grad()
        manu_h = model(feat)
        pos_logits = (manu_h[train_m] * head.mat_emb(train_v)).sum(-1)
        neg_v = sample_clean_negatives(train_m)
        neg_logits = (manu_h[train_m] * head.mat_emb(neg_v)).sum(-1)
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
        loss = criterion(logits, labels)
        loss.backward(); optimizer.step()

    def eval_recall(pos_by_manu):
        model.eval(); head.eval()
        with torch.no_grad():
            manu_h = model(feat)
            manu_idx = sorted(pos_by_manu.keys())
            all_scores = head.score_all(manu_h[torch.tensor(manu_idx, device=device)]).cpu().numpy()
        scores_by_row = {m: all_scores[i] for i, m in enumerate(manu_idx)}
        return mrp.recall_at_k(manu_idx, pos_by_manu, scores_by_row, mrp.EVAL_KS)

    best_r10, best_state_model, best_state_head, no_imp, stop_epoch = -1, None, None, 0, mrp.MAX_EPOCHS
    for epoch in range(1, mrp.MAX_EPOCHS + 1):
        run_epoch_train()
        r = eval_recall(pos_by_manu_val)
        if r[10] > best_r10:
            best_r10 = r[10]
            best_state_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_state_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= mrp.PATIENCE:
                stop_epoch = epoch
                break
    model.load_state_dict(best_state_model)
    head.load_state_dict(best_state_head)
    return model, head, best_r10, stop_epoch


def select_lr_mlp(feat, device, edges, seed):
    best = (None, -1, None, None)
    for lr in mrp.LR_GRID:
        model, head, r10, stop_ep = train_one_mlp(feat, device, edges, lr, seed)
        print(f"    seed={seed} lr={lr:.1e}  val_Recall@10={r10:.4f}  stop_epoch={stop_ep}")
        if r10 > best[1]:
            best = (lr, r10, model, head)
    return best


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edges = torch.load("material_edges.pt", map_location="cpu", weights_only=False)
    cov = torch.load("manufacturer_coverage_flags.pt", map_location="cpu", weights_only=False)
    has_image = cov["has_image"].numpy().astype(bool)
    feat = torch.load("manufacturer_features_no_material.pt", map_location=device, weights_only=False)

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
        print(f"\n=== seed={seed} condition=feature_mlp_no_graph ===")
        lr, val_r10, model, head = select_lr_mlp(feat, device, edges, seed)
        print(f"  selected lr={lr:.1e}  val_Recall@10={val_r10:.4f}")
        model.eval(); head.eval()
        with torch.no_grad():
            manu_h = model(feat)
            all_scores = head.score_all(manu_h[torch.tensor(manu_idx_all, device=device)]).cpu().numpy()
        scores_by_row = {m: all_scores[i] for i, m in enumerate(manu_idx_all)}
        entry = {"lr": lr, "val_r10": val_r10, "groups": {}}
        for g, idxs in groups.items():
            r = mrp.recall_at_k(idxs, pos_by_manu_test, scores_by_row, mrp.EVAL_KS)
            entry["groups"][g] = {"recall5": r[5], "recall10": r[10]}
            print(f"  [{g}] Recall@5={r[5]:.4f} Recall@10={r[10]:.4f}")
        all_results[seed] = entry

    print("\n=== mean +- sample std (ddof=1) across 3 seeds ===")
    summary_lines = []
    for g in groups:
        r5s = np.array([all_results[s]["groups"][g]["recall5"] for s in SEEDS])
        r10s = np.array([all_results[s]["groups"][g]["recall10"] for s in SEEDS])
        line = (f"feature_mlp_no_graph [{g}]: Recall@5={r5s.mean():.4f}+-{r5s.std(ddof=1):.4f}  "
                f"Recall@10={r10s.mean():.4f}+-{r10s.std(ddof=1):.4f}")
        print(" ", line)
        summary_lines.append(line)

    with open("material_recovery_mlp_baseline_results.md", "w", encoding="utf-8") as f:
        f.write("# Material-relation recovery: no-message-passing MLP baseline (3 seeds)\n\n")
        f.write("Same 56-category vocab, manufacturer-level split, negative-sampling-with-\n")
        f.write("exclusion rule, Recall@5/@10 eval, and val-only LR selection as\n")
        f.write("material_recovery_pilot.py / multiseed.py. The ONLY difference: the\n")
        f.write("manufacturer encoder is a plain 2-layer MLP over the 64-dim attribute\n")
        f.write("feature with NO access to the manufacturer-product graph at all -- no\n")
        f.write("message passing, no edge_index, nothing. This isolates how much of the\n")
        f.write("GNN baseline's (`baseline_no_image`) Recall@K comes from the manufacturer's\n")
        f.write("own attributes alone vs. from graph structure on top of them.\n\n")
        f.write("| seed | LR | val R@10 | group | Recall@5 | Recall@10 |\n|---|---|---|---|---|---|\n")
        for seed, entry in all_results.items():
            for g, r in entry["groups"].items():
                f.write(f"| {seed} | {entry['lr']:.1e} | {entry['val_r10']:.4f} | {g} | "
                        f"{r['recall5']:.4f} | {r['recall10']:.4f} |\n")
        f.write("\n## Mean +- sample std (ddof=1) across 3 seeds\n\n")
        for line in summary_lines:
            f.write(f"- {line}\n")

    print("\nwrote material_recovery_mlp_baseline_results.md")


if __name__ == "__main__":
    main()
