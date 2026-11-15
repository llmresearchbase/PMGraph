#!/usr/bin/env python3
"""
Recorded-material-relation recovery pilot.

Graph structure (manufacturer-product edges, ALL of them -- this is input
context, not a prediction target here) is fixed and given in full to a
HeteroSAGE encoder. The manufacturer-material LABELS are what's split:
train manufacturers' positive (manufacturer, material) pairs are the only
supervision; val/test manufacturers' material labels are held out entirely
and used only for Recall@K evaluation.

Negative sampling for training = materials NOT recorded for a manufacturer,
within the frozen 56-category vocab. These are PSEUDO-negatives (per the
text-evidence check, ~17-25% of "unrecorded" are plausibly real capabilities
the directory just didn't capture) -- NOT asserted as confirmed absence of
capability. This script does not claim to solve true capability
classification; it measures recovery of DIRECTORY-RECORDED relations only.

Reports Recall@5 / Recall@10 on TEST manufacturers for three conditions:
baseline (material-excluded text/graph features, no image slot at all),
REAL (+ has_image flag + real image embedding), ZEROED (+ same flag, image
slot forced to zero) -- same architecture-equivalence design as the
has-image/no-image ablation. has_image / all / no_image groups reported
separately for the two image-aware conditions.
"""
import random
import importlib.util
import numpy as np
import torch
import torch.nn as nn

spec = importlib.util.spec_from_file_location("lp_abcd", "LP-ABCD.py")
lp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp)

SEED = 42
HIDDEN, DROPOUT = 128, 0.3
MAX_EPOCHS, PATIENCE = 500, 15
LR_GRID = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
N_NEG_PER_POS = 1
EVAL_KS = (5, 10)


class MaterialHead(nn.Module):
    def __init__(self, hidden, n_vocab):
        super().__init__()
        self.mat_emb = nn.Embedding(n_vocab, hidden)

    def score_all(self, manu_h):
        # manu_h: [N_eval, hidden] -> [N_eval, n_vocab]
        return manu_h @ self.mat_emb.weight.t()


def recall_at_k(manu_idx_unique, pos_by_manu, scores_by_manu_row, ks):
    """pos_by_manu: manu_idx -> set of material indices (ground truth).
    scores_by_manu_row: dict manu_idx -> score vector [n_vocab]."""
    hits = {k: [] for k in ks}
    for m in manu_idx_unique:
        pos = pos_by_manu.get(m)
        if not pos:
            continue
        scores = scores_by_manu_row[m]
        order = np.argsort(-scores)
        for k in ks:
            topk = set(order[:k].tolist())
            hit = len(topk & pos) / len(pos)
            hits[k].append(hit)
    return {k: float(np.mean(v)) if v else float('nan') for k, v in hits.items()}


def train_one(feat, device, data, edges, lr, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    rel = ("manufacturer", "has_product", "product")
    rels = {"fwd": rel, "rev": ("product", "rev_has_product", "manufacturer")}
    d = data.clone()
    d["manufacturer"].x = feat.to(device)
    d = lp.add_reverse_edge(d)

    in_manuf, in_prod = feat.shape[1], d["product"].x.shape[1]
    model = lp.HeteroSAGE(in_manuf, in_prod, HIDDEN, rels, DROPOUT).to(device)
    head = MaterialHead(HIDDEN, len(edges["vocab"])).to(device)
    criterion = lp.FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)

    train_m = edges["train_manu_idx"].to(device)
    train_v = edges["train_mat_idx"].to(device)
    n_vocab = len(edges["vocab"])

    # pos set per manufacturer, for negative sampling exclusion and eval
    pos_by_manu_train = {}
    for m, v in zip(edges["train_manu_idx"].tolist(), edges["train_mat_idx"].tolist()):
        pos_by_manu_train.setdefault(m, set()).add(v)

    pos_by_manu_val = {}
    for m, v in zip(edges["val_manu_idx"].tolist(), edges["val_mat_idx"].tolist()):
        pos_by_manu_val.setdefault(m, set()).add(v)

    # dense boolean positive-mask for rejection sampling (bug fix: previously
    # negatives were sampled uniformly over the vocab WITHOUT excluding a
    # manufacturer's own true positives -- an avoidable training-label error,
    # not a modeling choice)
    train_pos_mask = torch.zeros(edges["n_manu"], n_vocab, dtype=torch.bool, device=device)
    for m, v in zip(edges["train_manu_idx"].tolist(), edges["train_mat_idx"].tolist()):
        train_pos_mask[m, v] = True

    def sample_clean_negatives(neg_m):
        neg_v = torch.randint(0, n_vocab, (neg_m.size(0),), device=device)
        for _ in range(10):  # a handful of rejection-resample rounds; vocab is tiny (56)
            bad = train_pos_mask[neg_m, neg_v]
            if not bad.any():
                break
            neg_v = torch.where(bad, torch.randint(0, n_vocab, (neg_m.size(0),), device=device), neg_v)
        return neg_v

    def run_epoch_train():
        model.train(); head.train()
        optimizer.zero_grad()
        h = model(d.x_dict, d.edge_index_dict)
        manu_h = h["manufacturer"]
        pos_logits = (manu_h[train_m] * head.mat_emb(train_v)).sum(-1)
        neg_m = train_m.repeat_interleave(N_NEG_PER_POS)
        neg_v = sample_clean_negatives(neg_m)
        neg_logits = (manu_h[neg_m] * head.mat_emb(neg_v)).sum(-1)
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
        loss = criterion(logits, labels)
        loss.backward(); optimizer.step()
        return loss.item()

    def eval_recall(pos_by_manu):
        model.eval(); head.eval()
        with torch.no_grad():
            h = model(d.x_dict, d.edge_index_dict)
            manu_h = h["manufacturer"]
            manu_idx = sorted(pos_by_manu.keys())
            all_scores = head.score_all(manu_h[torch.tensor(manu_idx, device=device)]).cpu().numpy()
        scores_by_manu_row = {m: all_scores[i] for i, m in enumerate(manu_idx)}
        return recall_at_k(manu_idx, pos_by_manu, scores_by_manu_row, EVAL_KS)

    best_r10, best_state_model, best_state_head, no_imp, stop_epoch = -1, None, None, 0, MAX_EPOCHS
    for epoch in range(1, MAX_EPOCHS + 1):
        run_epoch_train()
        r = eval_recall(pos_by_manu_val)
        if r[10] > best_r10:
            best_r10 = r[10]
            best_state_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_state_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                stop_epoch = epoch
                break
    model.load_state_dict(best_state_model)
    head.load_state_dict(best_state_head)
    return model, head, d, best_r10, stop_epoch


def select_lr(feat, device, data, edges):
    best = (None, -1, None, None, None)
    for lr in LR_GRID:
        model, head, d, r10, stop_ep = train_one(feat, device, data, edges, lr, SEED)
        print(f"    lr={lr:.1e}  val_Recall@10={r10:.4f}  stop_epoch={stop_ep}")
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
    # RELEASE PACKAGE NOTE: the original dev script loaded "manufacturer_features_real.pt"
    # here and sliced out columns [65:97] -- that 97-dim file's first 64 columns contain
    # the Material_Capability one-hot and are NOT included in this release. The clean
    # replacement asset below carries ONLY the verified-safe 32-dim image embedding.
    img_asset = torch.load("manufacturer_image_release_asset.pt", map_location=device, weights_only=False)
    img_part_real = img_asset["embedding_32dim_svd"].to(device)

    has_image_t = cov["has_image"].to(device).unsqueeze(1)
    feat_baseline = no_mat_feat  # 64-dim, no material, no image at all
    feat_real = torch.cat([no_mat_feat, has_image_t, img_part_real], dim=1)
    feat_zeroed = torch.cat([no_mat_feat, has_image_t, torch.zeros_like(img_part_real)], dim=1)

    conditions = {"baseline_no_image": feat_baseline, "real": feat_real, "zeroed": feat_zeroed}
    results = {}

    # ---- non-personalized frequency baseline: same score vector (train-set
    # popularity of each material) for every test manufacturer -- contextualizes
    # what Recall@10~0.7 over 56 categories actually means ----
    n_vocab = len(edges["vocab"])
    freq = np.zeros(n_vocab)
    for v in edges["train_mat_idx"].tolist():
        freq[v] += 1
    pos_by_manu_test = {}
    for m, v in zip(edges["test_manu_idx"].tolist(), edges["test_mat_idx"].tolist()):
        pos_by_manu_test.setdefault(m, set()).add(v)
    manu_idx_all_freq = sorted(pos_by_manu_test.keys())
    scores_by_manu_row_freq = {m: freq for m in manu_idx_all_freq}
    r_freq_all = recall_at_k(manu_idx_all_freq, pos_by_manu_test, scores_by_manu_row_freq, EVAL_KS)
    print(f"\n=== frequency baseline (train-set popularity, non-personalized) ===")
    print(f"  [all] n_test_manufacturers={len(manu_idx_all_freq)}  "
          f"Recall@5={r_freq_all[5]:.4f}  Recall@10={r_freq_all[10]:.4f}")
    results["frequency_baseline"] = {"lr": None, "val_recall10": None,
                                      "groups": {"all": {"n": len(manu_idx_all_freq),
                                                          "recall5": r_freq_all[5], "recall10": r_freq_all[10]}}}

    for cond_name, feat in conditions.items():
        print(f"\n=== condition: {cond_name} ===")
        lr, val_r10, model, head, d = select_lr(feat, device, data, edges)
        print(f"  selected lr={lr:.1e}  val_Recall@10={val_r10:.4f}")

        pos_by_manu_test = {}
        for m, v in zip(edges["test_manu_idx"].tolist(), edges["test_mat_idx"].tolist()):
            pos_by_manu_test.setdefault(m, set()).add(v)

        model.eval(); head.eval()
        with torch.no_grad():
            h = model(d.x_dict, d.edge_index_dict)
            manu_h = h["manufacturer"]
            manu_idx_all = sorted(pos_by_manu_test.keys())
            all_scores = head.score_all(manu_h[torch.tensor(manu_idx_all, device=device)]).cpu().numpy()
        scores_by_manu_row = {m: all_scores[i] for i, m in enumerate(manu_idx_all)}

        groups = {
            "all": manu_idx_all,
            "has_image": [m for m in manu_idx_all if has_image[m]],
            "no_image": [m for m in manu_idx_all if not has_image[m]],
        }
        results[cond_name] = {"lr": lr, "val_recall10": val_r10, "groups": {}}
        for g, idxs in groups.items():
            r = recall_at_k(idxs, pos_by_manu_test, scores_by_manu_row, EVAL_KS)
            n_manu = len(idxs)
            print(f"  [{g}] n_test_manufacturers={n_manu}  Recall@5={r[5]:.4f}  Recall@10={r[10]:.4f}")
            results[cond_name]["groups"][g] = {"n": n_manu, "recall5": r[5], "recall10": r[10]}

    with open("material_recovery_results.md", "w", encoding="utf-8") as f:
        f.write("# Recorded-material-relation recovery pilot (single seed=42)\n\n")
        f.write("Negatives during training are PSEUDO-negatives (unrecorded, not confirmed absent). "
                "Recall@K measures recovery of DIRECTORY-RECORDED materials only.\n\n")
        f.write("Negative sampling (fixed this run): rejection-sampled to exclude a manufacturer's own "
                "known positives from training negatives.\n\n")
        f.write("| Condition | selected LR | val Recall@10 | group | n_test | Recall@5 | Recall@10 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for cond, r in results.items():
            lr_str = f"{r['lr']:.1e}" if r['lr'] is not None else "n/a"
            vr_str = f"{r['val_recall10']:.4f}" if r['val_recall10'] is not None else "n/a"
            for g, gr in r["groups"].items():
                f.write(f"| {cond} | {lr_str} | {vr_str} | {g} | {gr['n']} | "
                        f"{gr['recall5']:.4f} | {gr['recall10']:.4f} |\n")
    print("\nwrote material_recovery_results.md")


if __name__ == "__main__":
    main()
