#!/usr/bin/env python3
"""
Target-relevance-weighted image aggregation vs. the current flat (uniform-
mean) image aggregation, for the no-graph MLP track (the strongest existing
baseline, per the user's explicit instruction to start from it). ONE
comparison, nothing else -- no new fusion architecture, no graph model
swap, no dimension sweep beyond picking ONE softmax temperature on the
validation set before touching test.

Design: for a manufacturer with k website images, the existing pipeline
uniformly mean-pools all k into one 768-dim vector before SVD-reducing to
32-dim -- the same vector is used regardless of which material category is
being scored. This script instead, for EACH of the 56 candidate materials,
re-weights the manufacturer's own k images by their frozen Jina-CLIP-v1
image-text cosine similarity to that candidate's own name (zero-shot,
same model, same joint space, NO new training of the weighting itself) --
so a manufacturer's effective image representation differs depending on
which material is being asked about. Everything else (64-dim attrs,
has_image flag, FeatureMLP architecture, FocalLoss, negative-sampling-with-
exclusion, 56-cat vocab, manufacturer split, val-only LR selection) is
IDENTICAL to material_recovery_mlp_image_baseline.py's real-image condition,
so this isolates the one changed variable: how images are aggregated, not
what information is available in principle.

No-image manufacturers are mathematically unaffected (their image slice is
zero regardless of candidate, same as the flat baseline) -- compute is only
spent on the has-image subset at both train and eval time.

Protocol: temperature selected on VALIDATION Recall@10 (seed 42 only, a
small fixed candidate set -- not an open grid search), frozen; then 3 seeds
(42/43/44) run to completion on the frozen config, LR re-selected per seed
from the same 5-value grid as always (never re-touched after seeing test).
"""
import importlib.util
import time

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("mb", "material_recovery_mlp_baseline.py")
mb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mb)

spec2 = importlib.util.spec_from_file_location("bw", "build_target_weighted_image_features.py")
bw = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(bw)

SEEDS = [42, 43, 44]
TEMPERATURE_CANDIDATES = [1.0, 0.1, 0.05]
HIDDEN, DROPOUT = mb.HIDDEN, mb.DROPOUT


def load_common(device):
    edges = torch.load("material_edges.pt", map_location="cpu", weights_only=False)
    cov = torch.load("manufacturer_coverage_flags.pt", map_location="cpu", weights_only=False)
    has_image = cov["has_image"].to(device)
    attrs64 = torch.load("manufacturer_features_no_material.pt", map_location=device, weights_only=False)
    return edges, has_image, attrs64


def build_image_cond_tensor(temperature, device, verify=True):
    feats, manus_sorted, vocab = bw.build(temperature, verify=verify)
    return feats.to(device)  # [n_manu, 56, 32]


class TargetedModel:
    """Wraps FeatureMLP + MaterialHead with per-(manufacturer,candidate)
    input construction instead of a single precomputed manu_h matrix."""

    def __init__(self, attrs64, has_image, image_cond, device, n_vocab):
        self.attrs64 = attrs64
        self.has_image_col = has_image.float().unsqueeze(1)
        self.image_cond = image_cond  # [n_manu, n_vocab, 32]
        self.device = device
        self.n_vocab = n_vocab

    def pair_input(self, m_idx, v_idx):
        # m_idx, v_idx: 1D LongTensor, same length -> [batch, 97]
        return torch.cat([
            self.attrs64[m_idx], self.has_image_col[m_idx], self.image_cond[m_idx, v_idx]
        ], dim=-1)

    def all_candidates_input(self, m_idx):
        # m_idx: 1D LongTensor [k] -> [k, n_vocab, 97]
        k = m_idx.shape[0]
        attrs_rep = self.attrs64[m_idx].unsqueeze(1).expand(-1, self.n_vocab, -1)
        has_rep = self.has_image_col[m_idx].unsqueeze(1).expand(-1, self.n_vocab, -1)
        img = self.image_cond[m_idx]  # [k, n_vocab, 32]
        return torch.cat([attrs_rep, has_rep, img], dim=-1)


def train_one_targeted(tm, device, edges, lr, seed, in_dim):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    model = mb.FeatureMLP(in_dim, HIDDEN, DROPOUT).to(device)
    head = mb.mrp.MaterialHead(HIDDEN, len(edges["vocab"])).to(device)
    criterion = mb.mrp.lp.FocalLoss(alpha=0.25, gamma=2.0)
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
        pos_input = tm.pair_input(train_m, train_v)
        pos_h = model(pos_input)
        pos_logits = (pos_h * head.mat_emb(train_v)).sum(-1)
        neg_v = sample_clean_negatives(train_m)
        neg_input = tm.pair_input(train_m, neg_v)
        neg_h = model(neg_input)
        neg_logits = (neg_h * head.mat_emb(neg_v)).sum(-1)
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
        loss = criterion(logits, labels)
        loss.backward(); optimizer.step()

    def eval_recall(pos_by_manu):
        model.eval(); head.eval()
        with torch.no_grad():
            manu_idx = sorted(pos_by_manu.keys())
            m_t = torch.tensor(manu_idx, device=device)
            batch_in = tm.all_candidates_input(m_t)  # [k, n_vocab, in_dim]
            k = batch_in.shape[0]
            h = model(batch_in.reshape(-1, in_dim)).reshape(k, n_vocab, HIDDEN)
            scores = (h * head.mat_emb.weight.unsqueeze(0)).sum(-1).cpu().numpy()  # [k, n_vocab]
        scores_by_row = {m: scores[i] for i, m in enumerate(manu_idx)}
        return mb.mrp.recall_at_k(manu_idx, pos_by_manu, scores_by_row, mb.mrp.EVAL_KS)

    best_r10, best_state_model, best_state_head, no_imp, stop_epoch = -1, None, None, 0, mb.mrp.MAX_EPOCHS
    for epoch in range(1, mb.mrp.MAX_EPOCHS + 1):
        run_epoch_train()
        r = eval_recall(pos_by_manu_val)
        if r[10] > best_r10:
            best_r10 = r[10]
            best_state_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_state_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= mb.mrp.PATIENCE:
                stop_epoch = epoch
                break
    model.load_state_dict(best_state_model)
    head.load_state_dict(best_state_head)
    return model, head, best_r10, stop_epoch


def select_lr_targeted(tm, device, edges, seed, in_dim):
    best = (None, -1, None, None)
    for lr in mb.mrp.LR_GRID:
        model, head, r10, stop_ep = train_one_targeted(tm, device, edges, lr, seed, in_dim)
        print(f"    seed={seed} lr={lr:.1e}  val_Recall@10={r10:.4f}  stop_epoch={stop_ep}")
        if r10 > best[1]:
            best = (lr, r10, model, head)
    return best


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edges, has_image, attrs64 = load_common(device)
    in_dim = attrs64.shape[1] + 1 + 32

    # ---- Phase 1: temperature selection, validation set, seed 42 only ----
    print("=== Phase 1: temperature selection (val Recall@10, seed=42 only) ===")
    t0 = time.time()
    temp_results = {}
    for temp in TEMPERATURE_CANDIDATES:
        print(f"\n-- temperature={temp} --")
        image_cond = build_image_cond_tensor(temp, device, verify=(temp == TEMPERATURE_CANDIDATES[0]))
        tm = TargetedModel(attrs64, has_image, image_cond, device, len(edges["vocab"]))
        lr, val_r10, model, head = select_lr_targeted(tm, device, edges, 42, in_dim)
        temp_results[temp] = val_r10
        print(f"  temperature={temp}: best val_Recall@10={val_r10:.4f} (lr={lr:.1e})")
    t1 = time.time()
    print(f"\nPhase 1 wall-clock: {t1-t0:.1f}s")

    best_temp = max(temp_results, key=temp_results.get)
    print(f"\nSelected temperature={best_temp} (val_Recall@10={temp_results[best_temp]:.4f}) -- FROZEN, "
          f"not re-touched after this point regardless of test outcome.")

    # compare to flat baseline's own val_Recall@10 for context (not used to pick temperature itself)
    # NOTE: manufacturer_features_real.pt's first 64 columns are NOT the clean attrs -- they are
    # the original dev tensor whose first 64 dims contain the Material_Capability one-hot (disclosed
    # leakage-adjacent detail, paper Section 4.2). Must reconstruct the clean 97-dim feature the
    # same way material_recovery_mlp_image_baseline.py does (clean attrs + has_image + img slice
    # only) -- an earlier version of this script fed the raw file directly, which leaked the
    # target label into this "context" number (produced an impossible val_Recall@10 ~0.93, caught
    # and fixed; this context value is informational only and does not affect temperature
    # selection or any reported result).
    raw_img_real = torch.load("manufacturer_features_real.pt", map_location=device, weights_only=False)
    flat_feat = torch.cat([attrs64, has_image.float().unsqueeze(1), raw_img_real[:, 65:]], dim=1)
    _, flat_val_r10, _, _ = mb.select_lr_mlp(flat_feat, device, edges, 42)
    print(f"(context, not used for selection) flat-pooled baseline val_Recall@10, seed=42: {flat_val_r10:.4f}")

    # ---- Phase 2: frozen config, 3 seeds, test touched once per seed ----
    print(f"\n=== Phase 2: frozen temperature={best_temp}, 3 seeds, test evaluation ===")
    image_cond = build_image_cond_tensor(best_temp, device, verify=True)
    tm = TargetedModel(attrs64, has_image, image_cond, device, len(edges["vocab"]))

    pos_by_manu_test = {}
    for m, v in zip(edges["test_manu_idx"].tolist(), edges["test_mat_idx"].tolist()):
        pos_by_manu_test.setdefault(m, set()).add(v)
    manu_idx_all = sorted(pos_by_manu_test.keys())
    has_image_np = has_image.cpu().numpy().astype(bool)
    groups = {
        "all": manu_idx_all,
        "has_image": [m for m in manu_idx_all if has_image_np[m]],
        "no_image": [m for m in manu_idx_all if not has_image_np[m]],
    }

    all_results = {}
    t2 = time.time()
    for seed in SEEDS:
        print(f"\n--- seed={seed} ---")
        s0 = time.time()
        lr, val_r10, model, head = select_lr_targeted(tm, device, edges, seed, in_dim)
        s1 = time.time()
        print(f"  selected lr={lr:.1e}  val_Recall@10={val_r10:.4f}  (train+select wall-clock: {s1-s0:.1f}s)")
        model.eval(); head.eval()
        with torch.no_grad():
            m_t = torch.tensor(manu_idx_all, device=device)
            batch_in = tm.all_candidates_input(m_t)
            k = batch_in.shape[0]
            h = model(batch_in.reshape(-1, in_dim)).reshape(k, len(edges["vocab"]), HIDDEN)
            scores = (h * head.mat_emb.weight.unsqueeze(0)).sum(-1).cpu().numpy()
        scores_by_row = {m: scores[i] for i, m in enumerate(manu_idx_all)}
        entry = {"lr": lr, "val_r10": val_r10, "groups": {}, "train_eval_seconds": s1 - s0}
        for g, idxs in groups.items():
            r = mb.mrp.recall_at_k(idxs, pos_by_manu_test, scores_by_row, mb.mrp.EVAL_KS)
            entry["groups"][g] = {"recall5": r[5], "recall10": r[10]}
            print(f"  [{g}] Recall@5={r[5]:.4f} Recall@10={r[10]:.4f}")
        all_results[seed] = entry
    t3 = time.time()
    print(f"\nPhase 2 total wall-clock (3 seeds, LR grid x each): {t3-t2:.1f}s")

    print("\n=== mean +- sample std (ddof=1), has_image group ===")
    r5s = np.array([all_results[s]["groups"]["has_image"]["recall5"] for s in SEEDS])
    r10s = np.array([all_results[s]["groups"]["has_image"]["recall10"] for s in SEEDS])
    print(f"  target_weighted_image [has_image]: Recall@5={r5s.mean():.4f}+-{r5s.std(ddof=1):.4f}  "
          f"Recall@10={r10s.mean():.4f}+-{r10s.std(ddof=1):.4f}")

    # already-archived flat-pooled baseline (has_image group), for direct comparison
    FLAT_R5 = np.array([0.5456, 0.5399, 0.5400])  # from material_recovery_mlp_image_baseline_results.md, mlp_real_image, has_image group, seeds 42/43/44
    FLAT_R10 = np.array([0.7380, 0.7334, 0.7333])
    diff_r5 = r5s - FLAT_R5
    diff_r10 = r10s - FLAT_R10
    print(f"\n  target_weighted MINUS flat-pooled, has_image, per seed:")
    for i, s in enumerate(SEEDS):
        print(f"    seed={s}: dR5={diff_r5[i]:+.4f}  dR10={diff_r10[i]:+.4f}")
    print(f"  mean+-std: dR5={diff_r5.mean():+.4f}+-{diff_r5.std(ddof=1):.4f}  "
          f"dR10={diff_r10.mean():+.4f}+-{diff_r10.std(ddof=1):.4f}")

    with open("material_recovery_mlp_target_weighted_results.md", "w", encoding="utf-8") as f:
        f.write("# Target-relevance-weighted image aggregation vs. flat mean-pooling (no-graph MLP)\n\n")
        f.write("Zero-shot Jina-CLIP-v1 image-text cosine-similarity softmax weighting per "
                "candidate material, vs. the existing uniform mean-pool. Same attrs/vocab/split/"
                "negative-sampling/eval/FeatureMLP/FocalLoss as the flat-pooled real-image MLP "
                "baseline -- only the image aggregation changes.\n\n")
        f.write(f"## Phase 1: temperature selection (val Recall@10, seed=42 only)\n\n")
        f.write("| temperature | val Recall@10 |\n|---|---|\n")
        for temp, v in temp_results.items():
            f.write(f"| {temp} | {v:.4f} |\n")
        f.write(f"\nSelected (frozen before test): temperature={best_temp}\n")
        f.write(f"(context) flat-pooled baseline val_Recall@10, seed=42: {flat_val_r10:.4f}\n\n")
        f.write("## Phase 2: frozen config, 3-seed test results\n\n")
        f.write("| seed | LR | val R@10 | group | Recall@5 | Recall@10 | train+select wall-clock (s) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for seed, entry in all_results.items():
            for g, r in entry["groups"].items():
                f.write(f"| {seed} | {entry['lr']:.1e} | {entry['val_r10']:.4f} | {g} | "
                        f"{r['recall5']:.4f} | {r['recall10']:.4f} | {entry['train_eval_seconds']:.1f} |\n")
        f.write(f"\n## Mean +- sample std (ddof=1), has_image group\n\n")
        f.write(f"- target_weighted_image: Recall@5={r5s.mean():.4f}+-{r5s.std(ddof=1):.4f}  "
                f"Recall@10={r10s.mean():.4f}+-{r10s.std(ddof=1):.4f}\n")
        f.write(f"- flat-pooled (archived): Recall@5={FLAT_R5.mean():.4f}  Recall@10={FLAT_R10.mean():.4f}\n")
        f.write(f"- target_weighted MINUS flat-pooled: dR5={diff_r5.mean():+.4f}+-{diff_r5.std(ddof=1):.4f}  "
                f"dR10={diff_r10.mean():+.4f}+-{diff_r10.std(ddof=1):.4f}\n")
        f.write(f"\nPhase 1 wall-clock: {t1-t0:.1f}s. Phase 2 wall-clock (3 seeds incl. LR grid): {t3-t2:.1f}s.\n")

    print("\nwrote material_recovery_mlp_target_weighted_results.md")


if __name__ == "__main__":
    main()
