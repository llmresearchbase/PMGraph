#!/usr/bin/env python3
"""
Alternative image-fusion baseline for the material-relation recovery task:
keep the RAW 768-dim Jina-CLIP mean-pooled image embedding (no SVD reduction
to 32-dim), project text and image through INDEPENDENT linear layers, fuse
by addition (plus a learned has_image-flag embedding, kept separate from
raw content) -- no GNN/attention beyond the existing 2-layer HeteroSAGE
message passing over the manufacturer-product graph. Addresses the standing
question: was the earlier modest/mixed result caused by compressing images
to 32-dim and naively concatenating with text into a single shared
projection?

Same 56-category vocab, manufacturer-level split, corrected (true-positive-
excluding) negative sampling, and Recall@5/@10 evaluation as
material_recovery_pilot.py. Config (architecture + LR) selected on
VALIDATION only; test touched once per seed after the config is frozen.
"""
import random
import importlib.util
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv

spec = importlib.util.spec_from_file_location("lp_abcd", "LP-ABCD.py")
lp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp)
spec2 = importlib.util.spec_from_file_location("mrp", "material_recovery_pilot.py")
mrp = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(mrp)

HIDDEN, DROPOUT = 128, 0.3
MAX_EPOCHS, PATIENCE = 500, 15
LR_GRID = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
SEEDS = [42, 43, 44]


class IndependentFusionSAGE(nn.Module):
    """Text and raw-image go through SEPARATE projections and are summed,
    along with a has_image-flag embedding kept separate from image content
    (so 'knowing coverage' and 'seeing content' are distinguishable, same
    principle as the earlier concat-based ablation, just not forced through
    one shared linear layer)."""
    def __init__(self, in_text, in_image_raw, in_prod, hidden, rels, dropout, use_image):
        super().__init__()
        self.use_image = use_image
        self.text_proj = nn.Linear(in_text, hidden)
        self.text_norm = nn.LayerNorm(hidden)
        if use_image:
            self.image_proj = nn.Linear(in_image_raw, hidden)
            self.image_norm = nn.LayerNorm(hidden)
            self.flag_emb = nn.Embedding(2, hidden)
            self.flag_norm = nn.LayerNorm(hidden)
        self.prod_proj = nn.Linear(in_prod, hidden)
        self.dropout = nn.Dropout(p=dropout)
        convs = {}
        for rel in ['fwd', 'rev']:
            et = rels[rel]
            conv = SAGEConv((-1, -1), hidden)
            conv.add_self_loops = False
            convs[et] = conv
        self.conv1 = HeteroConv(convs, aggr='sum')
        self.conv2 = HeteroConv(dict(convs), aggr='sum')
        # NOTE: conv1/conv2 sharing convs dict was a bug in naive copy; build fresh
        convs2 = {}
        for rel in ['fwd', 'rev']:
            et = rels[rel]
            conv = SAGEConv((-1, -1), hidden)
            conv.add_self_loops = False
            convs2[et] = conv
        self.conv2 = HeteroConv(convs2, aggr='sum')

    @staticmethod
    def _fill(dic_new, dic_old):
        return {k: dic_new.get(k, dic_old[k]) for k in dic_old}

    def forward(self, text, image_raw, flag_idx, prod_x, edge_index_dict):
        manu_x = self.text_norm(self.text_proj(text))
        if self.use_image:
            manu_x = manu_x + self.image_norm(self.image_proj(image_raw)) + self.flag_norm(self.flag_emb(flag_idx))
        x = {'manufacturer': manu_x, 'product': self.prod_proj(prod_x)}
        h1 = self.conv1(x, edge_index_dict)
        h1 = self._fill(h1, x)
        h1 = {k: self.dropout(v.relu()) for k, v in h1.items()}
        h2 = self.conv2(h1, edge_index_dict)
        h2 = {k: self.dropout(v.relu()) for k, v in h2.items()}
        return self._fill(h2, h1)


def train_one_v2(text, image_raw, flag_idx, device, data, edges, lr, seed, use_image):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    rel = ("manufacturer", "has_product", "product")
    rels = {"fwd": rel, "rev": ("product", "rev_has_product", "manufacturer")}
    d = data.clone()
    d = lp.add_reverse_edge(d)
    in_prod = d["product"].x.shape[1]

    model = IndependentFusionSAGE(text.shape[1], image_raw.shape[1] if use_image else 0,
                                   in_prod, HIDDEN, rels, DROPOUT, use_image).to(device)
    head = mrp.MaterialHead(HIDDEN, len(edges["vocab"])).to(device)
    criterion = lp.FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)

    train_m = edges["train_manu_idx"].to(device)
    train_v = edges["train_mat_idx"].to(device)
    n_vocab = len(edges["vocab"])

    pos_by_manu_val = {}
    for m, v in zip(edges["val_manu_idx"].tolist(), edges["val_mat_idx"].tolist()):
        pos_by_manu_val.setdefault(m, set()).add(v)

    train_pos_mask = torch.zeros(edges["n_manu"], n_vocab, dtype=torch.bool, device=device)
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

    def forward_all():
        return model(text, image_raw, flag_idx, d["product"].x, d.edge_index_dict)

    def run_epoch_train():
        model.train(); head.train()
        optimizer.zero_grad()
        h = forward_all()
        manu_h = h["manufacturer"]
        pos_logits = (manu_h[train_m] * head.mat_emb(train_v)).sum(-1)
        neg_m = train_m
        neg_v = sample_clean_negatives(neg_m)
        neg_logits = (manu_h[neg_m] * head.mat_emb(neg_v)).sum(-1)
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
        loss = criterion(logits, labels)
        loss.backward(); optimizer.step()

    def eval_recall(pos_by_manu):
        model.eval(); head.eval()
        with torch.no_grad():
            h = forward_all()
            manu_h = h["manufacturer"]
            manu_idx = sorted(pos_by_manu.keys())
            all_scores = head.score_all(manu_h[torch.tensor(manu_idx, device=device)]).cpu().numpy()
        scores_by_row = {m: all_scores[i] for i, m in enumerate(manu_idx)}
        return mrp.recall_at_k(manu_idx, pos_by_manu, scores_by_row, mrp.EVAL_KS)

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
    return model, head, best_r10, stop_epoch


def select_lr_v2(text, image_raw, flag_idx, device, data, edges, seed, use_image):
    best = (None, -1, None, None)
    for lr in LR_GRID:
        model, head, r10, stop_ep = train_one_v2(text, image_raw, flag_idx, device, data, edges, lr, seed, use_image)
        print(f"    seed={seed} lr={lr:.1e}  val_Recall@10={r10:.4f}  stop_epoch={stop_ep}")
        if r10 > best[1]:
            best = (lr, r10, model, head)
    return best


def main(sanity_only=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load("hetero_graphs/Graph_A.pt", map_location=device, weights_only=False)
    edges = torch.load("material_edges.pt", map_location="cpu", weights_only=False)
    cov = torch.load("manufacturer_coverage_flags.pt", map_location="cpu", weights_only=False)
    has_image = cov["has_image"].numpy().astype(bool)

    text = torch.load("manufacturer_features_no_material.pt", map_location=device, weights_only=False)
    # RELEASE PACKAGE NOTE: loads the 768-dim raw embedding from the combined,
    # leakage-safe release asset (see material_recovery_pilot.py's equivalent note).
    img_asset = torch.load("manufacturer_image_release_asset.pt", map_location=device, weights_only=False)
    image_raw = img_asset["embedding_768dim_raw"].to(device)
    flag_idx = cov["has_image"].long().to(device)

    pos_by_manu_test = {}
    for m, v in zip(edges["test_manu_idx"].tolist(), edges["test_mat_idx"].tolist()):
        pos_by_manu_test.setdefault(m, set()).add(v)
    manu_idx_all = sorted(pos_by_manu_test.keys())
    groups = {
        "all": manu_idx_all,
        "has_image": [m for m in manu_idx_all if has_image[m]],
        "no_image": [m for m in manu_idx_all if not has_image[m]],
    }

    conditions = {
        "text_only": False,
        "real_image": True,
        "zeroed_image": True,  # architecture identical to real_image; content zeroed below
    }

    seeds = [42] if sanity_only else SEEDS
    all_results = {}
    for seed in seeds:
        for cond_name, use_image in conditions.items():
            img_input = image_raw if cond_name != "zeroed_image" else torch.zeros_like(image_raw)
            print(f"\n=== seed={seed} condition={cond_name} ===")
            lr, val_r10, model, head = select_lr_v2(text, img_input, flag_idx, device, data, edges, seed, use_image)
            print(f"  selected lr={lr}  val_Recall@10={val_r10:.4f}")
            model.eval(); head.eval()
            with torch.no_grad():
                d = lp.add_reverse_edge(data.clone())
                h = model(text, img_input, flag_idx, d["product"].x, d.edge_index_dict)
                manu_h = h["manufacturer"]
                all_scores = head.score_all(manu_h[torch.tensor(manu_idx_all, device=device)]).cpu().numpy()
            scores_by_row = {m: all_scores[i] for i, m in enumerate(manu_idx_all)}
            entry = {"lr": lr, "val_r10": val_r10, "groups": {}}
            for g, idxs in groups.items():
                r = mrp.recall_at_k(idxs, pos_by_manu_test, scores_by_row, mrp.EVAL_KS)
                entry["groups"][g] = {"recall5": r[5], "recall10": r[10]}
                print(f"  [{g}] Recall@5={r[5]:.4f} Recall@10={r[10]:.4f}")
            all_results[(seed, cond_name)] = entry
        if sanity_only:
            print("\nSANITY CHECK PASSED (ran end to end for seed=42, all 3 conditions).")
            return

    # ---- paired diffs across seeds ----
    print("\n=== mean +- sample std across seeds ===")
    summary_lines = []
    for comp_name, (a, b) in [("real_minus_text", ("real_image", "text_only")),
                               ("real_minus_zeroed", ("real_image", "zeroed_image"))]:
        for g in groups:
            r5s, r10s = [], []
            for seed in seeds:
                r5s.append(all_results[(seed, a)]["groups"][g]["recall5"] - all_results[(seed, b)]["groups"][g]["recall5"])
                r10s.append(all_results[(seed, a)]["groups"][g]["recall10"] - all_results[(seed, b)]["groups"][g]["recall10"])
            r5s, r10s = np.array(r5s), np.array(r10s)
            std5 = r5s.std(ddof=1) if len(r5s) > 1 else float('nan')
            std10 = r10s.std(ddof=1) if len(r10s) > 1 else float('nan')
            line = f"{comp_name} [{g}]: dR5={r5s.mean():+.4f}+-{std5:.4f}  dR10={r10s.mean():+.4f}+-{std10:.4f}"
            print(" ", line)
            summary_lines.append(line)

    with open("material_recovery_v2fusion_results.md", "w", encoding="utf-8") as f:
        f.write("# Material recovery: independent-projection image-fusion baseline (raw 768-dim, 3 seeds)\n\n")
        f.write("| seed | condition | LR | val R@10 | group | Recall@5 | Recall@10 |\n|---|---|---|---|---|---|---|\n")
        for (seed, cond), entry in all_results.items():
            for g, r in entry["groups"].items():
                f.write(f"| {seed} | {cond} | {entry['lr']:.1e} | {entry['val_r10']:.4f} | {g} | "
                        f"{r['recall5']:.4f} | {r['recall10']:.4f} |\n")
        f.write("\n## Paired differences, mean +- sample std across seeds\n\n")
        for line in summary_lines:
            f.write(f"- {line}\n")
    print("\nwrote material_recovery_v2fusion_results.md")


if __name__ == "__main__":
    import sys
    main(sanity_only=("--sanity" in sys.argv))
