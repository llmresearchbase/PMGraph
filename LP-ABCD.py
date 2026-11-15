
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, HGTConv
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_recall_fscore_support
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
import itertools
import numpy as np


# Supported architectures
MODEL_CLASSES = ['sage', 'gat', 'hgt']


def add_reverse_edge(data: HeteroData) -> HeteroData:
    fwd = ('manufacturer', 'has_product', 'product')
    rev = ('product', 'rev_has_product', 'manufacturer')
    src, dst = data[fwd].edge_index
    data[rev].edge_index = torch.stack([dst, src], dim=0)
    return data

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce)                       # p_t = sigmoid(logits) if target=1
        loss = self.alpha * (1 - p_t)**self.gamma * bce
        if self.reduction=='mean':
            return loss.mean()
        elif self.reduction=='sum':
            return loss.sum()
        else:
            return loss

class HeteroSAGE(nn.Module):
    def __init__(self, in_manuf, in_prod, hidden, rels, dropout):
        super().__init__()
        self.man_proj = nn.Linear(in_manuf, hidden)
        self.prod_proj = nn.Linear(in_prod, hidden)
        self.dropout = nn.Dropout(p=dropout)
        convs = {}
        for rel in ['fwd', 'rev']:
            et = rels[rel]
            conv = SAGEConv((-1, -1), hidden)
            conv.add_self_loops = False
            convs[et] = conv
        self.conv1 = HeteroConv(convs, aggr='sum')
        self.conv2 = HeteroConv(convs, aggr='sum')

    @staticmethod
    def _fill(dic_new, dic_old):
        return {k: dic_new.get(k, dic_old[k]) for k in dic_old}

    def forward(self, x_dict, edge_index_dict):
        x = {
            'manufacturer': self.man_proj(x_dict['manufacturer']),
            'product':      self.prod_proj(x_dict['product'])
        }
        h1 = self.conv1(x, edge_index_dict)
        h1 = self._fill(h1, x)
        h1 = {k: self.dropout(v.relu()) for k, v in h1.items()}
        h2 = self.conv2(h1, edge_index_dict)
        h2 = {k: self.dropout(v.relu()) for k, v in h2.items()}
        return self._fill(h2, h1)


class HeteroGAT(nn.Module):
    """
    Two-layer heterogeneous GAT with residual + dropout.
    heads : number of attention heads per relation
    dropout : applied to node features after each layer
    """
    def __init__(self, in_manuf, in_prod, hidden,
                 rels, heads: int = 4, dropout: float = 0.5):
        super().__init__()

        self.man_proj  = nn.Linear(in_manuf, hidden)
        self.prod_proj = nn.Linear(in_prod,  hidden)
        self.dropout   = nn.Dropout(p=dropout)


        def make_conv():
            convs = {}
            for rel in ['fwd', 'rev']:
                et = rels[rel]
                convs[et] = GATConv(
                    (-1, -1), hidden,
                    heads=heads,
                    dropout=dropout,
                    concat=False          # keep out_dim == hidden
                )
                convs[et].add_self_loops = False
            return HeteroConv(convs, aggr='sum')

        self.conv1 = make_conv()
        self.conv2 = make_conv()

    def forward(self, x_dict, edge_index_dict):
        # initial projections
        x = {
            'manufacturer': self.man_proj(x_dict['manufacturer']),
            'product':      self.prod_proj(x_dict['product'])
        }

        # layer 1 + residual
        h1 = self.conv1(x, edge_index_dict)
        h1 = {k: self.dropout(F.relu(h1[k] + x[k])) for k in h1}

        # layer 2 + residual
        h2 = self.conv2(h1, edge_index_dict)
        h2 = {k: self.dropout(F.relu(h2[k] + h1[k])) for k in h2}
        return h2



import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv

class HeteroHGT(nn.Module):
    def __init__(self, in_manuf, in_prod, hidden, rels, metadata, dropout):
        super().__init__()
        # we ignore `rels` here, but accept it so instantiation matches
        self.man_proj  = nn.Linear(in_manuf, hidden)
        self.prod_proj = nn.Linear(in_prod,  hidden)

        self.hgt1 = HGTConv(hidden, hidden, metadata=metadata, heads=4)
        self.norm1 = nn.LayerNorm(hidden)
        self.hgt2 = HGTConv(hidden, hidden, metadata=metadata, heads=4)
        self.norm2 = nn.LayerNorm(hidden)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        x = {
            'manufacturer': self.man_proj(x_dict['manufacturer']),
            'product':      self.prod_proj(x_dict['product'])
        }

        h1 = self.hgt1(x, edge_index_dict)
        h1 = {
            k: self.norm1(self.dropout(F.relu(h1[k] + x[k])))
            for k in h1
        }

        h2 = self.hgt2(h1, edge_index_dict)
        h2 = {
            k: self.norm2(self.dropout(F.relu(h2[k] + h1[k])))
            for k in h2
        }

        return h2


def compute_mrr(edge_index, labels, probs):
    """
    Mean Reciprocal Rank on a head-based ranking task.

    Parameters
    ----------
    edge_index : LongTensor shape [2, N]
        (head, tail) indices of the *test* edges.
    labels : 1-D tensor size N of 0/1 ground-truth.
    probs  : 1-D tensor size N of predicted probabilities.
    """
    heads = edge_index[0].cpu().numpy()
    labs  = labels.cpu().numpy()
    prbs  = probs.cpu().numpy()
    mrrs  = []
    for h in np.unique(heads):
        mask   = (heads == h)
        scores = prbs[mask]
        labs_h = labs[mask]
        order  = np.argsort(-scores)          # descending
        pos    = np.where(labs_h[order] == 1)[0]
        if pos.size:
            mrrs.append(1.0 / (pos[0] + 1))
    return float(np.mean(mrrs)) if mrrs else 0.0



    
def train_step(data_split, model, optimizer, criterion, device):
    model.train(); optimizer.zero_grad()
    h = model(data_split.x_dict, data_split.edge_index_dict)
    rel = ('manufacturer', 'has_product', 'product')
    idx = data_split[rel].edge_label_index
    labels = data_split[rel].edge_label.float()
    logits = (h['manufacturer'][idx[0]] * h['product'][idx[1]]).sum(dim=-1)
    loss = criterion(logits, labels); loss.backward(); optimizer.step()
    return loss.item()


def eval_step(data_split, model, criterion, device):
    model.eval()
    with torch.no_grad():
        h = model(data_split.x_dict, data_split.edge_index_dict)
        rel = ('manufacturer', 'has_product', 'product')
        idx = data_split[rel].edge_label_index
        labels = data_split[rel].edge_label.float()
        logits = (h['manufacturer'][idx[0]] * h['product'][idx[1]]).sum(dim=-1)
        loss = float(criterion(logits, labels))
        probs = torch.sigmoid(logits).cpu()
        auc = roc_auc_score(labels.cpu(), probs)
    return loss, auc, labels.cpu(), probs, logits.cpu()


def best_f1_threshold_on_logits(labels, logits):
    """Search candidate thresholds ON RAW LOGITS (not post-sigmoid probabilities,
    to avoid numerical saturation when probs are squashed near 0/1) for the
    threshold maximizing F1. MUST be called on validation data only -- the
    returned threshold is then applied, fixed, to the test set."""
    labels_np = labels.numpy()
    logits_np = logits.numpy()
    candidates = np.unique(logits_np)
    if candidates.size > 500:  # subsample candidate thresholds for speed on large val sets
        idx = np.linspace(0, candidates.size - 1, 500).astype(int)
        candidates = candidates[idx]
    best_f1, best_t = -1.0, 0.0
    for t in candidates:
        pred = (logits_np > t).astype(int)
        if pred.sum() == 0:
            continue
        _, _, f1_t, _ = precision_recall_fscore_support(labels_np, pred, average='binary', zero_division=0)
        if f1_t > best_f1:
            best_f1, best_t = f1_t, float(t)
    return best_t


def binary_stats_at_threshold(labels, logits, threshold):
    labels_np = labels.numpy()
    pred = (logits.numpy() > threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(labels_np, pred, average='binary', zero_division=0)
    return prec, rec, f1, float(pred.mean())


def all_positive_baseline(labels):
    labels_np = labels.numpy()
    pred = np.ones_like(labels_np)
    prec, rec, f1, _ = precision_recall_fscore_support(labels_np, pred, average='binary', zero_division=0)
    return prec, rec, f1


def evaluate_model(
    data, in_manuf, in_prod, hidden, lr, dropout, gat_heads,
    max_epochs, patience, device, model_type, seed,
    val_ratio, test_ratio, loss_type
):
    """
    Train + evaluate one (model, lr) setting.  Early-stops on validation AUC.
    Returns:
        test_loss, test_auc, test_pr_auc, precision, recall, f1, stop_epoch, mrr
    """
    # reproducibility --------------------------------------------------
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    data = add_reverse_edge(data)
    metadata = data.metadata()
    rels = {
        'fwd': ('manufacturer', 'has_product',      'product'),
        'rev': ('product',      'rev_has_product',  'manufacturer')
    }

    splitter = RandomLinkSplit(
        num_val=val_ratio, num_test=test_ratio,
        is_undirected=True, add_negative_train_samples=True,
        neg_sampling_ratio=1,
        edge_types=[rels['fwd']], rev_edge_types=[rels['rev']]
    )
    train_data, val_data, test_data = [d.to(device) for d in splitter(data)]

    # ---------------  loss  ------------------------------------------
    train_labels = train_data[rels['fwd']].edge_label.float()
    pos = train_labels.sum().item()
    neg = len(train_labels) - pos
    pos_weight = torch.tensor([neg / pos], device=device)

    if loss_type == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ---------------  model  -----------------------------------------
    if model_type == 'sage':
        model = HeteroSAGE(in_manuf, in_prod, hidden, rels, dropout)
    elif model_type == 'gat':
        model = HeteroGAT(in_manuf, in_prod, hidden, rels,
                          heads=gat_heads, dropout=dropout)
    elif model_type == 'hgt':
        model = HeteroHGT(in_manuf, in_prod, hidden, rels, metadata, dropout)
    else:
        raise ValueError(f"Unknown model '{model_type}'")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # ---------------  training loop  ---------------------------------
    best_auc, best_state, no_imp, stop_epoch = -float('inf'), None, 0, max_epochs
    for epoch in range(1, max_epochs + 1):
        train_step(train_data, model, optimizer, criterion, device)
        val_loss, val_auc, *_ = eval_step(val_data, model, criterion, device)

        if val_auc > best_auc:
            best_auc, best_state, no_imp = val_auc, {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }, 0
        else:
            no_imp += 1
            if no_imp >= patience:
                stop_epoch = epoch
                print(f"Early stop @ {epoch:03d}; no val-AUC gain for {patience} ep.")
                break

    # restore best weights --------------------------------------------
    model.load_state_dict(best_state)

    # ---------------  test metrics (unchanged: AUC/PR-AUC/F1@0.5/MRR) -
    test_loss, test_auc, test_labels, test_probs, test_logits = eval_step(
        test_data, model, criterion, device
    )
    test_pr_auc = average_precision_score(test_labels, test_probs)

    preds = (test_probs > 0.5).long()
    prec, rec, f1, _ = precision_recall_fscore_support(
        test_labels, preds, average='binary', zero_division=0
    )

    rel = ('manufacturer', 'has_product', 'product')
    mrr = compute_mrr(test_data[rel].edge_label_index, test_labels, test_probs)

    # ---------------  validation-selected threshold (additive, diagnostic) --
    # Re-eval val_data at this SAME best checkpoint (the training loop's last
    # eval_step call was for the final epoch tried, not necessarily this one).
    _, val_auc_recheck, val_labels, val_probs, val_logits = eval_step(
        val_data, model, criterion, device
    )
    val_thresh = best_f1_threshold_on_logits(val_labels, val_logits)
    test_prec_vt, test_rec_vt, test_f1_vt, test_posrate_vt = binary_stats_at_threshold(
        test_labels, test_logits, val_thresh
    )
    base_prec, base_rec, base_f1 = all_positive_baseline(test_labels)

    threshold_report = {
        'val_auc_recheck': val_auc_recheck,     # sanity check vs best_auc from training loop
        'val_thresh_logit': val_thresh,
        'f1_at_0.5': f1, 'prec_at_0.5': prec, 'rec_at_0.5': rec,
        'f1_at_val_thresh': test_f1_vt, 'prec_at_val_thresh': test_prec_vt,
        'rec_at_val_thresh': test_rec_vt, 'posrate_at_val_thresh': test_posrate_vt,
        'all_positive_baseline_f1': base_f1,
    }

    return test_loss, test_auc, test_pr_auc, prec, rec, f1, stop_epoch, mrr, best_auc, threshold_report


# ---------------------------------------------------------------------
# main()     grid-search over learning rates and models
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Hetero-GNN link prediction with LR grid search + MRR'
    )
    parser.add_argument('--input-pt',   type=str,   default='hetero_graph.pt')
    parser.add_argument('--max-epochs', type=int,   default=1000)
    parser.add_argument('--patience',   type=int,   default=20)
    parser.add_argument('--hidden',     type=int,   default=128)
    parser.add_argument('--dropout',    type=float, default=0.5)
    parser.add_argument('--loss',       type=str,   default='weighted_bce',
                        choices=['weighted_bce', 'focal'])
    parser.add_argument('--gat-heads',  type=int,   default=4)
    parser.add_argument('--val-ratio',  type=float, default=0.10)
    parser.add_argument('--test-ratio', type=float, default=0.10)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--device',     type=str,   default='cpu')
    parser.add_argument('--models',     type=str,   default=','.join(MODEL_CLASSES))
    parser.add_argument('--lrs',        type=str,   default=None,
                        help='Comma-separated LR values to use instead of the default '
                             '10-value grid, e.g. "5e-2" to re-evaluate a single already-'
                             'selected LR without re-searching the full grid.')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    data   = torch.load(args.input_pt, map_location=device)
    in_manuf = data['manufacturer'].x.size(1)
    in_prod  = data['product'].x.size(1)

    # ---------------  learning-rate grid  -----------------------------
    if args.lrs:
        lr_grid = [float(x) for x in args.lrs.split(',')]
    else:
        base_lrs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        factors  = [1, 5]
        lr_grid  = [f * b for f, b in itertools.product(factors, base_lrs)]

    summary = {}          # (model, lr) -> metrics tuple
    best_lr = {}          # model -> (lr, best_metric)

    for m in args.models.split(','):
        m = m.strip().lower()
        if m not in MODEL_CLASSES:
            print(f"Skipping unknown model '{m}'");  continue

        best_lr[m] = (None, -float('inf'))  # sentinel (metric to maximise)

        for lr in lr_grid:
            try:
                out = evaluate_model(
                    data, in_manuf, in_prod, args.hidden, lr,
                    args.dropout, args.gat_heads, args.max_epochs, args.patience,
                    device, m, args.seed, args.val_ratio, args.test_ratio, args.loss
                )
                loss, auc, pr_auc, p, r, f1, stop_ep, mrr, val_auc, thr_rep = out
                summary[(m, lr)] = out

                # select LR by validation ROC-AUC (matches paper Sec. 4.3.2),
                # NOT test AUC -- using test AUC here would be test-set leakage.
                metric = val_auc
                if metric > best_lr[m][1]:
                    best_lr[m] = (lr, metric)

                print(f"Model={m:<4} | LR={lr:.1e} | "
                      f"Stop={stop_ep:03d} | ValAUC={val_auc:.4f} | TestAUC={auc:.4f} | PR-AUC={pr_auc:.4f} "
                      f"| F1={f1:.4f} | MRR={mrr:.4f}")
                print(f"    [threshold] val-selected logit-thresh={thr_rep['val_thresh_logit']:.4f} | "
                      f"F1@0.5={thr_rep['f1_at_0.5']:.4f} (P={thr_rep['prec_at_0.5']:.4f} R={thr_rep['rec_at_0.5']:.4f}) | "
                      f"F1@val-thresh={thr_rep['f1_at_val_thresh']:.4f} (P={thr_rep['prec_at_val_thresh']:.4f} "
                      f"R={thr_rep['rec_at_val_thresh']:.4f} pred-pos-rate={thr_rep['posrate_at_val_thresh']:.4f}) | "
                      f"all-positive-baseline-F1={thr_rep['all_positive_baseline_f1']:.4f}")
            except Exception as e:
                print(f"!! ERR model={m} lr={lr:.1e}: {e}")

    # ---------- final report ---------------------------------------
    print("\n=== Best LR per model (selected by validation ROC-AUC) ===")
    for m in sorted(best_lr):
        lr, best_val_auc = best_lr[m]
        l, a, pa, p, r, f, s, mrr, val_auc, thr_rep = summary[(m, lr)]
        print(f"{m:<4}  best-lr={lr:.1e}  ValAUC={val_auc:.4f}  TestAUC={a:.4f}  PR-AUC={pa:.4f}  "
              f"F1={f:.4f}  MRR={mrr:.4f}  StopEp={s}")
        print(f"      F1@val-thresh={thr_rep['f1_at_val_thresh']:.4f} "
              f"(P={thr_rep['prec_at_val_thresh']:.4f} R={thr_rep['rec_at_val_thresh']:.4f} "
              f"pred-pos-rate={thr_rep['posrate_at_val_thresh']:.4f})  "
              f"all-positive-baseline-F1={thr_rep['all_positive_baseline_f1']:.4f}")


if __name__ == '__main__':
    main()
