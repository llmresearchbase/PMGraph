#!/usr/bin/env python3
"""
Freeze the train/val/test RandomLinkSplit for Graph_A (seed=42, val=0.05,
test=0.10, matching the official LP-ABCD.py protocol) and SAVE it to disk,
so every downstream baseline and GNN evaluation in this round reuses the
exact same split object -- not just "same seed", but the literal same
tensors -- for a fair, reproducible comparison.
"""
import random
import numpy as np
import torch
from torch_geometric.transforms import RandomLinkSplit

SEED = 42
VAL_RATIO, TEST_RATIO = 0.05, 0.10

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

data = torch.load("hetero_graphs/Graph_A.pt", map_location="cpu", weights_only=False)

fwd = ("manufacturer", "has_product", "product")
rev = ("product", "rev_has_product", "manufacturer")
src, dst = data[fwd].edge_index
data[rev].edge_index = torch.stack([dst, src], dim=0)

splitter = RandomLinkSplit(
    num_val=VAL_RATIO, num_test=TEST_RATIO,
    is_undirected=True, add_negative_train_samples=True,
    neg_sampling_ratio=1,
    edge_types=[fwd], rev_edge_types=[rev],
)
train_data, val_data, test_data = splitter(data)

torch.save(
    {"train": train_data, "val": val_data, "test": test_data,
     "seed": SEED, "val_ratio": VAL_RATIO, "test_ratio": TEST_RATIO,
     "source_graph": "hetero_graphs/Graph_A.pt"},
    "frozen_split_GraphA_seed42.pt",
)

print("saved frozen_split_GraphA_seed42.pt")
print("train pos+neg:", train_data[fwd].edge_label.numel(),
      " val:", val_data[fwd].edge_label.numel(),
      " test:", test_data[fwd].edge_label.numel())
