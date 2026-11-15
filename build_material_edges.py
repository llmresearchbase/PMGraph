#!/usr/bin/env python3
"""
Build the manufacturer-material positive-edge tensors for train/val/test,
using the frozen 56-category vocabulary and manufacturer-level split from
material_task_split.json. Manufacturer index order matches Graph_A.pt
exactly (same build_edges() logic, verified by assertion).
"""
import csv
import json
import torch


def canon(w):
    w = str(w).strip().lower()
    for p in ("https://", "http://"):
        if w.startswith(p):
            w = w[len(p):]
    return w.rstrip("/")


def main():
    with open("material_task_split.json", encoding="utf-8") as f:
        split = json.load(f)
    vocab = split["selected_vocab"]
    v2i = {v: i for i, v in enumerate(vocab)}
    train_set = set(split["train_manufacturers"])
    val_set = set(split["val_manufacturers"])
    test_set = set(split["test_manufacturers"])

    data = torch.load("hetero_graphs/Graph_A.pt", map_location="cpu", weights_only=False)
    n_manu = data["manufacturer"].x.size(0)

    edge_set = set()
    cols = [f"Product_{i}" for i in range(1, 151)]
    with open("Cleaned_Master_CSV_v3.csv", newline='', encoding='utf-8-sig', errors='replace') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        for r in rows:
            m = canon(r.get("Web", ""))
            for c in cols:
                p = r.get(c)
                if p and p.strip() and p.strip().lower() != "yes":
                    edge_set.add(m)
    manus_sorted = sorted(edge_set)
    assert len(manus_sorted) == n_manu
    m2i = {m: i for i, m in enumerate(manus_sorted)}

    mat_cols = [c for c in rows[0].keys() if c.startswith("Material_Capability_")]
    manu_materials = {}
    for r in rows:
        web = canon(r.get("Web", ""))
        if web not in m2i:
            continue
        vals = set(str(r.get(c)).strip().lower() for c in mat_cols if r.get(c) and str(r.get(c)).strip())
        manu_materials[web] = vals & set(vocab)  # restrict to frozen vocab

    def build(manu_set):
        m_idx, mat_idx = [], []
        for w in manu_set:
            for v in manu_materials.get(w, []):
                m_idx.append(m2i[w])
                mat_idx.append(v2i[v])
        return torch.tensor(m_idx, dtype=torch.long), torch.tensor(mat_idx, dtype=torch.long)

    train_m, train_v = build(train_set)
    val_m, val_v = build(val_set)
    test_m, test_v = build(test_set)

    print(f"positive edges (restricted to {len(vocab)}-category vocab): "
          f"train={len(train_m)} val={len(val_m)} test={len(test_m)}")

    torch.save({
        "vocab": vocab,
        "n_manu": n_manu,
        "train_manu_idx": train_m, "train_mat_idx": train_v,
        "val_manu_idx": val_m, "val_mat_idx": val_v,
        "test_manu_idx": test_m, "test_mat_idx": test_v,
        "manus_sorted": manus_sorted,
    }, "material_edges.pt")
    print("saved material_edges.pt")


if __name__ == "__main__":
    main()
