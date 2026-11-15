#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph-A_builder.py

This script loads a manufacturer-product dataset from CSV,
constructs a bipartite manufacturer?product heterograph (Graph A),
uses the 'Name' column as the manufacturer identifier,
and assigns TF-IDF + one-hot attribute features to each node.
Performs TruncatedSVD to reduce manufacturer features to 64 dims.
Saves the final graph to hetero_graphs/Graph_A.pt
"""
import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def load_dataframe(csv_path: str, nrows: int = None) -> pd.DataFrame:

    return pd.read_csv(csv_path, nrows=nrows, low_memory=False)


def canonical_manufacturer_id(web: str) -> str:
    """Stable manufacturer ID: normalized Web domain (NOT Name -- Name is not
    unique: 31 groups of distinct companies share a display-name string, the
    largest being 3 real companies all named 'PRECISION INDUSTRIES, INC.').
    Normalization confirmed collision-free against the full 14,007-row CSV."""
    w = str(web).strip().lower()
    for prefix in ("https://", "http://"):
        if w.startswith(prefix):
            w = w[len(prefix):]
    return w.rstrip("/")


def build_edges(df: pd.DataFrame, max_products: int = 150):
    product_cols = [f'Product_{i}' for i in range(1, max_products+1)]
    edge_set = set()
    for _, row in df.iterrows():
        manu_id = canonical_manufacturer_id(row['Web'])
        for col in product_cols:
            prod = row.get(col)
            if pd.notna(prod) and str(prod).strip().lower() != 'yes':
                edge_set.add((manu_id, str(prod).strip()))
    edges = sorted(edge_set)
    manufacturers = sorted({m for m, _ in edges})
    products = sorted({p for _, p in edges})
    return edges, manufacturers, products


def build_heterodata(edges, manufacturers, products):
    m2i = {m: i for i, m in enumerate(manufacturers)}
    p2i = {p: i for i, p in enumerate(products)}
    src = [m2i[m] for m, _ in edges]
    dst = [p2i[p] for _, p in edges]
    data = HeteroData()
    data['manufacturer','has_product','product'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    return data, m2i, p2i


def build_manufacturer_features(df: pd.DataFrame, m2i: dict) -> np.ndarray:
    # TF-IDF on Name & NAICS_Description_1
    name_vec = TfidfVectorizer(max_features=64, stop_words='english')
    naics_desc_vec = TfidfVectorizer(max_features=256, stop_words='english')
    names = df['Name'].fillna('').astype(str)
    naics_descs = df['NAICS_Description_1'].fillna('').astype(str)
    name_tfidf = name_vec.fit_transform(names).toarray()
    naics_tfidf = naics_desc_vec.fit_transform(naics_descs).toarray()

    # One-hot groups including city and state
    cat_groups = {
        'status': [c for c in df.columns if c.startswith('Business_Status_')],
        'industry': [c for c in df.columns if c.startswith('Industry_')],
        'cert': [c for c in df.columns if c.startswith('Certification_')],
        'proc': [c for c in df.columns if c.startswith('Process_Capability_')],
        'mat': [c for c in df.columns if c.startswith('Material_Capability_')],
        'naics_cat': [c for c in df.columns if c.startswith('NAICS_Code_')],
        'city': ['City_1'],
        'state': ['State_1']
    }
    # Build vocabulary for each categorical group
    vocabs = {}
    for grp, cols in cat_groups.items():
        if grp == 'naics_cat':
            vocabs[grp] = sorted({str(v)[:2] for c in cols for v in df[c].dropna()})
        else:
            vocabs[grp] = sorted({str(v).strip() for c in cols for v in df[c].dropna()})
    tok2i = {grp: {v:i for i, v in enumerate(vocabs[grp])} for grp in vocabs}

    # Map canonical manufacturer ID (normalized Web) to row index -- NOT Name,
    # which collides for 31 groups of distinct real companies.
    id2pos = {canonical_manufacturer_id(v): idx for idx, v in enumerate(df['Web'].fillna('').astype(str))}
    rows = []
    for manu, _ in sorted(m2i.items(), key=lambda x: x[1]):
        pos = id2pos.get(manu)
        row = df.iloc[pos]
        feats = []
        # Numeric: Employees
        feats.append(np.log1p(row.get('Employees', 0) or 0))
        # Text embeddings via TF-IDF
        feats += name_tfidf[pos].tolist()
        feats += naics_tfidf[pos].tolist()
        # One-hot multi-hot for each group
        for grp in ['status','industry','cert','proc','mat','city','state']:
            vec = np.zeros(len(vocabs[grp]), dtype=np.float32)
            tokens = []
            if grp in ['city','state']:
                val = row.get(grp.capitalize() + '_1')
                if pd.notna(val): tokens = [str(val).strip()]
            elif grp == 'naics_cat':
                tokens = [str(row[c])[:2] for c in cat_groups['naics_cat'] if pd.notna(row[c])]
            else:
                tokens = [str(row[c]).strip() for c in cat_groups[grp] if pd.notna(row[c])]
            for t in tokens:
                idx = tok2i[grp].get(t)
                if idx is not None:
                    vec[idx] = 1.0
            feats += vec.tolist()
        # Geo coords
        feats += [row.get('Latitude_1') or 0.0, row.get('Longitude_1') or 0.0]
        rows.append(feats)

    X = np.vstack(rows).astype(np.float32)
    # Standardize numerical columns: Employees at 0, coords at -2,-1
    for j in [0, X.shape[1]-2, X.shape[1]-1]:
        col = X[:, j]; std = col.std()
        X[:, j] = (col - col.mean())/std if std>1e-6 else 0.0

    # Reduce to 64 dims via TruncatedSVD
    reducer = TruncatedSVD(n_components=64, random_state=42)
    X_reduced = reducer.fit_transform(X)
    return X_reduced


def build_product_features(df: pd.DataFrame, p2i: dict, device: torch.device) -> torch.Tensor:

    # 1) Recover the exact strings that went into p2i
    #    (these are the non-empty entries from Product_1�Product_n)
    idx2prod = [None] * len(p2i)
    for prod_text, idx in p2i.items():
        idx2prod[idx] = prod_text

    # 2) TF-IDF encode those raw product strings
    tfidf = TfidfVectorizer(max_features=256, stop_words='english')
    X_tfidf = tfidf.fit_transform(idx2prod)   # shape [N_prod, 256]

    # 3) Reduce to 64 dims
    svd = TruncatedSVD(n_components=64, random_state=42)
    X_reduced = svd.fit_transform(X_tfidf)    # shape [N_prod, 64]

    # 4) Return as a torch tensor on the chosen device
    return torch.tensor(X_reduced, dtype=torch.float32).to(device)


def reload_graph(path: str, map_location: str = 'cpu') -> HeteroData:
    return torch.load(path, map_location=map_location)


def print_graph_stats(data: HeteroData):
    n_m = data['manufacturer'].x.size(0)
    n_p = data['product'].x.size(0)
    n_e = data['manufacturer','has_product','product'].edge_index.size(1)
    print(f"# manufacturers: {n_m}")
    print(f"# products:      {n_p}")
    print(f"# edges:         {n_e}")
    print(f"manuf feat dim: {data['manufacturer'].x.size(1)}")
    print(f"prod feat dim:  {data['product'].x.size(1)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',      type=str, default='./Cleaned_Master_CSV_v3.csv')
    parser.add_argument('--nrows',    type=int, default=None)
    parser.add_argument('--reload',   action='store_true')
    parser.add_argument('--input-pt', type=str, default='hetero_graphs/Graph_A.pt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.reload:
        data = reload_graph(args.input_pt, 'cpu').to(device)
        print(f"Reloaded: {args.input_pt}")
        print_graph_stats(data)
        return

    df = load_dataframe(args.csv, args.nrows)
    edges, manufacturers, products = build_edges(df)
    data, m2i, p2i = build_heterodata(edges, manufacturers, products)

    mf = build_manufacturer_features(df, m2i)
    data['manufacturer'].x = torch.tensor(mf, dtype=torch.float32).to(device)
    data['product'].x = build_product_features(df, p2i, device)

    os.makedirs(os.path.dirname(args.input_pt), exist_ok=True)
    torch.save(data, args.input_pt)
    print(f"Saved: {args.input_pt}")
    print_graph_stats(data)


if __name__ == '__main__':
    main()