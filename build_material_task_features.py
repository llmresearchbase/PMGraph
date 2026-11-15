#!/usr/bin/env python3
"""
Build manufacturer features for the material-relation-recovery pilot, with
ALL Material_Capability-derived information removed: the 'mat' one-hot group
is excluded entirely from feature construction (unlike the standard
Graph_A features, which include it). Everything else (TF-IDF Name/NAICS
text, status/industry/cert/proc/naics_cat/city/state one-hot, geo) is built
identically to Graph-A-Builder.py, then SVD-reduced to 64 dims as before.

Natural free text (Business_Description, Product_i strings) is NOT used as
a manufacturer feature in Graph_A's existing build (only Name + NAICS_Description_1
are), so there is no concatenation-based leakage path to worry about here --
confirmed by inspection, not assumed.
"""
import importlib.util
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

spec = importlib.util.spec_from_file_location("ga", "Graph-A-Builder.py")
ga = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ga)


def build_manufacturer_features_no_material(df, m2i):
    name_vec = TfidfVectorizer(max_features=64, stop_words='english')
    naics_desc_vec = TfidfVectorizer(max_features=256, stop_words='english')
    names = df['Name'].fillna('').astype(str)
    naics_descs = df['NAICS_Description_1'].fillna('').astype(str)
    name_tfidf = name_vec.fit_transform(names).toarray()
    naics_tfidf = naics_desc_vec.fit_transform(naics_descs).toarray()

    cat_groups = {
        'status': [c for c in df.columns if c.startswith('Business_Status_')],
        'industry': [c for c in df.columns if c.startswith('Industry_')],
        'cert': [c for c in df.columns if c.startswith('Certification_')],
        'proc': [c for c in df.columns if c.startswith('Process_Capability_')],
        # 'mat' intentionally excluded -- this is the material-prediction target
        'naics_cat': [c for c in df.columns if c.startswith('NAICS_Code_')],
        'city': ['City_1'],
        'state': ['State_1'],
    }
    vocabs = {}
    for grp, cols in cat_groups.items():
        if grp == 'naics_cat':
            vocabs[grp] = sorted({str(v)[:2] for c in cols for v in df[c].dropna()})
        else:
            vocabs[grp] = sorted({str(v).strip() for c in cols for v in df[c].dropna()})
    tok2i = {grp: {v: i for i, v in enumerate(vocabs[grp])} for grp in vocabs}

    id2pos = {ga.canonical_manufacturer_id(v): idx for idx, v in enumerate(df['Web'].fillna('').astype(str))}
    rows = []
    for manu, _ in sorted(m2i.items(), key=lambda x: x[1]):
        pos = id2pos.get(manu)
        row = df.iloc[pos]
        feats = [np.log1p(row.get('Employees', 0) or 0)]
        feats += name_tfidf[pos].tolist()
        feats += naics_tfidf[pos].tolist()
        for grp in ['status', 'industry', 'cert', 'proc', 'city', 'state']:
            vec = np.zeros(len(vocabs[grp]), dtype=np.float32)
            if grp in ['city', 'state']:
                val = row.get(grp.capitalize() + '_1')
                tokens = [str(val).strip()] if pd.notna(val) else []
            else:
                tokens = [str(row[c]).strip() for c in cat_groups[grp] if pd.notna(row[c])]
            for t in tokens:
                idx = tok2i[grp].get(t)
                if idx is not None:
                    vec[idx] = 1.0
            feats += vec.tolist()
        feats += [row.get('Latitude_1') or 0.0, row.get('Longitude_1') or 0.0]
        rows.append(feats)

    X = np.vstack(rows).astype(np.float32)
    for j in [0, X.shape[1] - 2, X.shape[1] - 1]:
        col = X[:, j]
        std = col.std()
        X[:, j] = (col - col.mean()) / std if std > 1e-6 else 0.0

    reducer = TruncatedSVD(n_components=64, random_state=42)
    return reducer.fit_transform(X)


def main():
    df = ga.load_dataframe("Cleaned_Master_CSV_v3.csv")
    edges, manufacturers, products = ga.build_edges(df)
    data, m2i, p2i = ga.build_heterodata(edges, manufacturers, products)

    mf = build_manufacturer_features_no_material(df, m2i)
    print("built manufacturer features, shape:", mf.shape, " (material info excluded)")

    torch.save(torch.tensor(mf, dtype=torch.float32), "manufacturer_features_no_material.pt")
    with open("manufacturer_order_for_material_task.json", "w", encoding="utf-8") as f:
        import json
        json.dump(sorted(m2i.items(), key=lambda x: x[1]), f)
    print("saved manufacturer_features_no_material.pt + manufacturer_order_for_material_task.json")


if __name__ == "__main__":
    main()
