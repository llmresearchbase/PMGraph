#!/usr/bin/env python3
"""
Per-(manufacturer, candidate-material) target-relevance-weighted image
feature, as an alternative to the current flat (uniform-mean) image
aggregation. Zero-training weighting: each manufacturer's images are
softmax-weighted by their Jina-CLIP-v1 cosine similarity to the CANDIDATE
material's own text embedding (same frozen model, same joint space -- not a
newly trained attention mechanism). Temperature is a single value selected
on the validation set by the training script that consumes this file, not
grid-searched here.

Reuses the EXACT SVD fit recipe from build_image_augmented_features.py
(TruncatedSVD(n_components=32, random_state=42), fit on has-image rows'
flat-pooled embeddings only) -- refit here since the original fitted object
was never saved, only its output. Verified below to reproduce the already-
archived manufacturer_features_real.pt image slice before being trusted to
transform the new target-conditioned vectors into the same space.

Output: manufacturer_target_weighted_image_features.pt
  { "features": FloatTensor[n_manu, n_vocab, 32],  # zero row for no-image manufacturers
    "temperatures_available": [float, ...] }  -- one 32-dim vector PER
  TEMPERATURE CANDIDATE is NOT precomputed here (that would bake in a choice);
  instead this script is called once per temperature candidate by the
  training script during validation-set selection, and the chosen
  temperature's output is what gets saved/used for the frozen test run.
"""
import sys
import csv
import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD


def canon(w):
    w = str(w).strip().lower()
    for p in ("https://", "http://"):
        if w.startswith(p):
            w = w[len(p):]
    return w.rstrip("/")


def build(temperature: float, verify: bool = True):
    data = torch.load("hetero_graphs/Graph_A.pt", map_location="cpu", weights_only=False)
    n_manu = data["manufacturer"].x.shape[0]

    edge_set = set()
    with open("Cleaned_Master_CSV_v3.csv", newline='', encoding='utf-8-sig', errors='replace') as f:
        reader = csv.DictReader(f)
        cols = [f"Product_{i}" for i in range(1, 151)]
        for r in reader:
            m = canon(r.get("Web", ""))
            for c in cols:
                p = r.get(c)
                if p and p.strip() and p.strip().lower() != "yes":
                    edge_set.add((m, p.strip()))
    manus_sorted = sorted({m for m, _ in edge_set})
    assert len(manus_sorted) == n_manu

    flat_img_emb = torch.load("manufacturer_image_embeddings.pt", map_location="cpu", weights_only=False)
    per_image = torch.load("manufacturer_per_image_embeddings.pt", map_location="cpu", weights_only=False)
    vocab_text = torch.load("material_vocab_text_embeddings.pt", map_location="cpu", weights_only=False)
    vocab, text_embs = vocab_text["vocab"], vocab_text["embeddings"]  # [56, 768]

    edges = torch.load("material_edges.pt", map_location="cpu", weights_only=False)
    assert edges["vocab"] == vocab, "vocab order mismatch between material_edges.pt and material_vocab_text_embeddings.pt"

    raw_img = np.zeros((n_manu, 768), dtype=np.float32)
    real_rows = []
    for i, m in enumerate(manus_sorted):
        rec = flat_img_emb.get(m)
        if rec is not None and rec["n_images"] > 0:
            raw_img[i] = rec["embedding"]
            real_rows.append(i)
    print(f"has-image manufacturers: {len(real_rows)} / {n_manu}")

    svd = TruncatedSVD(n_components=32, random_state=42)
    svd.fit(raw_img[real_rows])

    if verify:
        archived = torch.load("manufacturer_features_real.pt", map_location="cpu", weights_only=False).numpy()
        archived_img_slice = archived[:, 65:]
        recomputed = svd.transform(raw_img[real_rows])
        recomputed_full = np.zeros((n_manu, 32), dtype=np.float32)
        for idx, row in zip(real_rows, recomputed):
            recomputed_full[idx] = row
        max_diff = float(np.abs(recomputed_full[real_rows] - archived_img_slice[real_rows]).max())
        print(f"SVD-refit verification: max abs diff vs archived manufacturer_features_real.pt "
              f"image slice (has-image rows only) = {max_diff:.8f}")
        if max_diff > 1e-4:
            print("WARNING: refit SVD does not closely match the archived transform -- "
                  "do not trust this feature set until this is resolved.")

    text_norm = text_embs / (np.linalg.norm(text_embs, axis=1, keepdims=True) + 1e-9)  # [56,768]

    out = np.zeros((n_manu, len(vocab), 32), dtype=np.float32)
    for idx in real_rows:
        m = manus_sorted[idx]
        imgs = per_image.get(m)
        if imgs is None or imgs.shape[0] == 0:
            continue  # shouldn't happen given real_rows construction, but guard anyway
        imgs_norm = imgs / (np.linalg.norm(imgs, axis=1, keepdims=True) + 1e-9)  # [k,768]
        sims = imgs_norm @ text_norm.T  # [k, 56]
        weights = np.exp(sims / temperature)
        weights /= weights.sum(axis=0, keepdims=True)  # softmax over images, per candidate column
        weighted_768 = weights.T @ imgs  # [56, 768]
        out[idx] = svd.transform(weighted_768)  # [56, 32]

    return torch.tensor(out), manus_sorted, vocab


def main():
    temperature = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
    feats, manus_sorted, vocab = build(temperature, verify=True)
    torch.save({"features": feats, "temperature": temperature,
                "manus_sorted": manus_sorted, "vocab": vocab},
               "manufacturer_target_weighted_image_features.pt")
    print(f"saved manufacturer_target_weighted_image_features.pt  "
          f"shape={tuple(feats.shape)}  temperature={temperature}")


if __name__ == "__main__":
    main()
