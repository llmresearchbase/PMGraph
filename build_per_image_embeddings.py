#!/usr/bin/env python3
"""
Per-image Jina-CLIP-v1 embeddings (not saved by the original
build_manufacturer_image_embeddings.py, which only persisted the running
mean -- verified by reading that script directly before committing to this
rerun). Same encoder, same frozen weights, same data source, so the
per-manufacturer MEAN of what this script saves should reproduce the
already-archived manufacturer_image_embeddings.pt exactly -- checked below,
not assumed.

Also embeds the 56 frozen material-vocabulary category names with the SAME
model's text tower, into the SAME joint embedding space, so image-vs-
candidate cosine similarity is meaningful (a text embedder from a different
model/space would not be comparable -- checked and rejected as an option
before starting, see conversation).

Output: manufacturer_per_image_embeddings.pt
  { "web_domain": str -> np.ndarray[n_images, 768] }  (only has-image
  manufacturers with >=1 successfully-decoded image appear as keys)
  material_vocab_text_embeddings.pt
  { "vocab": [str], "embeddings": np.ndarray[56, 768] }
"""
import io
import os
import re
import yaml
import numpy as np
import torch
from PIL import Image
import pyarrow.parquet as pq
from transformers import AutoModel

# Raw images are not redistributed (see PROVENANCE_AND_ETHICS.md), so this script needs a
# local copy of the original v1 release; set PMGRAPH_V1_DIR to its location.
V1_DIR = os.environ.get("PMGRAPH_V1_DIR", "PMGraph-v1")
PARQUET_PATH = os.path.join(V1_DIR, "data", "train-00000-of-00001.parquet")


def load_idx2name():
    readme_path = os.path.join(V1_DIR, "README.md")
    with open(readme_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    frontmatter = yaml.safe_load(m.group(1))
    names_map = frontmatter["dataset_info"]["features"][1]["dtype"]["class_label"]["names"]
    return {int(k): v for k, v in names_map.items()}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    idx2name = load_idx2name()
    table = pq.read_table(PARQUET_PATH, columns=["image_path", "manufacturer"])
    image_col = table.column("image_path")
    manu_col = table.column("manufacturer").to_pylist()
    n = len(manu_col)
    print("total rows:", n)

    model = AutoModel.from_pretrained("jinaai/jina-clip-v1", trust_remote_code=True).to(device)
    model.eval()

    per_image = {}  # web_domain -> list[np.ndarray(768)]
    failed = {}

    BATCH = 32
    buf_imgs, buf_domains = [], []

    def flush():
        if not buf_imgs:
            return
        try:
            with torch.no_grad():
                embs = model.encode_image(buf_imgs)
            embs = np.asarray(embs)
        except Exception as e:
            print(f"batch encode failed ({len(buf_imgs)} imgs): {e}; falling back one-by-one")
            embs = []
            for im in buf_imgs:
                try:
                    with torch.no_grad():
                        v = model.encode_image([im])[0]
                except Exception:
                    v = None
                embs.append(v)
        for dom, v in zip(buf_domains, embs):
            if v is None:
                failed[dom] = failed.get(dom, 0) + 1
                continue
            per_image.setdefault(dom, []).append(np.asarray(v, dtype=np.float32))
        buf_imgs.clear(); buf_domains.clear()

    for i in range(n):
        dom = idx2name[manu_col[i]]
        rec = image_col[i].as_py()
        try:
            im = Image.open(io.BytesIO(rec["bytes"])).convert("RGB")
        except Exception:
            failed[dom] = failed.get(dom, 0) + 1
            continue
        buf_imgs.append(im)
        buf_domains.append(dom)
        if len(buf_imgs) >= BATCH:
            flush()
        if i % 2000 == 0:
            print(f"  processed {i}/{n}")
    flush()

    result = {dom: np.stack(vs, axis=0) for dom, vs in per_image.items()}
    torch.save(result, "manufacturer_per_image_embeddings.pt")
    total_ok = sum(v.shape[0] for v in result.values())
    total_failed = sum(failed.values())
    print(f"\nsaved manufacturer_per_image_embeddings.pt  ({len(result)} manufacturers, "
          f"{total_ok} images ok, {total_failed} failed)")

    # ---- consistency check against the already-archived flat-pooled means ----
    archived = torch.load("manufacturer_image_embeddings.pt", map_location="cpu", weights_only=False)
    max_diff, n_checked = 0.0, 0
    for dom, arr in result.items():
        if dom not in archived:
            continue
        recomputed_mean = arr.mean(axis=0)
        archived_mean = np.asarray(archived[dom]["embedding"], dtype=np.float32)
        d = float(np.abs(recomputed_mean - archived_mean).max())
        max_diff = max(max_diff, d)
        n_checked += 1
    print(f"consistency check: recomputed per-manufacturer mean vs archived "
          f"manufacturer_image_embeddings.pt -- max abs diff over {n_checked} "
          f"manufacturers = {max_diff:.6f}")

    # ---- embed the 56-category frozen vocabulary with the SAME model's text tower ----
    edges = torch.load("material_edges.pt", map_location="cpu", weights_only=False)
    vocab = edges["vocab"]
    with torch.no_grad():
        text_embs = model.encode_text(list(vocab))
    text_embs = np.asarray(text_embs, dtype=np.float32)
    torch.save({"vocab": vocab, "embeddings": text_embs}, "material_vocab_text_embeddings.pt")
    print(f"saved material_vocab_text_embeddings.pt  ({len(vocab)} categories, dim={text_embs.shape[1]})")


if __name__ == "__main__":
    main()
