#!/usr/bin/env python3
"""
Selects up to 3 images per has_image TEST manufacturer, fixed seed, saved
with manufacturer_id/order/sha256 recorded -- not selected based on material
labels. Extracts actual image bytes from the official HF parquet (the same
source build_manufacturer_image_embeddings.py read from) into
vlm_experiment/images/<manufacturer_id_safe>/<order>.jpg, plus a manifest.
"""
import hashlib
import io
import json
import random
import re

import pandas as pd
import pyarrow.parquet as pq
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image

SEED = 123
MAX_IMAGES_PER_MANU = 3
# Raw images exist only in the original (v1) HuggingFace release, pinned here.
V1_REVISION = "v1"
PARQUET_PATH = hf_hub_download(repo_id="shawntzx/PMGraph", filename="data/train-00000-of-00001.parquet",
                               repo_type="dataset", revision=V1_REVISION)
OUT_DIR = "vlm_experiment/images"


def canon(w):
    w = str(w).strip().lower()
    for p in ("https://", "http://"):
        if w.startswith(p):
            w = w[len(p):]
    return w.rstrip("/")


def safe_name(manu_id):
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", manu_id)[:120]


def main():
    import os
    with open("material_task_split.json", encoding="utf-8") as f:
        split = json.load(f)
    test_manus = set(split["test_manufacturers"])

    import torch
    cov = torch.load("manufacturer_coverage_flags.pt", map_location="cpu", weights_only=False)
    manus_sorted = cov["manus_sorted"]
    has_image = cov["has_image"]
    idx_of = {m: i for i, m in enumerate(manus_sorted)}
    test_has_image = sorted(m for m in test_manus if m in idx_of and has_image[idx_of[m]])
    print(f"test has_image manufacturers: {len(test_has_image)}")

    readme_path = hf_hub_download(repo_id="shawntzx/PMGraph", filename="README.md", repo_type="dataset",
                                  revision=V1_REVISION)
    text = open(readme_path, encoding="utf-8", errors="replace").read()
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    frontmatter = yaml.safe_load(m.group(1))
    names_map = frontmatter["dataset_info"]["features"][1]["dtype"]["class_label"]["names"]
    idx2name = {int(k): v for k, v in names_map.items()}
    name2idx = {canon(v): k for k, v in idx2name.items()}

    pf = pq.ParquetFile(PARQUET_PATH)
    rows_by_manu_label = {}
    row_i = 0
    for batch in pf.iter_batches(batch_size=512, columns=["manufacturer"]):
        labels = batch.column("manufacturer").to_pylist()
        for lab in labels:
            rows_by_manu_label.setdefault(lab, []).append(row_i)
            row_i += 1
    print(f"total parquet rows scanned: {row_i}")

    manifest = []
    missing_label = []
    rng = random.Random(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    target_labels = {}
    for manu in test_has_image:
        lab = name2idx.get(canon(manu))
        if lab is None:
            missing_label.append(manu)
            continue
        target_labels[manu] = lab
    print(f"manufacturers with a resolvable parquet label: {len(target_labels)} "
          f"(unresolvable: {len(missing_label)})")

    needed_rows = {}
    for manu, lab in target_labels.items():
        all_rows = rows_by_manu_label.get(lab, [])
        rng_local = random.Random(f"{SEED}-{manu}")
        chosen = all_rows if len(all_rows) <= MAX_IMAGES_PER_MANU else rng_local.sample(all_rows, MAX_IMAGES_PER_MANU)
        chosen = sorted(chosen)
        needed_rows[manu] = chosen

    all_needed_row_set = set(r for rows in needed_rows.values() for r in rows)
    print(f"total image rows to extract: {len(all_needed_row_set)}")

    extracted = {}
    row_i = 0
    for batch in pf.iter_batches(batch_size=512, columns=["image_path"]):
        imgs = batch.column("image_path")
        for j in range(len(imgs)):
            if row_i in all_needed_row_set:
                rec = imgs[j].as_py()
                extracted[row_i] = rec["bytes"]
            row_i += 1
        if len(extracted) == len(all_needed_row_set):
            break

    for manu, rows in needed_rows.items():
        d = safe_name(manu)
        manu_dir = f"{OUT_DIR}/{d}"
        os.makedirs(manu_dir, exist_ok=True)
        for order, r in enumerate(rows):
            raw = extracted.get(r)
            if raw is None:
                continue
            try:
                im = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception as e:
                manifest.append({"manufacturer": manu, "order": order, "parquet_row": r,
                                  "status": "decode_failed", "error": str(e)})
                continue
            fname = f"{order}.jpg"
            im.save(f"{manu_dir}/{fname}", quality=90)
            sha = hashlib.sha256(raw).hexdigest()
            manifest.append({"manufacturer": manu, "order": order, "parquet_row": r,
                              "file": f"{d}/{fname}", "sha256": sha, "width": im.width,
                              "height": im.height, "status": "ok"})

    n_ok = sum(1 for m in manifest if m["status"] == "ok")
    n_failed = sum(1 for m in manifest if m["status"] != "ok")
    manus_with_zero_ok = [m for m in target_labels if not any(
        e["manufacturer"] == m and e["status"] == "ok" for e in manifest)]

    with open("vlm_experiment/image_manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "seed": SEED, "max_images_per_manu": MAX_IMAGES_PER_MANU,
            "n_test_has_image_manufacturers": len(test_has_image),
            "n_resolvable_label": len(target_labels),
            "n_unresolvable_label": len(missing_label),
            "unresolvable_manufacturers": missing_label,
            "n_images_ok": n_ok, "n_images_failed": n_failed,
            "manufacturers_with_zero_ok_images": manus_with_zero_ok,
            "manifest": manifest,
        }, f, indent=1)

    print(f"images ok: {n_ok}  failed: {n_failed}")
    print(f"manufacturers with ZERO usable extracted images despite has_image=True: {len(manus_with_zero_ok)}")
    print("wrote vlm_experiment/image_manifest.json")


if __name__ == "__main__":
    main()
