#!/usr/bin/env python3
"""
Same extraction logic as vlm_select_images.py, but for a 25-manufacturer VAL
dry-run sample (fixed seed 999, matching VAL_DRYRUN_SEED in run_vlm_eval.py)
instead of the full TEST has_image set -- so the format/VRAM/speed dry run
uses real VAL manufacturers' own images, not a stand-in borrowed from TEST.
"""
import hashlib
import io
import os
import json
import random
import re

import pyarrow.parquet as pq
import yaml
from PIL import Image

SEED = 999
N_SAMPLE = 25
MAX_IMAGES_PER_MANU = 3
# Raw images are not redistributed (see PROVENANCE_AND_ETHICS.md), so this script needs a
# local copy of the original v1 release; set PMGRAPH_V1_DIR to its location.
V1_DIR = os.environ.get("PMGRAPH_V1_DIR", "PMGraph-v1")
PARQUET_PATH = os.path.join(V1_DIR, "data", "train-00000-of-00001.parquet")
OUT_DIR = "vlm_experiment/images_val_dryrun"


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
    import torch

    edges = torch.load("material_edges.pt", map_location="cpu", weights_only=False)
    cov = torch.load("manufacturer_coverage_flags.pt", map_location="cpu", weights_only=False)
    manus_sorted = cov["manus_sorted"]
    has_image = cov["has_image"]
    val_manu_idx = set(edges["val_manu_idx"].tolist())
    val_has_image = sorted(manus_sorted[i] for i in val_manu_idx if has_image[i])
    print(f"val has_image manufacturers (with >=1 recorded material edge): {len(val_has_image)}")

    rng = random.Random(SEED)
    target_manus = rng.sample(val_has_image, min(N_SAMPLE, len(val_has_image)))
    print(f"sampled {len(target_manus)} for the val dry-run")

    readme_path = os.path.join(V1_DIR, "README.md")
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
        for lab in batch.column("manufacturer").to_pylist():
            rows_by_manu_label.setdefault(lab, []).append(row_i)
            row_i += 1

    target_labels, missing = {}, []
    for manu in target_manus:
        lab = name2idx.get(canon(manu))
        if lab is None:
            missing.append(manu)
        else:
            target_labels[manu] = lab
    print(f"resolvable: {len(target_labels)}  unresolvable: {len(missing)}")

    needed_rows = {}
    for manu, lab in target_labels.items():
        all_rows = rows_by_manu_label.get(lab, [])
        rng_local = random.Random(f"{SEED}-{manu}")
        chosen = all_rows if len(all_rows) <= MAX_IMAGES_PER_MANU else rng_local.sample(all_rows, MAX_IMAGES_PER_MANU)
        needed_rows[manu] = sorted(chosen)
    all_needed = set(r for rows in needed_rows.values() for r in rows)
    print(f"image rows to extract: {len(all_needed)}")

    extracted = {}
    row_i = 0
    for batch in pf.iter_batches(batch_size=512, columns=["image_path"]):
        imgs = batch.column("image_path")
        for j in range(len(imgs)):
            if row_i in all_needed:
                extracted[row_i] = imgs[j].as_py()["bytes"]
            row_i += 1
        if len(extracted) == len(all_needed):
            break

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = []
    for manu, rows in needed_rows.items():
        d = safe_name(manu)
        manu_dir = f"{OUT_DIR}/{d}"
        os.makedirs(manu_dir, exist_ok=True)
        for order, r in enumerate(rows):
            raw = extracted.get(r)
            if raw is None:
                manifest.append({"manufacturer": manu, "order": order, "status": "missing_bytes"})
                continue
            try:
                im = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception as e:
                manifest.append({"manufacturer": manu, "order": order, "status": "decode_failed", "error": str(e)})
                continue
            im.save(f"{manu_dir}/{order}.jpg", quality=90)
            manifest.append({"manufacturer": manu, "order": order, "file": f"{d}/{order}.jpg",
                              "sha256": hashlib.sha256(raw).hexdigest(), "status": "ok"})

    n_ok = sum(1 for m in manifest if m["status"] == "ok")
    with open("vlm_experiment/image_manifest_val_dryrun.json", "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "sampled_manufacturers": target_manus,
                    "unresolvable_label": missing, "n_images_ok": n_ok,
                    "manifest": manifest}, f, indent=1)
    print(f"images ok: {n_ok} / {len(manifest)}")
    print("wrote vlm_experiment/image_manifest_val_dryrun.json")


if __name__ == "__main__":
    main()
