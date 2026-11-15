#!/usr/bin/env python3
"""
Qwen3-VL-2B-Instruct zero-shot reference track for recorded-material-relation
recovery. Two conditions: TEXT (manufacturer text only) and TEXT+IMAGE (same
text + up to 3 of that manufacturer's website images). Same 56-category
frozen vocabulary and test split as the supervised MLP/GNN tracks. This is a
zero-shot reference, reported separately from the supervised tracks -- no
claim that parameter count makes this a "fair" comparison either way.

Usage:
  python run_vlm_eval.py --mode val_dryrun     # 20-30 val manufacturers, smoke test
  python run_vlm_eval.py --mode test           # frozen test run, config must not change after this
"""
import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
SEED = 123
MAX_NEW_TOKENS = 256
VAL_DRYRUN_N = 25
VAL_DRYRUN_SEED = 999

# Pixel budget per image, found necessary by the val dry-run: the processor's
# default cap (longest_edge=16,777,216 total pixels) is effectively unbounded
# for real photos. One val manufacturer's raw website photo (4128x2322,
# ~9.6MP) produced ~9,300 vision tokens, causing a 374s/21GB outlier (vs.
# ~4s/4.4GB for normal images) -- almost certainly spilling into Windows'
# slow shared-GPU-memory fallback rather than true VRAM. Capped here to a
# fixed, documented budget (same order of magnitude as common Qwen-VL
# deployment defaults) so per-call cost is bounded and reproducible.
IMAGE_MAX_PIXELS = 1024 * 1024
IMAGE_MIN_PIXELS = 256 * 256

ROOT = Path(__file__).resolve().parent.parent  # SupplyChain/
IMAGES_DIR = Path(__file__).resolve().parent / "images"


def load_vocab():
    with open(ROOT / "material_task_split.json", encoding="utf-8") as f:
        split = json.load(f)
    vocab = split["selected_vocab"]  # fixed order already frozen in this file
    return vocab, split


def load_test_positives():
    edges = torch.load(ROOT / "material_edges.pt", map_location="cpu", weights_only=False)
    vocab = edges["vocab"]
    idx2mat = {i: v for i, v in enumerate(vocab)}
    cov = torch.load(ROOT / "manufacturer_coverage_flags.pt", map_location="cpu", weights_only=False)
    manus_sorted = cov["manus_sorted"]
    pos_by_manu = {}
    for m_idx, v_idx in zip(edges["test_manu_idx"].tolist(), edges["test_mat_idx"].tolist()):
        manu_id = manus_sorted[m_idx]
        pos_by_manu.setdefault(manu_id, set()).add(idx2mat[v_idx])
    pos_by_manu_val = {}
    for m_idx, v_idx in zip(edges["val_manu_idx"].tolist(), edges["val_mat_idx"].tolist()):
        manu_id = manus_sorted[m_idx]
        pos_by_manu_val.setdefault(manu_id, set()).add(idx2mat[v_idx])
    return pos_by_manu, pos_by_manu_val, vocab


def load_text_fields():
    with open(ROOT / "vlm_experiment" / "text_fields.json", encoding="utf-8") as f:
        return json.load(f)


def load_image_manifest():
    with open(ROOT / "vlm_experiment" / "image_manifest.json", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(vocab, manu_text, with_image):
    candidate_block = "\n".join(f"{i+1}. {v}" for i, v in enumerate(vocab))
    img_note = (
        "You are also given up to 3 photos from this company's own website."
        if with_image else
        "No images are provided for this company."
    )
    prompt = f"""You are evaluating a manufacturer's material-processing capabilities from
a business directory record. {img_note}

Company record:
{manu_text}

Candidate material list (fixed order, {len(vocab)} items):
{candidate_block}

Task: based ONLY on the information above, select the 10 materials from the
candidate list above that this company is MOST LIKELY able to process or
work with, ranked from most to least likely. Use the exact candidate text as
written above for each item.

Respond with ONLY a JSON array of exactly 10 strings, each one an exact
candidate from the list above, ranked most-likely first, no duplicates, no
other text. Example format: ["material a", "material b", ...]"""
    return prompt


def parse_response(text, vocab_set):
    """Returns (ranked_valid_unique_list, diagnostics_dict). Never patched with gold labels."""
    diag = {"raw": text, "parse_ok": False, "n_returned_raw": 0, "n_duplicates": 0,
            "n_invalid": 0, "n_valid_unique": 0}
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        diag["error"] = "no_json_array_found"
        return [], diag
    try:
        items = json.loads(match.group(0))
    except Exception as e:
        diag["error"] = f"json_parse_failed: {e}"
        return [], diag
    if not isinstance(items, list):
        diag["error"] = "not_a_list"
        return [], diag
    diag["n_returned_raw"] = len(items)
    seen = set()
    valid_unique = []
    n_dup, n_invalid = 0, 0
    for it in items:
        if not isinstance(it, str):
            n_invalid += 1
            continue
        key = it.strip().lower()
        if key not in vocab_set:
            n_invalid += 1
            continue
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        valid_unique.append(key)
    diag["n_duplicates"] = n_dup
    diag["n_invalid"] = n_invalid
    diag["n_valid_unique"] = len(valid_unique)
    diag["parse_ok"] = True
    return valid_unique, diag


def cap_image_pixels(im, max_pixels=IMAGE_MAX_PIXELS, min_pixels=IMAGE_MIN_PIXELS):
    """Resize preserving aspect ratio so total pixel count falls in
    [min_pixels, max_pixels]. Downscales oversized real-photo uploads
    (the cause of the val dry-run's 374s/21GB outlier); leaves already-small
    images alone unless they're below the floor, in which case it upscales
    just enough to clear it (keeps a very tiny image from producing a
    degenerate/near-empty token sequence)."""
    w, h = im.size
    n = w * h
    if n > max_pixels:
        scale = (max_pixels / n) ** 0.5
    elif n < min_pixels:
        scale = (min_pixels / n) ** 0.5
    else:
        return im
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return im.resize((new_w, new_h), Image.LANCZOS)


def recall_at_k_single(ranked_list, true_positives, k):
    if not true_positives:
        return None
    top_k = set(ranked_list[:k])
    hit = len(top_k & true_positives)
    return hit / len(true_positives)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["val_dryrun", "test"], required=True)
    ap.add_argument("--conditions", nargs="+", default=["text", "text_image"])
    args = ap.parse_args()

    vocab, split = load_vocab()
    vocab_set = set(v.strip().lower() for v in vocab)
    pos_by_manu_test, pos_by_manu_val, edge_vocab = load_test_positives()
    text_fields = load_text_fields()

    if args.mode == "val_dryrun":
        with open(ROOT / "vlm_experiment" / "image_manifest_val_dryrun.json", encoding="utf-8") as f:
            manifest_data = json.load(f)
        target_manus = manifest_data["sampled_manufacturers"]
        pos_by_manu = pos_by_manu_val
        images_dir = Path(__file__).resolve().parent / "images_val_dryrun"
        manifest_entries = manifest_data["manifest"]
        print(f"VAL DRY RUN: {len(target_manus)} manufacturers, real val images "
              f"({manifest_data['n_images_ok']} extracted) -- format/VRAM/speed smoke test, "
              f"not a scored result; config must not change based on this run's output")
    else:
        manifest_data = load_image_manifest()
        # Scoped to has_image test manufacturers only (per instruction: pair
        # primarily on has_image manufacturers) -- for a no_image manufacturer,
        # the "text_image" condition would have no image to add and would
        # silently degenerate into a second, redundant text-only call, doubling
        # runtime for zero comparison value. Caught before the frozen run
        # started (0 manufacturers had been scored), not a post-hoc change.
        target_manus = sorted(e["manufacturer"] for e in manifest_data["manifest"]
                               if e["status"] == "ok")
        target_manus = sorted(set(target_manus) & set(pos_by_manu_test.keys()))
        pos_by_manu = pos_by_manu_test
        images_dir = IMAGES_DIR
        manifest_entries = manifest_data["manifest"]
        print(f"FROZEN TEST RUN: {len(target_manus)} has_image manufacturers "
              f"(text-only, no-image manufacturers are not re-run under a redundant "
              f"text_image label)")

    print(f"loading {MODEL_ID} ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.image_processor.size = {"longest_edge": IMAGE_MAX_PIXELS, "shortest_edge": IMAGE_MIN_PIXELS}
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    print(f"model loaded in {time.time()-t0:.1f}s")
    print(f"model revision/config: {model.config.to_dict().get('_name_or_path', MODEL_ID)}, "
          f"dtype={model.dtype}")

    # Crash-safe incremental save: append each result to a .jsonl immediately
    # (flushed to disk) rather than only writing one big JSON at the very
    # end. The first run of this mode lost ALL progress (96/605 manufacturers)
    # when the process was killed by an unrelated GPU-contention crash --
    # this file is both the safety net and the resume source for next time.
    jsonl_path = ROOT / "vlm_experiment" / f"vlm_results_{args.mode}.jsonl"
    done_pairs = set()
    results = []
    if jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                results.append(r)
                if "condition" in r:
                    done_pairs.add((r["manufacturer"], r["condition"]))
        print(f"RESUMING: {len(done_pairs)} (manufacturer, condition) pairs already done "
              f"in {jsonl_path.name}, will skip those")
    jsonl_f = open(jsonl_path, "a", encoding="utf-8")

    for manu in target_manus:
        if all((manu, c) in done_pairs for c in args.conditions):
            continue  # fully resumed from jsonl, skip re-running this manufacturer
        manu_text = text_fields.get(manu, {}).get("text")
        if manu_text is None:
            if (manu, "error") not in done_pairs:
                rec = {"manufacturer": manu, "error": "no_text_fields"}
                results.append(rec)
                jsonl_f.write(json.dumps(rec) + "\n"); jsonl_f.flush()
            continue
        true_pos = pos_by_manu.get(manu, set())

        manu_images = []
        entries = [e for e in manifest_entries if e.get("manufacturer") == manu and e["status"] == "ok"]
        for e in sorted(entries, key=lambda x: x["order"]):
            p = images_dir / e["file"]
            if p.exists():
                manu_images.append(cap_image_pixels(Image.open(p).convert("RGB")))

        for cond in args.conditions:
            if (manu, cond) in done_pairs:
                continue
            with_image = cond == "text_image" and len(manu_images) > 0
            prompt = build_prompt(vocab, manu_text, with_image)

            if with_image:
                content = [{"type": "image", "image": img} for img in manu_images]
                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content}]
            else:
                messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)

            torch.cuda.reset_peak_memory_stats()
            t_start = time.time()
            try:
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
                elapsed = time.time() - t_start
                peak_mem = torch.cuda.max_memory_allocated() / 1e9
                gen_text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            except Exception as e:
                # Catchable failures (e.g. CUDA OOM) get logged and skipped rather than
                # killing the whole run -- two prior runs died with NO Python traceback
                # at inconsistent points (96 then 34 manufacturers in), so this guards
                # the subset of future failures that ARE catchable; it cannot help with
                # a true native/driver-level crash, which is what resume + this
                # try/except together are meant to make merely costly, not fatal.
                elapsed = time.time() - t_start
                peak_mem = torch.cuda.max_memory_allocated() / 1e9
                rec = {"manufacturer": manu, "condition": cond, "error": f"generate_failed: {e}",
                       "elapsed_sec": elapsed, "peak_mem_gb": peak_mem}
                results.append(rec)
                jsonl_f.write(json.dumps(rec) + "\n"); jsonl_f.flush()
                print(f"  {manu[:40]:40s} [{cond:10s}] GENERATE FAILED: {e}")
                torch.cuda.empty_cache()
                continue

            ranked, diag = parse_response(gen_text, vocab_set)

            r5 = recall_at_k_single(ranked, true_pos, 5)
            r10 = recall_at_k_single(ranked, true_pos, 10)

            rec = {
                "manufacturer": manu, "condition": cond, "with_image_actual": with_image,
                "n_images_used": len(manu_images) if with_image else 0,
                "n_true_positives": len(true_pos),
                "recall5": r5, "recall10": r10,
                "elapsed_sec": elapsed, "peak_mem_gb": peak_mem,
                "diag": diag,
            }
            results.append(rec)
            jsonl_f.write(json.dumps(rec) + "\n"); jsonl_f.flush()
            print(f"  {manu[:40]:40s} [{cond:10s}] R5={r5} R10={r10} "
                  f"parse_ok={diag['parse_ok']} n_valid={diag['n_valid_unique']} "
                  f"t={elapsed:.2f}s mem={peak_mem:.2f}GB")
        torch.cuda.empty_cache()  # reduce fragmentation risk across ~600 iterations

    jsonl_f.close()
    out_path = ROOT / "vlm_experiment" / f"vlm_results_{args.mode}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "mode": args.mode, "model_id": MODEL_ID, "seed": SEED,
            "max_new_tokens": MAX_NEW_TOKENS, "n_manufacturers": len(target_manus),
            "image_max_pixels": IMAGE_MAX_PIXELS, "image_min_pixels": IMAGE_MIN_PIXELS,
            "dtype": str(model.dtype), "do_sample": False,
            "results": results,
        }, f, indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
