#!/usr/bin/env python3
"""
Task B addendum (user-directed, priority 2 of the "what does this benchmark
actually discriminate" round): does the material-recovery task mainly test
recovery of a text clue already visible elsewhere, or genuine cross-field /
structural inference?

For each TEST positive (manufacturer, material) pair, checks whether that
specific material's name is word-boundary-mentioned in the manufacturer's
own VISIBLE text fields (product strings, business description, NAICS
description -- NEVER the Material_Capability field itself, which is the
masked target). Groups PER LABEL, not per manufacturer, since a
manufacturer can have several true materials and only some may be
text-visible -- a per-manufacturer "mentioned/not" tag would blur this.

For each group, reports a per-label hit rate: is THIS specific material
within the model's top-5/top-10 ranked predictions for that manufacturer --
not the usual per-manufacturer Recall@K (which aggregates across a
manufacturer's full positive set), since that's the wrong unit for this
question.

Reuses already-established models/features -- frequency and the
attributes-only MLP (single seed, same config already selected and
archived in material_recovery_mlp_baseline_results.md, not a fresh grid
search) -- exactly as instructed: analyze on an existing-protocol model,
no new training regime invented for this.
"""
import importlib.util
import re
from collections import defaultdict

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("mb", "material_recovery_mlp_baseline.py")
mb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mb)

# archived-selected config for seed 42, from material_recovery_mlp_baseline_results.md
ARCHIVED_LR = 1.0e-03
SEED = 42


def canon(w):
    w = str(w).strip().lower()
    for p in ("https://", "http://"):
        if w.startswith(p):
            w = w[len(p):]
    return w.rstrip("/")


def build_visible_text_by_manufacturer(manus_sorted):
    """Product strings + Business_Description + NAICS_Description_1 ONLY --
    never Material_Capability_* (that's the masked target field)."""
    m2i = {m: i for i, m in enumerate(manus_sorted)}
    text_by_idx = defaultdict(str)
    import csv
    with open("Cleaned_Master_CSV_v3.csv", newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        product_cols = [f"Product_{i}" for i in range(1, 151)]
        for r in reader:
            m = canon(r.get("Web", ""))
            if m not in m2i:
                continue
            idx = m2i[m]
            parts = [r.get("Business_Description", "") or "", r.get("NAICS_Description_1", "") or ""]
            for c in product_cols:
                v = r.get(c)
                if v and v.strip() and v.strip().lower() != "yes":
                    parts.append(v)
            text_by_idx[idx] += " " + " ".join(parts)
    return text_by_idx


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edges = torch.load("material_edges.pt", map_location="cpu", weights_only=False)
    cov = torch.load("manufacturer_coverage_flags.pt", map_location="cpu", weights_only=False)
    has_image = cov["has_image"].numpy().astype(bool)
    feat = torch.load("manufacturer_features_no_material.pt", map_location=device, weights_only=False)
    vocab = edges["vocab"]
    manus_sorted = edges["manus_sorted"]

    print("Building visible-text index per manufacturer (product strings + business/NAICS description, "
          "NEVER Material_Capability fields)...")
    text_by_idx = build_visible_text_by_manufacturer(manus_sorted)

    test_m = edges["test_manu_idx"].tolist()
    test_v = edges["test_mat_idx"].tolist()
    print(f"test positive (manufacturer, material) pairs: {len(test_m)}")

    # ---- per-pair: is this specific material word-boundary-mentioned in visible text? ----
    mentioned = []
    compiled = {vi: re.compile(r"\b" + re.escape(vocab[vi]) + r"\b", re.IGNORECASE) for vi in set(test_v)}
    for m, v in zip(test_m, test_v):
        txt = text_by_idx.get(m, "")
        mentioned.append(bool(compiled[v].search(txt)))
    mentioned = np.array(mentioned)
    print(f"mentioned in visible text: {mentioned.sum()} / {len(mentioned)} ({100*mentioned.mean():.1f}%)")
    print(f"NOT mentioned: {(~mentioned).sum()} / {len(mentioned)} ({100*(~mentioned).mean():.1f}%)")

    # ---- frequency baseline: category popularity rank in TRAIN ----
    train_v = edges["train_mat_idx"].tolist()
    freq = np.zeros(len(vocab))
    for v in train_v:
        freq[v] += 1
    freq_rank = (-freq).argsort().argsort()  # rank 0 = most frequent

    def hit_at_k_freq(v, k):
        return freq_rank[v] < k

    # ---- attributes-only MLP: single seed, archived-selected LR, no new grid search ----
    print(f"\nTraining attributes-only MLP once (seed={SEED}, lr={ARCHIVED_LR:.1e}, "
          f"matching material_recovery_mlp_baseline_results.md's selected config for this seed)...")
    model, head, val_r10, stop_ep = mb.train_one_mlp(feat, device, edges, ARCHIVED_LR, SEED)
    model.eval(); head.eval()
    print(f"  val_Recall@10={val_r10:.4f}  stop_epoch={stop_ep}")
    with torch.no_grad():
        manu_h = model(feat)
        all_scores = head.score_all(manu_h).cpu().numpy()  # [n_manu, n_vocab]
    mlp_rank = np.argsort(-all_scores, axis=1).argsort(axis=1)  # per-manufacturer rank of each category

    def hit_at_k_mlp(m, v, k):
        return mlp_rank[m, v] < k

    # ---- grouped hit rates, PER LABEL (not per-manufacturer aggregate) ----
    groups = {"mentioned_in_text": mentioned, "not_mentioned_in_text": ~mentioned}
    rows = []
    for gname, gmask in groups.items():
        n = int(gmask.sum())
        freq_hit5 = np.mean([hit_at_k_freq(test_v[i], 5) for i in range(len(test_m)) if gmask[i]])
        freq_hit10 = np.mean([hit_at_k_freq(test_v[i], 10) for i in range(len(test_m)) if gmask[i]])
        mlp_hit5 = np.mean([hit_at_k_mlp(test_m[i], test_v[i], 5) for i in range(len(test_m)) if gmask[i]])
        mlp_hit10 = np.mean([hit_at_k_mlp(test_m[i], test_v[i], 10) for i in range(len(test_m)) if gmask[i]])
        rows.append((gname, n, freq_hit5, freq_hit10, mlp_hit5, mlp_hit10))
        print(f"\n[{gname}] n={n}")
        print(f"  Frequency: hit@5={freq_hit5:.4f} hit@10={freq_hit10:.4f}")
        print(f"  MLP:       hit@5={mlp_hit5:.4f} hit@10={mlp_hit10:.4f}")

    with open("experiment_task_b_text_mention_grouping_results.md", "w", encoding="utf-8") as f:
        f.write("# Task B: test positives grouped by whether the target material is text-visible\n\n")
        f.write(
            "Per-POSITIVE-LABEL grouping (not per-manufacturer): each test (manufacturer, "
            "material) positive pair is tagged by whether that specific material's name is "
            "word-boundary-mentioned in the manufacturer's own visible text (product strings, "
            "business description, NAICS description) -- never the masked Material_Capability "
            "field itself. Metric is per-label hit rate (is this one material within the "
            "model's top-K for that manufacturer), not the usual per-manufacturer Recall@K.\n\n"
        )
        f.write(f"Total test positives: {len(test_m)}. Mentioned in visible text: "
                f"{mentioned.sum()} ({100*mentioned.mean():.1f}%). Not mentioned: "
                f"{(~mentioned).sum()} ({100*(~mentioned).mean():.1f}%).\n\n")
        f.write(f"Attributes-only MLP: single seed={SEED}, lr={ARCHIVED_LR:.1e} "
                f"(archived-selected config, not a fresh grid search), val_Recall@10={val_r10:.4f}, "
                f"stop_epoch={stop_ep}.\n\n")
        f.write("| Group | n | Frequency hit@5 | Frequency hit@10 | MLP hit@5 | MLP hit@10 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for gname, n, f5, f10, m5, m10 in rows:
            f.write(f"| {gname} | {n} | {f5:.4f} | {f10:.4f} | {m5:.4f} | {m10:.4f} |\n")

    print("\nwrote experiment_task_b_text_mention_grouping_results.md")


if __name__ == "__main__":
    main()
