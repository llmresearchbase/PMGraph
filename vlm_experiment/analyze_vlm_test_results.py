#!/usr/bin/env python3
"""
Analyzes vlm_results_test.json: absolute Recall@5/@10 per condition, paired
per-manufacturer differences, manufacturer-level bootstrap CIs, parse-failure
/diagnostic rates, and latency/VRAM summary. Never patches missing/invalid
model output with gold labels -- manufacturers with a parse failure are
excluded from that condition's scored set and counted explicitly.
"""
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 777


def main():
    with open(ROOT / "vlm_results_test.json", encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"]

    by_manu_cond = {}
    for r in results:
        if "error" in r:
            continue
        by_manu_cond[(r["manufacturer"], r["condition"])] = r

    manus = sorted(set(m for m, c in by_manu_cond))
    conditions = sorted(set(c for m, c in by_manu_cond))
    print(f"manufacturers: {len(manus)}  conditions: {conditions}")

    # ---- diagnostics: parse failure / invalid / duplicate rates ----
    print("\n=== diagnostics per condition ===")
    diag_summary = {}
    for cond in conditions:
        rows = [by_manu_cond[(m, cond)] for m in manus if (m, cond) in by_manu_cond]
        n = len(rows)
        n_parse_fail = sum(1 for r in rows if not r["diag"]["parse_ok"])
        n_zero_valid = sum(1 for r in rows if r["diag"]["parse_ok"] and r["diag"]["n_valid_unique"] == 0)
        avg_invalid = np.mean([r["diag"]["n_invalid"] for r in rows])
        avg_dup = np.mean([r["diag"]["n_duplicates"] for r in rows])
        avg_valid = np.mean([r["diag"]["n_valid_unique"] for r in rows])
        avg_latency = np.mean([r["elapsed_sec"] for r in rows])
        max_latency = np.max([r["elapsed_sec"] for r in rows])
        avg_mem = np.mean([r["peak_mem_gb"] for r in rows])
        max_mem = np.max([r["peak_mem_gb"] for r in rows])
        diag_summary[cond] = dict(n=n, n_parse_fail=n_parse_fail, n_zero_valid=n_zero_valid,
                                   avg_invalid=avg_invalid, avg_dup=avg_dup, avg_valid=avg_valid,
                                   avg_latency=avg_latency, max_latency=max_latency,
                                   avg_mem=avg_mem, max_mem=max_mem)
        print(f"  [{cond}] n={n} parse_fail={n_parse_fail} ({n_parse_fail/n*100:.1f}%) "
              f"zero_valid={n_zero_valid} ({n_zero_valid/n*100:.1f}%) "
              f"avg_valid={avg_valid:.2f}/10 avg_invalid={avg_invalid:.2f} avg_dup={avg_dup:.2f} "
              f"latency_avg={avg_latency:.2f}s max={max_latency:.2f}s "
              f"mem_avg={avg_mem:.2f}GB max={max_mem:.2f}GB")

    # ---- absolute Recall@5/@10 per condition (only scorable manufacturers: n_true_positives>0) ----
    print("\n=== absolute Recall@K per condition (scorable manufacturers only) ===")
    recall_summary = {}
    for cond in conditions:
        r5s, r10s = [], []
        for m in manus:
            r = by_manu_cond.get((m, cond))
            if r is None or r["recall5"] is None:
                continue
            r5s.append(r["recall5"]); r10s.append(r["recall10"])
        recall_summary[cond] = {"r5": np.array(r5s), "r10": np.array(r10s)}
        print(f"  [{cond}] n_scored={len(r5s)} Recall@5={np.mean(r5s):.4f} Recall@10={np.mean(r10s):.4f}")

    # ---- paired per-manufacturer differences: text_image - text ----
    print("\n=== paired differences: text_image - text (manufacturers scorable in BOTH) ===")
    paired_manus = []
    d5, d10 = [], []
    for m in manus:
        a = by_manu_cond.get((m, "text"))
        b = by_manu_cond.get((m, "text_image"))
        if a is None or b is None or a["recall5"] is None or b["recall5"] is None:
            continue
        paired_manus.append(m)
        d5.append(b["recall5"] - a["recall5"])
        d10.append(b["recall10"] - a["recall10"])
    d5, d10 = np.array(d5), np.array(d10)
    print(f"  n paired = {len(paired_manus)}")
    print(f"  mean dRecall@5  = {d5.mean():+.4f}  (sample std {d5.std(ddof=1):.4f})")
    print(f"  mean dRecall@10 = {d10.mean():+.4f}  (sample std {d10.std(ddof=1):.4f})")

    # ---- manufacturer-level bootstrap CI (95%) ----
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    n = len(paired_manus)
    boot_means_d5, boot_means_d10 = [], []
    boot_r5_text, boot_r10_text, boot_r5_ti, boot_r10_ti = [], [], [], []
    idx_array = np.arange(n)
    r5_text_arr = np.array([by_manu_cond[(m, "text")]["recall5"] for m in paired_manus])
    r10_text_arr = np.array([by_manu_cond[(m, "text")]["recall10"] for m in paired_manus])
    r5_ti_arr = np.array([by_manu_cond[(m, "text_image")]["recall5"] for m in paired_manus])
    r10_ti_arr = np.array([by_manu_cond[(m, "text_image")]["recall10"] for m in paired_manus])
    for _ in range(N_BOOTSTRAP):
        sample_idx = rng.choice(idx_array, size=n, replace=True)
        boot_means_d5.append((r5_ti_arr[sample_idx] - r5_text_arr[sample_idx]).mean())
        boot_means_d10.append((r10_ti_arr[sample_idx] - r10_text_arr[sample_idx]).mean())
        boot_r5_text.append(r5_text_arr[sample_idx].mean())
        boot_r10_text.append(r10_text_arr[sample_idx].mean())
        boot_r5_ti.append(r5_ti_arr[sample_idx].mean())
        boot_r10_ti.append(r10_ti_arr[sample_idx].mean())

    def ci(arr):
        return np.percentile(arr, 2.5), np.percentile(arr, 97.5)

    print(f"\n=== manufacturer-level bootstrap 95% CI (n_bootstrap={N_BOOTSTRAP}, seed={BOOTSTRAP_SEED}) ===")
    print(f"  Recall@5  TEXT:       {np.mean(boot_r5_text):.4f}  95% CI {ci(boot_r5_text)}")
    print(f"  Recall@5  TEXT+IMAGE: {np.mean(boot_r5_ti):.4f}  95% CI {ci(boot_r5_ti)}")
    print(f"  Recall@10 TEXT:       {np.mean(boot_r10_text):.4f}  95% CI {ci(boot_r10_text)}")
    print(f"  Recall@10 TEXT+IMAGE: {np.mean(boot_r10_ti):.4f}  95% CI {ci(boot_r10_ti)}")
    print(f"  dRecall@5  (TI-TEXT): {np.mean(boot_means_d5):+.4f}  95% CI {ci(boot_means_d5)}")
    print(f"  dRecall@10 (TI-TEXT): {np.mean(boot_means_d10):+.4f}  95% CI {ci(boot_means_d10)}")

    out = {
        "n_manufacturers_test_set": len(manus),
        "n_paired_scorable": len(paired_manus),
        "diagnostics": diag_summary,
        "absolute_recall": {c: {"n": len(v["r5"]), "recall5": float(v["r5"].mean()),
                                  "recall10": float(v["r10"].mean())} for c, v in recall_summary.items()},
        "paired_diff_text_image_minus_text": {
            "n": len(paired_manus),
            "mean_dRecall5": float(d5.mean()), "std_dRecall5": float(d5.std(ddof=1)),
            "mean_dRecall10": float(d10.mean()), "std_dRecall10": float(d10.std(ddof=1)),
        },
        "bootstrap_95ci": {
            "n_bootstrap": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED,
            "recall5_text": [float(x) for x in ci(boot_r5_text)],
            "recall5_text_image": [float(x) for x in ci(boot_r5_ti)],
            "recall10_text": [float(x) for x in ci(boot_r10_text)],
            "recall10_text_image": [float(x) for x in ci(boot_r10_ti)],
            "d_recall5": [float(x) for x in ci(boot_means_d5)],
            "d_recall10": [float(x) for x in ci(boot_means_d10)],
        },
    }
    with open(ROOT / "vlm_test_analysis.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\nwrote vlm_test_analysis.json")


if __name__ == "__main__":
    main()
