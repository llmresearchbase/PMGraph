#!/usr/bin/env python3
"""
Hit@1 for the VLM track, computed from already-saved raw outputs
(vlm_results_test.json's diag["raw"] field) by re-parsing with the SAME
parse_response() already used to score Recall@5/@10 -- genuinely zero
rerun, no new model calls. Post-hoc metric (test results already seen);
supplements, does not replace, the primary R@5/@10 result.

PRIMARY scoring (full 605-manufacturer population, every condition on the
identical denominator): a manufacturer with no valid ranked candidate under
a condition counts as a MISS (hit1=0) for that condition, not an exclusion
-- the only way every condition is directly comparable by denominator. An
earlier version of this script excluded TEXT+IMAGE's zero-valid cases
instead, which is reported below as a SECONDARY view, not the primary one.
"""
import json
from pathlib import Path

import numpy as np

import run_vlm_eval as rv

ROOT = Path(__file__).resolve().parent


def bootstrap_ci(arrs, n_boot=2000, seed=777):
    rng = np.random.RandomState(seed)
    n = len(next(iter(arrs.values())))
    boots = {k: np.empty(n_boot) for k in arrs}
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        for k, a in arrs.items():
            boots[k][i] = a[idx].mean()
    return boots


def ci(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    with open(ROOT / "vlm_results_test.json", encoding="utf-8") as f:
        data = json.load(f)

    vocab, split = rv.load_vocab()
    vocab_set = set(v.strip().lower() for v in vocab)
    pos_by_manu_test, _, _ = rv.load_test_positives()

    # hit1_primary: miss (0) when no valid ranked candidate. hit1_excl: None when no valid candidate.
    by_manu_primary = {"text": {}, "text_image": {}}
    by_manu_excl = {"text": {}, "text_image": {}}
    for r in data["results"]:
        if "error" in r or "condition" not in r:
            continue
        cond = r["condition"]
        raw = r["diag"]["raw"]
        ranked, diag2 = rv.parse_response(raw, vocab_set)
        true_pos = pos_by_manu_test.get(r["manufacturer"], set())
        if not true_pos:
            continue
        hit1_primary = 1 if (ranked and ranked[0] in true_pos) else 0
        by_manu_primary[cond][r["manufacturer"]] = hit1_primary
        if ranked:
            by_manu_excl[cond][r["manufacturer"]] = hit1_primary

    print("=== PRIMARY: full population, zero-valid scored as miss ===")
    manus_primary = sorted(set(by_manu_primary["text"]) & set(by_manu_primary["text_image"]))
    t_p = np.array([by_manu_primary["text"][m] for m in manus_primary])
    ti_p = np.array([by_manu_primary["text_image"][m] for m in manus_primary])
    print(f"  n={len(manus_primary)}  TEXT Hit@1={t_p.mean():.4f}  TEXT+IMAGE Hit@1={ti_p.mean():.4f}")
    d_p = ti_p - t_p
    print(f"  paired diff mean={d_p.mean():+.4f} (sample std {d_p.std(ddof=1):.4f})")
    boots_p = bootstrap_ci({"text": t_p, "text_image": ti_p, "d": ti_p - t_p})
    print(f"  bootstrap TEXT={boots_p['text'].mean():.4f} CI={ci(boots_p['text'])}")
    print(f"  bootstrap TEXT+IMAGE={boots_p['text_image'].mean():.4f} CI={ci(boots_p['text_image'])}")
    print(f"  bootstrap diff={boots_p['d'].mean():+.4f} CI={ci(boots_p['d'])}")

    print("\n=== SECONDARY: matched subset, zero-valid excluded (not scored as miss) ===")
    manus_excl = sorted(set(by_manu_excl["text"]) & set(by_manu_excl["text_image"]))
    t_e = np.array([by_manu_excl["text"][m] for m in manus_excl])
    ti_e = np.array([by_manu_excl["text_image"][m] for m in manus_excl])
    print(f"  n={len(manus_excl)}  TEXT Hit@1={t_e.mean():.4f}  TEXT+IMAGE Hit@1={ti_e.mean():.4f}")
    boots_e = bootstrap_ci({"text": t_e, "text_image": ti_e, "d": ti_e - t_e})
    print(f"  bootstrap TEXT={boots_e['text'].mean():.4f} CI={ci(boots_e['text'])}")
    print(f"  bootstrap TEXT+IMAGE={boots_e['text_image'].mean():.4f} CI={ci(boots_e['text_image'])}")
    print(f"  bootstrap diff={boots_e['d'].mean():+.4f} CI={ci(boots_e['d'])}")

    with open(ROOT / "vlm_hit_at_1_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "note": "post-hoc metric, computed after R5/R10 test results were seen; "
                     "zero rerun -- re-parses already-saved raw model output text. "
                     "PRIMARY scores zero-valid-response manufacturers as misses (full "
                     "605-pop, consistent denominator across conditions); SECONDARY "
                     "excludes them (matched subset) -- listed for transparency, not primary.",
            "primary_full_population": {
                "n": len(manus_primary),
                "hit1_text": float(t_p.mean()), "hit1_text_image": float(ti_p.mean()),
                "paired_diff_mean": float(d_p.mean()), "paired_diff_std": float(d_p.std(ddof=1)),
                "bootstrap_95ci": {"hit1_text": list(ci(boots_p["text"])),
                                   "hit1_text_image": list(ci(boots_p["text_image"])),
                                   "d_hit1": list(ci(boots_p["d"]))},
            },
            "secondary_matched_subset_excluding_zero_valid": {
                "n": len(manus_excl),
                "hit1_text": float(t_e.mean()), "hit1_text_image": float(ti_e.mean()),
                "bootstrap_95ci": {"hit1_text": list(ci(boots_e["text"])),
                                   "hit1_text_image": list(ci(boots_e["text_image"])),
                                   "d_hit1": list(ci(boots_e["d"]))},
            },
        }, f, indent=1)
    print("\nwrote vlm_hit_at_1_results.json")


if __name__ == "__main__":
    main()
