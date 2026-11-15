#!/usr/bin/env python3
"""
User-directed addendum to Section 3's data-quality evidence: the existing
checks (build-reconstruction, label-set coverage, the targeted contamination
screen in Section 3.5) are either full-population exact checks or a
TARGETED heuristic screen (free-hosting-platform domains) -- none of them
is a random sample of ordinary manufacturer-product RELATIONS themselves.
Per this project's own standing lesson (a targeted/top-similarity sample
reveals failure modes but never a citable population error rate), this
draws an unbiased random sample of (manufacturer, product) edges and
reports plausibility judgments, methodology, and error types -- not just
another targeted screen.

Reproduces the deterministic sample only; the judgments themselves (which
require reading each record, and for 3 borderline/suspicious cases, the
live manufacturer website) are recorded by hand in
random_relation_audit_results.md, not re-derived by this script.
"""
import csv
import random

SEED = 2026
N = 20


def main():
    edges = []
    with open("Cleaned_Master_CSV_v3.csv", newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        cols = [f"Product_{i}" for i in range(1, 151)]
        for r in reader:
            web = r.get("Web", "")
            for c in cols:
                p = r.get(c)
                if p and p.strip() and p.strip().lower() != "yes":
                    edges.append((web, r.get("Name", ""), r.get("NAICS_Description_1", ""),
                                  r.get("Industry_1", ""), p.strip()))

    print(f"total manufacturer-product edges (raw CSV rows, before entity-ID dedup): {len(edges)}")
    rng = random.Random(SEED)
    sample = rng.sample(edges, N)
    for i, (web, name, naics, ind1, prod) in enumerate(sample):
        print(f"--- {i+1} ---")
        print(f"  web: {web}")
        print(f"  name: {name}")
        print(f"  naics_desc: {naics}")
        print(f"  industry_1: {ind1}")
        print(f"  product: {prod}")


if __name__ == "__main__":
    main()
