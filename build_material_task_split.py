#!/usr/bin/env python3
"""
Manufacturer-level split + frozen material vocabulary for the "recorded
material-relation recovery" pilot.

- Manufacturers with >=1 recorded Material_Capability value are eligible for
  this task's supervised train/val/test; manufacturers with the field
  entirely empty are KEPT in the resource (still graph nodes) but excluded
  from this task's supervised split entirely (not used as train negatives,
  not evaluated).
- Split is by MANUFACTURER identity (not by edge), fixed seed, 70/15/15.
- Material vocabulary (candidate prediction classes) is selected using
  ONLY the train-split manufacturers' positive counts (>=50 threshold),
  then frozen -- val/test are never consulted for vocabulary selection.
"""
import csv
import random
import json

SEED = 42
MIN_TRAIN_POSITIVES = 50
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # test = remainder


def canon(w):
    w = str(w).strip().lower()
    for p in ("https://", "http://"):
        if w.startswith(p):
            w = w[len(p):]
    return w.rstrip("/")


def main():
    with open("Cleaned_Master_CSV_v3.csv", newline='', encoding='utf-8-sig', errors='replace') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        rows = list(reader)

    mat_cols = [c for c in cols if c.startswith("Material_Capability_")]
    prod_cols = [f"Product_{i}" for i in range(1, 151)]

    graph_nodes = set()
    for r in rows:
        m = canon(r.get("Web", ""))
        for c in prod_cols:
            p = r.get(c)
            if p and p.strip() and p.strip().lower() != "yes":
                graph_nodes.add(m)

    manu_materials = {}  # web -> set of recorded material values (lowercased)
    for r in rows:
        web = canon(r.get("Web", ""))
        if web not in graph_nodes:
            continue
        vals = set(str(r.get(c)).strip().lower() for c in mat_cols if r.get(c) and str(r.get(c)).strip())
        manu_materials[web] = vals

    eligible = sorted([w for w, v in manu_materials.items() if v])  # has >=1 recorded material
    excluded_empty = sorted([w for w, v in manu_materials.items() if not v])
    print(f"graph nodes: {len(graph_nodes)}")
    print(f"eligible (>=1 recorded material): {len(eligible)}")
    print(f"excluded from this task (material field entirely empty, kept in resource): {len(excluded_empty)}")

    rng = random.Random(SEED)
    shuffled = list(eligible)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train_manus = sorted(shuffled[:n_train])
    val_manus = sorted(shuffled[n_train:n_train + n_val])
    test_manus = sorted(shuffled[n_train + n_val:])
    print(f"\nmanufacturer-level split: train={len(train_manus)} val={len(val_manus)} test={len(test_manus)}")

    # vocabulary selection: train-set positive counts ONLY
    train_set = set(train_manus)
    from collections import Counter
    train_pos_count = Counter()
    for w in train_manus:
        for v in manu_materials[w]:
            train_pos_count[v] += 1

    selected_vocab = sorted([v for v, c in train_pos_count.items() if c >= MIN_TRAIN_POSITIVES])
    print(f"\nmaterial vocabulary: {len(train_pos_count)} distinct values seen in train; "
          f"{len(selected_vocab)} selected at >={MIN_TRAIN_POSITIVES}-train-positive threshold")

    # fraction of all positives (across train+val+test eligible manufacturers) retained by this vocab
    total_pos_all = sum(len(manu_materials[w]) for w in eligible)
    retained_pos_all = sum(1 for w in eligible for v in manu_materials[w] if v in selected_vocab)
    print(f"positive-instance retention: {retained_pos_all}/{total_pos_all} "
          f"({100*retained_pos_all/total_pos_all:.1f}%) of all recorded (manufacturer,material) pairs "
          f"fall within the selected vocabulary")

    # per-selected-category counts in val/test, for the "don't over-interpret n<10" flag later
    val_set, test_set = set(val_manus), set(test_manus)
    cat_counts = {}
    for v in selected_vocab:
        n_train_c = sum(1 for w in train_manus if v in manu_materials[w])
        n_val_c = sum(1 for w in val_manus if v in manu_materials[w])
        n_test_c = sum(1 for w in test_manus if v in manu_materials[w])
        cat_counts[v] = {"train": n_train_c, "val": n_val_c, "test": n_test_c}

    low_val_test = [v for v, c in cat_counts.items() if c["val"] < 10 or c["test"] < 10]
    print(f"\ncategories with <10 positives in val OR test (flag, don't individually interpret): {len(low_val_test)}")

    with open("material_task_split.json", "w", encoding="utf-8") as f:
        json.dump({
            "seed": SEED,
            "min_train_positives_threshold": MIN_TRAIN_POSITIVES,
            "train_manufacturers": train_manus,
            "val_manufacturers": val_manus,
            "test_manufacturers": test_manus,
            "excluded_empty_material_field": excluded_empty,
            "selected_vocab": selected_vocab,
            "category_counts": cat_counts,
            "categories_flagged_low_val_test": low_val_test,
            "positive_retention_pct": 100 * retained_pos_all / total_pos_all,
        }, f, indent=2)
    print("\nsaved material_task_split.json")

    print("\nselected vocab with (train/val/test) counts:")
    for v in selected_vocab:
        c = cat_counts[v]
        flag = "  <-- LOW val/test, flagged" if v in low_val_test else ""
        print(f"  {v:<20} train={c['train']:>4} val={c['val']:>3} test={c['test']:>3}{flag}")


if __name__ == "__main__":
    main()
