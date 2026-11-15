---
license: cc-by-4.0
pretty_name: PMGraph
language:
- en
tags:
- knowledge-graph
- manufacturing
- supply-chain
- link-prediction
- multimodal
- benchmark
task_categories:
- graph-ml
size_categories:
- 10K<n<100K
configs:
- config_name: directory
  data_files: Cleaned_Master_CSV_v3.csv
---

> **This is PMGraph v2** (corrected and audited). Code and reproduction
> steps: https://github.com/llmresearchbase/PMGraph (see `REPRODUCE.md` here for
> the same instructions). The original C-MAG-era release is preserved at
> revision [`v1`](https://huggingface.co/datasets/shawntzx/PMGraph/tree/v1);
> `CHANGELOG.md` lists what changed.

## Files

| File | Contents |
|---|---|
| `Cleaned_Master_CSV_v3.csv` | Source manufacturer directory (14,007 records) |
| `hetero_graphs/Graph_A.pt` | The manufacturer–product heterogeneous graph (PyG) |
| `frozen_split_GraphA_seed42.pt` | Task A (relation completion) frozen split |
| `material_task_split.json`, `material_edges.pt` | Task B (material recovery) frozen split and edges |
| `manufacturer_features_no_material.pt` | Task B manufacturer features |
| `manufacturer_image_release_asset.pt` | Image embeddings + website domains (no raw pixels) |
| `manufacturer_validity_mask.pt`, `excluded_entries_v1.json` | Disclosed data-quality issues |
| `manufacturer_coverage_flags.pt`, `manufacturer_order_for_material_task.json` | Image-coverage flags and Task B manufacturer order |

```bash
hf download shawntzx/PMGraph --repo-type dataset --revision v2 --local-dir PMGraph-data
```

