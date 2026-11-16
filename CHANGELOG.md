# PMGraph — Version Changes and Maintenance Plan

## v2 (this submission) vs. the original C-MAG-era release (HuggingFace `shawntzx/PMGraph`)

| Area | Original release | This release |
|---|---|---|
| Entity key | Name-keyed; 31 groups of distinct companies were silently merged by name collisions | Domain-keyed; the 31 collision groups are split correctly, verified collision-free across all 14,007 directory rows |
| Task A hyperparameter selection | Learning rate selected by **test** ROC-AUC (a methodology bug — contradicts the paper's own stated validation-based procedure) | Learning rate selected by **validation** ROC-AUC only; test is scored once after selection |
| Task A negative sampling | Single protocol reported | Both random-negative and degree-matched-negative protocols reported, since they give materially different numbers for the same checkpoint |
| Benchmark tasks | One (manufacturer–product relation completion) | Two (adds recorded-material-relation recovery, with a frozen 56-category vocabulary and manufacturer-level split) |
| Baselines | Not systematically reported | Frequency, degree, shared-space text-similarity, feature-only MLP, and GraphSAGE/GNN baselines reported for both tasks |
| Image fusion evidence | Not evaluated | Two independent fusion designs tested (one replicated by an independent full rerun; the other's comparison to it is confounded, disclosed as such), plus a zero-shot VLM comparison (Qwen3-VL-2B-Instruct) |
| Data-quality audits | Not performed | Full-population edge/label-set reconstruction audit, a targeted contamination screen (3 non-manufacturer nodes + 3 artifact edges found and disclosed), and a 200-item stratified human image-relevance audit (two annotators, Cohen's κ=0.79 on relevance) |
| Validity mask | Not provided | `manufacturer_validity_mask.pt` released at the current (non-reindexed) node/edge indexing; its effect on main results was checked directly (≤0.0001 AUC, ≤0.0021 Recall@K) rather than left unquantified |
| License | Not set (checked directly against the HF dataset's own metadata) | CC BY 4.0 (see `LICENSE`) |
| Documentation | Dataset card only | Dataset card, this changelog, a dedicated provenance/ethics statement (`PROVENANCE_AND_ETHICS.md`), and a reproduction README with an independently verified quick-start |

## Maintenance plan

- **Versioning**: this release is tagged as PMGraph v2 relative to the
  original C-MAG-era HuggingFace release (treated as v1, though it was
  never explicitly versioned). Future corrected releases should bump this
  tag and add a new row to the table above rather than silently overwrite
  v2's numbers — every result in the accompanying paper is pinned to v2.
- **Known-issue tracking**: `excluded_entries_v1.json` and
  `manufacturer_validity_mask.pt` are the canonical records of disclosed
  data-quality issues as of v2. A future v3 that applies the mask (i.e.
  rebuilds the graph with reindexing) should keep both the pre- and
  post-mask node/edge counts documented, since downstream frozen splits
  (`frozen_split_GraphA_seed42.pt`, `material_task_split.json`) are
  index-dependent and would need to be regenerated, not reused, if
  reindexing ever happens.
- **Reproducibility caveats that persist across versions**: Task A's SAGE
  training has a documented cross-environment (CPU/GPU) training-time
  non-determinism (Section 5.1 of the paper); this is a property of the
  model/training stack, not something a future data version fixes.
- **Contact for issues**: open an issue at https://github.com/llmresearchbase/PMGraph/issues, or email
  yli59@lenovo.com.
- **Where each version lives**: v2 data is at https://huggingface.co/datasets/shawntzx/PMGraph
  (git tag `v2`), v2 code at https://github.com/llmresearchbase/PMGraph (tag `v2.0`). The original
  C-MAG-era release, which included raw website images, was withdrawn from
  HuggingFace on 2026-11-15 (repository history removed) to match this
  release's no-pixel image policy; see `PROVENANCE_AND_ETHICS.md`.
