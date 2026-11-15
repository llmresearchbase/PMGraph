# Experiment 1: grouped evaluation + simple baselines (Graph_A, frozen split, seed=42)

**SUPERSEDED NOTICE (round-22 correction): the "AUC (text-similarity baseline)"
column below is INVALID.** It computes cosine similarity between
`data['manufacturer'].x` and `data['product'].x`, which are two
INDEPENDENTLY-fit TF-IDF+SVD spaces (confirmed by reading
`Graph-A-Builder.py` directly) -- not a shared embedding, so cosine
similarity between them carries no real semantic meaning. The `degree`
column is unaffected and remains valid. See
`experiment1_textsim_shared_space_results.md` for the corrected baseline
(one shared TF-IDF vocabulary over manufacturer + product text, no SVD).

Generic-term threshold: products shared by >= 20 manufacturers (declared audit rule, not a statistical cutoff -- see script docstring).

| Bucket | n | pos rate | AUC (product-degree baseline) | AUC (text-similarity baseline) |
|---|---|---|---|---|
| 1_singleton | 16041 | 0.3891 | 0.0769 | 0.5070 |
| 2-5 | 3287 | 0.6459 | 0.3241 | 0.5180 |
| 6-19 | 1672 | 0.8559 | 0.5177 | 0.5269 |
| high_frequency_20+ | 1518 | 0.9638 | 0.6683 | 0.5328 |
| **ALL** | 22518 | 0.5000 | 0.4177 | 0.5102 |
