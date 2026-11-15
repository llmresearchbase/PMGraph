# Corrected (shared-space) text-similarity baseline, Task A

Supersedes the `textsim` column in experiment1_baseline_results.md / experiment1_degree_matched_results.md, which computed cosine similarity between two INDEPENDENTLY-fit SVD spaces (manufacturer vs. product) -- not a valid similarity measure. This version fits ONE shared TF-IDF vocabulary over manufacturer text (Name + NAICS_Description_1) and product strings together, no SVD.

Shared TF-IDF vocab size: 28575

| Protocol | Bucket | AUC (textsim_shared, corrected) |
|---|---|---|
| random negatives | singleton | 0.5241 |
| random negatives | overall | 0.5227 |
| degree-matched (seed=1) | overall | 0.5218 |
| degree-matched (seed=1) | singleton | 0.5236 |
| degree-matched (seed=2) | overall | 0.5219 |
| degree-matched (seed=2) | singleton | 0.5238 |
| degree-matched (seed=3) | overall | 0.5220 |
| degree-matched (seed=3) | singleton | 0.5237 |
