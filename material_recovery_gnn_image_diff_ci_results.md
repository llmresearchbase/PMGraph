# GNN (Design A): manufacturer-level bootstrap CI for REAL-ZEROED image difference

Reused already-selected per-seed LRs (no grid search). GNN training is NOT
bit-reproducible across runs (GPU/cuDNN, already established in Section 5.3) --
point estimates here are expected to differ somewhat from the archived table;
this is disclosed, not treated as an error.

Bootstrap: manufacturer-level resampling with replacement, n=2000, seed=777.

| seed | metric | point diff (real-zeroed) | bootstrap mean | 95% CI low | 95% CI high | excludes 0? |
|---|---|---|---|---|---|---|
| 42 | Recall@5 | +0.0118 | +0.0117 | +0.0050 | +0.0193 | yes |
| 42 | Recall@10 | +0.0090 | +0.0089 | +0.0032 | +0.0155 | yes |
| 43 | Recall@5 | +0.0076 | +0.0076 | +0.0008 | +0.0144 | yes |
| 43 | Recall@10 | +0.0053 | +0.0053 | +0.0004 | +0.0104 | yes |
| 44 | Recall@5 | +0.0014 | +0.0014 | -0.0059 | +0.0083 | no |
| 44 | Recall@10 | +0.0132 | +0.0131 | +0.0059 | +0.0205 | yes |

## Cross-seed mean +- sample std (ddof=1) -- this rerun's own value, for comparison against the archived table

- Recall@5 real-zeroed, has_image: +0.0069 +- 0.0052 (this rerun; archived table: see material_recovery_multiseed_results.md)
- Recall@10 real-zeroed, has_image: +0.0092 +- 0.0040 (this rerun; archived table: see material_recovery_multiseed_results.md)
