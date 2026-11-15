# Manufacturer-level bootstrap CI for REAL-ZEROED image difference (MLP, has_image)

Test-sample uncertainty (this file) is a DIFFERENT dimension from the already-
reported cross-seed mean+-std (training-seed randomness) -- reported side by side,
not combined into one number.

Bootstrap: manufacturer-level resampling with replacement, n=2000, seed=777.

| seed | metric | point diff (real-zeroed) | bootstrap mean | 95% CI low | 95% CI high | excludes 0? |
|---|---|---|---|---|---|---|
| 42 | Recall@5 | +0.0133 | +0.0133 | +0.0046 | +0.0225 | yes |
| 42 | Recall@10 | +0.0107 | +0.0106 | +0.0023 | +0.0193 | yes |
| 43 | Recall@5 | +0.0064 | +0.0063 | -0.0008 | +0.0137 | no |
| 43 | Recall@10 | +0.0062 | +0.0063 | -0.0003 | +0.0126 | no |
| 44 | Recall@5 | +0.0076 | +0.0077 | -0.0002 | +0.0158 | no |
| 44 | Recall@10 | +0.0072 | +0.0073 | -0.0010 | +0.0156 | no |

## Cross-seed mean +- sample std (ddof=1) -- training-randomness dimension, already reported in the paper

- Recall@5 real-zeroed, has_image: +0.0091 +- 0.0037 (this script's recomputation)
- Recall@10 real-zeroed, has_image: +0.0080 +- 0.0023 (this script's recomputation)
