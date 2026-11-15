# Material-relation recovery: MLP + image (completes the 2x3 table), 3 seeds

Same concat-style input Design A's GNN uses (64-dim attrs + 1-dim has_image
flag + 32-dim SVD image slice), fed into the no-message-passing FeatureMLP
instead of HeteroSAGE. No new fusion design introduced.

| seed | condition | LR | val R@10 | group | Recall@5 | Recall@10 |
|---|---|---|---|---|---|---|
| 42 | mlp_real_image | 5.0e-03 | 0.7297 | all | 0.5436 | 0.7354 |
| 42 | mlp_real_image | 5.0e-03 | 0.7297 | has_image | 0.5456 | 0.7380 |
| 42 | mlp_real_image | 5.0e-03 | 0.7297 | no_image | 0.5415 | 0.7328 |
| 42 | mlp_zeroed_image | 1.0e-02 | 0.7255 | all | 0.5330 | 0.7265 |
| 42 | mlp_zeroed_image | 1.0e-02 | 0.7255 | has_image | 0.5323 | 0.7273 |
| 42 | mlp_zeroed_image | 1.0e-02 | 0.7255 | no_image | 0.5338 | 0.7258 |
| 43 | mlp_real_image | 1.0e-02 | 0.7295 | all | 0.5403 | 0.7333 |
| 43 | mlp_real_image | 1.0e-02 | 0.7295 | has_image | 0.5399 | 0.7334 |
| 43 | mlp_real_image | 1.0e-02 | 0.7295 | no_image | 0.5408 | 0.7333 |
| 43 | mlp_zeroed_image | 1.0e-02 | 0.7263 | all | 0.5365 | 0.7271 |
| 43 | mlp_zeroed_image | 1.0e-02 | 0.7263 | has_image | 0.5335 | 0.7271 |
| 43 | mlp_zeroed_image | 1.0e-02 | 0.7263 | no_image | 0.5394 | 0.7270 |
| 44 | mlp_real_image | 5.0e-03 | 0.7274 | all | 0.5370 | 0.7314 |
| 44 | mlp_real_image | 5.0e-03 | 0.7274 | has_image | 0.5400 | 0.7333 |
| 44 | mlp_real_image | 5.0e-03 | 0.7274 | no_image | 0.5338 | 0.7294 |
| 44 | mlp_zeroed_image | 1.0e-02 | 0.7224 | all | 0.5347 | 0.7282 |
| 44 | mlp_zeroed_image | 1.0e-02 | 0.7224 | has_image | 0.5325 | 0.7261 |
| 44 | mlp_zeroed_image | 1.0e-02 | 0.7224 | no_image | 0.5370 | 0.7302 |

## Real - zeroed, has_image group, per seed

- seed=42: dR5=+0.0133  dR10=+0.0107
- seed=43: dR5=+0.0064  dR10=+0.0062
- seed=44: dR5=+0.0076  dR10=+0.0072

mean +- sample std (ddof=1): dR5=+0.0091+-0.0037  dR10=+0.0080+-0.0023
