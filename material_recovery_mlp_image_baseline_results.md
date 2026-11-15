# Material-relation recovery: MLP + image (completes the 2x3 table), 3 seeds

Same concat-style input Design A's GNN uses (64-dim attrs + 1-dim has_image
flag + 32-dim SVD image slice), fed into the no-message-passing FeatureMLP
instead of HeteroSAGE. No new fusion design introduced.

| seed | condition | LR | val R@10 | group | Recall@5 | Recall@10 |
|---|---|---|---|---|---|---|
| 42 | mlp_real_image | 1.0e-02 | 0.7267 | all | 0.5389 | 0.7273 |
| 42 | mlp_real_image | 1.0e-02 | 0.7267 | has_image | 0.5463 | 0.7299 |
| 42 | mlp_real_image | 1.0e-02 | 0.7267 | no_image | 0.5314 | 0.7248 |
| 42 | mlp_zeroed_image | 5.0e-03 | 0.7262 | all | 0.5370 | 0.7300 |
| 42 | mlp_zeroed_image | 5.0e-03 | 0.7262 | has_image | 0.5295 | 0.7268 |
| 42 | mlp_zeroed_image | 5.0e-03 | 0.7262 | no_image | 0.5446 | 0.7333 |
| 43 | mlp_real_image | 5.0e-03 | 0.7320 | all | 0.5496 | 0.7338 |
| 43 | mlp_real_image | 5.0e-03 | 0.7320 | has_image | 0.5509 | 0.7372 |
| 43 | mlp_real_image | 5.0e-03 | 0.7320 | no_image | 0.5483 | 0.7303 |
| 43 | mlp_zeroed_image | 1.0e-02 | 0.7249 | all | 0.5353 | 0.7278 |
| 43 | mlp_zeroed_image | 1.0e-02 | 0.7249 | has_image | 0.5319 | 0.7237 |
| 43 | mlp_zeroed_image | 1.0e-02 | 0.7249 | no_image | 0.5388 | 0.7320 |
| 44 | mlp_real_image | 1.0e-02 | 0.7309 | all | 0.5391 | 0.7349 |
| 44 | mlp_real_image | 1.0e-02 | 0.7309 | has_image | 0.5380 | 0.7346 |
| 44 | mlp_real_image | 1.0e-02 | 0.7309 | no_image | 0.5403 | 0.7352 |
| 44 | mlp_zeroed_image | 5.0e-03 | 0.7240 | all | 0.5269 | 0.7270 |
| 44 | mlp_zeroed_image | 5.0e-03 | 0.7240 | has_image | 0.5241 | 0.7258 |
| 44 | mlp_zeroed_image | 5.0e-03 | 0.7240 | no_image | 0.5298 | 0.7282 |

## Real - zeroed, has_image group, per seed

- seed=42: dR5=+0.0168  dR10=+0.0031
- seed=43: dR5=+0.0191  dR10=+0.0135
- seed=44: dR5=+0.0139  dR10=+0.0088

mean +- sample std (ddof=1): dR5=+0.0166+-0.0026  dR10=+0.0085+-0.0052
