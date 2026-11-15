# Material-relation recovery: no-message-passing MLP baseline (3 seeds)

Same 56-category vocab, manufacturer-level split, negative-sampling-with-
exclusion rule, Recall@5/@10 eval, and val-only LR selection as
material_recovery_pilot.py / multiseed.py. The ONLY difference: the
manufacturer encoder is a plain 2-layer MLP over the 64-dim attribute
feature with NO access to the manufacturer-product graph at all -- no
message passing, no edge_index, nothing. This isolates how much of the
GNN baseline's (`baseline_no_image`) Recall@K comes from the manufacturer's
own attributes alone vs. from graph structure on top of them.

| seed | LR | val R@10 | group | Recall@5 | Recall@10 |
|---|---|---|---|---|---|
| 42 | 1.0e-03 | 0.7207 | all | 0.5327 | 0.7304 |
| 42 | 1.0e-03 | 0.7207 | has_image | 0.5309 | 0.7297 |
| 42 | 1.0e-03 | 0.7207 | no_image | 0.5345 | 0.7310 |
| 43 | 5.0e-03 | 0.7256 | all | 0.5352 | 0.7281 |
| 43 | 5.0e-03 | 0.7256 | has_image | 0.5321 | 0.7293 |
| 43 | 5.0e-03 | 0.7256 | no_image | 0.5383 | 0.7269 |
| 44 | 5.0e-03 | 0.7291 | all | 0.5352 | 0.7290 |
| 44 | 5.0e-03 | 0.7291 | has_image | 0.5307 | 0.7261 |
| 44 | 5.0e-03 | 0.7291 | no_image | 0.5398 | 0.7319 |

## Mean +- sample std (ddof=1) across 3 seeds

- feature_mlp_no_graph [all]: Recall@5=0.5344+-0.0014  Recall@10=0.7292+-0.0011
- feature_mlp_no_graph [has_image]: Recall@5=0.5312+-0.0008  Recall@10=0.7284+-0.0020
- feature_mlp_no_graph [no_image]: Recall@5=0.5375+-0.0027  Recall@10=0.7299+-0.0027
