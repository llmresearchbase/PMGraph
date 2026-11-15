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
| 42 | 1.0e-02 | 0.7249 | all | 0.5350 | 0.7314 |
| 42 | 1.0e-02 | 0.7249 | has_image | 0.5369 | 0.7314 |
| 42 | 1.0e-02 | 0.7249 | no_image | 0.5330 | 0.7314 |
| 43 | 1.0e-02 | 0.7239 | all | 0.5326 | 0.7289 |
| 43 | 1.0e-02 | 0.7239 | has_image | 0.5299 | 0.7280 |
| 43 | 1.0e-02 | 0.7239 | no_image | 0.5353 | 0.7298 |
| 44 | 1.0e-02 | 0.7258 | all | 0.5318 | 0.7315 |
| 44 | 1.0e-02 | 0.7258 | has_image | 0.5309 | 0.7291 |
| 44 | 1.0e-02 | 0.7258 | no_image | 0.5328 | 0.7339 |

## Mean +- sample std (ddof=1) across 3 seeds

- feature_mlp_no_graph [all]: Recall@5=0.5331+-0.0016  Recall@10=0.7306+-0.0014
- feature_mlp_no_graph [has_image]: Recall@5=0.5326+-0.0038  Recall@10=0.7295+-0.0017
- feature_mlp_no_graph [no_image]: Recall@5=0.5337+-0.0014  Recall@10=0.7317+-0.0021
