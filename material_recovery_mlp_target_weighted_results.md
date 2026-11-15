# Target-relevance-weighted image aggregation vs. flat mean-pooling (no-graph MLP)

Zero-shot Jina-CLIP-v1 image-text cosine-similarity softmax weighting per candidate material, vs. the existing uniform mean-pool. Same attrs/vocab/split/negative-sampling/eval/FeatureMLP/FocalLoss as the flat-pooled real-image MLP baseline -- only the image aggregation changes.

## Phase 1: temperature selection (val Recall@10, seed=42 only)

| temperature | val Recall@10 |
|---|---|
| 1.0 | 0.7289 |
| 0.1 | 0.7281 |
| 0.05 | 0.7292 |

Selected (frozen before test): temperature=0.05
(context) flat-pooled baseline val_Recall@10, seed=42: 0.7297

## Phase 2: frozen config, 3-seed test results

| seed | LR | val R@10 | group | Recall@5 | Recall@10 | train+select wall-clock (s) |
|---|---|---|---|---|---|---|
| 42 | 5.0e-03 | 0.7292 | all | 0.5446 | 0.7355 | 37.5 |
| 42 | 5.0e-03 | 0.7292 | has_image | 0.5478 | 0.7371 | 37.5 |
| 42 | 5.0e-03 | 0.7292 | no_image | 0.5414 | 0.7339 | 37.5 |
| 43 | 5.0e-03 | 0.7305 | all | 0.5505 | 0.7355 | 34.5 |
| 43 | 5.0e-03 | 0.7305 | has_image | 0.5513 | 0.7378 | 34.5 |
| 43 | 5.0e-03 | 0.7305 | no_image | 0.5498 | 0.7332 | 34.5 |
| 44 | 5.0e-03 | 0.7314 | all | 0.5411 | 0.7353 | 26.1 |
| 44 | 5.0e-03 | 0.7314 | has_image | 0.5416 | 0.7373 | 26.1 |
| 44 | 5.0e-03 | 0.7314 | no_image | 0.5406 | 0.7332 | 26.1 |

## Mean +- sample std (ddof=1), has_image group

- target_weighted_image: Recall@5=0.5469+-0.0049  Recall@10=0.7374+-0.0004
- flat-pooled (archived): Recall@5=0.5418  Recall@10=0.7349
- target_weighted MINUS flat-pooled: dR5=+0.0051+-0.0055  dR10=+0.0025+-0.0030

Phase 1 wall-clock: 134.4s. Phase 2 wall-clock (3 seeds incl. LR grid): 98.2s.
