# Hit@1 (post-hoc analysis, added after seeing test results -- see caveat)

**Post-hoc disclosure**: this metric was computed after the R@5/@10 test results
were already seen. It supplements, does not replace, those primary results.
Re-executes training with the ALREADY-SELECTED LR per seed (no grid search, no
new hyperparameter choice) because the original runs never saved a raw
per-manufacturer score matrix. R5/R10 are recomputed as a consistency check.

**Caveat**: a miss at rank-1 may reflect the field's own under-recording
(17.4%/24.6% inconsistency rates, Section 4.2), not necessarily a true
capability-prediction error.

| condition | seed | LR (reused) | group | Recall@5 | Recall@10 | Hit@1 |
|---|---|---|---|---|---|---|
| attrs_only | 42 | 1.0e-03 | all | 0.5327 | 0.7304 | 0.7358 |
| attrs_only | 42 | 1.0e-03 | has_image | 0.5309 | 0.7297 | 0.7322 |
| attrs_only | 42 | 1.0e-03 | no_image | 0.5345 | 0.7310 | 0.7395 |
| attrs_only | 43 | 5.0e-03 | all | 0.5352 | 0.7281 | 0.7342 |
| attrs_only | 43 | 5.0e-03 | has_image | 0.5321 | 0.7293 | 0.7355 |
| attrs_only | 43 | 5.0e-03 | no_image | 0.5383 | 0.7269 | 0.7328 |
| attrs_only | 44 | 5.0e-03 | all | 0.5352 | 0.7290 | 0.7300 |
| attrs_only | 44 | 5.0e-03 | has_image | 0.5307 | 0.7261 | 0.7322 |
| attrs_only | 44 | 5.0e-03 | no_image | 0.5398 | 0.7319 | 0.7277 |
| real_image | 42 | 5.0e-03 | all | 0.5436 | 0.7354 | 0.7500 |
| real_image | 42 | 5.0e-03 | has_image | 0.5456 | 0.7380 | 0.7504 |
| real_image | 42 | 5.0e-03 | no_image | 0.5415 | 0.7328 | 0.7496 |
| real_image | 43 | 1.0e-02 | all | 0.5403 | 0.7333 | 0.7433 |
| real_image | 43 | 1.0e-02 | has_image | 0.5399 | 0.7334 | 0.7471 |
| real_image | 43 | 1.0e-02 | no_image | 0.5408 | 0.7333 | 0.7395 |
| real_image | 44 | 5.0e-03 | all | 0.5370 | 0.7314 | 0.7392 |
| real_image | 44 | 5.0e-03 | has_image | 0.5400 | 0.7333 | 0.7405 |
| real_image | 44 | 5.0e-03 | no_image | 0.5338 | 0.7294 | 0.7378 |
| zeroed_image | 42 | 1.0e-02 | all | 0.5330 | 0.7265 | 0.7283 |
| zeroed_image | 42 | 1.0e-02 | has_image | 0.5323 | 0.7273 | 0.7372 |
| zeroed_image | 42 | 1.0e-02 | no_image | 0.5338 | 0.7258 | 0.7193 |
| zeroed_image | 43 | 1.0e-02 | all | 0.5365 | 0.7271 | 0.7342 |
| zeroed_image | 43 | 1.0e-02 | has_image | 0.5335 | 0.7271 | 0.7339 |
| zeroed_image | 43 | 1.0e-02 | no_image | 0.5394 | 0.7270 | 0.7345 |
| zeroed_image | 44 | 1.0e-02 | all | 0.5347 | 0.7282 | 0.7408 |
| zeroed_image | 44 | 1.0e-02 | has_image | 0.5325 | 0.7261 | 0.7388 |
| zeroed_image | 44 | 1.0e-02 | no_image | 0.5370 | 0.7302 | 0.7429 |

## Mean +- sample std (ddof=1), has_image group

- attrs_only has_image recall5: 0.5312 +- 0.0008
- attrs_only has_image recall10: 0.7284 +- 0.0020
- attrs_only has_image hit1: 0.7333 +- 0.0019
- real_image has_image recall5: 0.5418 +- 0.0033
- real_image has_image recall10: 0.7349 +- 0.0027
- real_image has_image hit1: 0.7460 +- 0.0050
- zeroed_image has_image recall5: 0.5328 +- 0.0007
- zeroed_image has_image recall10: 0.7269 +- 0.0006
- zeroed_image has_image hit1: 0.7366 +- 0.0025
