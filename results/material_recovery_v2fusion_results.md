# Material recovery: independent-projection image-fusion baseline (raw 768-dim, 3 seeds)

| seed | condition | LR | val R@10 | group | Recall@5 | Recall@10 |
|---|---|---|---|---|---|---|
| 42 | text_only | 1.0e-02 | 0.7184 | all | 0.5289 | 0.7248 |
| 42 | text_only | 1.0e-02 | 0.7184 | has_image | 0.5309 | 0.7246 |
| 42 | text_only | 1.0e-02 | 0.7184 | no_image | 0.5269 | 0.7250 |
| 42 | real_image | 1.0e-02 | 0.5599 | all | 0.3845 | 0.5711 |
| 42 | real_image | 1.0e-02 | 0.5599 | has_image | 0.3865 | 0.5614 |
| 42 | real_image | 1.0e-02 | 0.5599 | no_image | 0.3825 | 0.5810 |
| 42 | zeroed_image | 1.0e-02 | 0.6996 | all | 0.5080 | 0.7001 |
| 42 | zeroed_image | 1.0e-02 | 0.6996 | has_image | 0.5068 | 0.7025 |
| 42 | zeroed_image | 1.0e-02 | 0.6996 | no_image | 0.5093 | 0.6978 |
| 43 | text_only | 5.0e-03 | 0.7297 | all | 0.5293 | 0.7302 |
| 43 | text_only | 5.0e-03 | 0.7297 | has_image | 0.5278 | 0.7331 |
| 43 | text_only | 5.0e-03 | 0.7297 | no_image | 0.5309 | 0.7272 |
| 43 | real_image | 5.0e-03 | 0.6236 | all | 0.4564 | 0.6201 |
| 43 | real_image | 5.0e-03 | 0.6236 | has_image | 0.4839 | 0.6351 |
| 43 | real_image | 5.0e-03 | 0.6236 | no_image | 0.4284 | 0.6048 |
| 43 | zeroed_image | 5.0e-03 | 0.5717 | all | 0.2494 | 0.5628 |
| 43 | zeroed_image | 5.0e-03 | 0.5717 | has_image | 0.2481 | 0.5650 |
| 43 | zeroed_image | 5.0e-03 | 0.5717 | no_image | 0.2508 | 0.5606 |
| 44 | text_only | 1.0e-02 | 0.5754 | all | 0.4439 | 0.5767 |
| 44 | text_only | 1.0e-02 | 0.5754 | has_image | 0.4456 | 0.5815 |
| 44 | text_only | 1.0e-02 | 0.5754 | no_image | 0.4423 | 0.5718 |
| 44 | real_image | 5.0e-03 | 0.4970 | all | 0.3116 | 0.5030 |
| 44 | real_image | 5.0e-03 | 0.4970 | has_image | 0.2805 | 0.4506 |
| 44 | real_image | 5.0e-03 | 0.4970 | no_image | 0.3433 | 0.5563 |
| 44 | zeroed_image | 1.0e-02 | 0.5520 | all | 0.3453 | 0.5594 |
| 44 | zeroed_image | 1.0e-02 | 0.5520 | has_image | 0.3607 | 0.5875 |
| 44 | zeroed_image | 1.0e-02 | 0.5520 | no_image | 0.3297 | 0.5308 |

## Paired differences, mean +- sample std across seeds

- real_minus_text [all]: dR5=-0.1165+-0.0383  dR10=-0.1125+-0.0401
- real_minus_text [has_image]: dR5=-0.1178+-0.0648  dR10=-0.1307+-0.0326
- real_minus_text [no_image]: dR5=-0.1153+-0.0253  dR10=-0.0940+-0.0688
- real_minus_zeroed [all]: dR5=+0.0166+-0.1709  dR10=-0.0427+-0.0939
- real_minus_zeroed [has_image]: dR5=+0.0118+-0.1951  dR10=-0.0693+-0.1207
- real_minus_zeroed [no_image]: dR5=+0.0215+-0.1523  dR10=-0.0157+-0.0880
