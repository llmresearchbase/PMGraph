# Material-relation recovery: 3-seed (42/43/44) paired comparison

Same categories/split/fusion/val-LR-selection as the single-seed run; only training seed varies.

## Frequency baseline, per group

| group | n | Recall@5 | Recall@10 |
|---|---|---|---|
| all | 1200 | 0.4799 | 0.6766 |
| has_image | 605 | 0.4822 | 0.6770 |
| no_image | 595 | 0.4776 | 0.6762 |

## Per-seed, per-condition results

| seed | condition | LR | val R@10 | group | Recall@5 | Recall@10 |
|---|---|---|---|---|---|---|
| 42 | baseline_no_image | 1.0e-03 | 0.7197 | all | 0.5392 | 0.7225 |
| 42 | baseline_no_image | 1.0e-03 | 0.7197 | has_image | 0.5324 | 0.7218 |
| 42 | baseline_no_image | 1.0e-03 | 0.7197 | no_image | 0.5462 | 0.7231 |
| 42 | real | 1.0e-02 | 0.7289 | all | 0.5245 | 0.7291 |
| 42 | real | 1.0e-02 | 0.7289 | has_image | 0.5294 | 0.7295 |
| 42 | real | 1.0e-02 | 0.7289 | no_image | 0.5195 | 0.7287 |
| 42 | zeroed | 1.0e-02 | 0.7098 | all | 0.5028 | 0.7066 |
| 42 | zeroed | 1.0e-02 | 0.7098 | has_image | 0.5044 | 0.7108 |
| 42 | zeroed | 1.0e-02 | 0.7098 | no_image | 0.5011 | 0.7024 |
| 43 | baseline_no_image | 1.0e-03 | 0.7253 | all | 0.5314 | 0.7277 |
| 43 | baseline_no_image | 1.0e-03 | 0.7253 | has_image | 0.5334 | 0.7292 |
| 43 | baseline_no_image | 1.0e-03 | 0.7253 | no_image | 0.5294 | 0.7261 |
| 43 | real | 1.0e-02 | 0.7187 | all | 0.5298 | 0.7235 |
| 43 | real | 1.0e-02 | 0.7187 | has_image | 0.5275 | 0.7287 |
| 43 | real | 1.0e-02 | 0.7187 | no_image | 0.5320 | 0.7183 |
| 43 | zeroed | 1.0e-02 | 0.7132 | all | 0.5209 | 0.7208 |
| 43 | zeroed | 1.0e-02 | 0.7132 | has_image | 0.5221 | 0.7230 |
| 43 | zeroed | 1.0e-02 | 0.7132 | no_image | 0.5197 | 0.7185 |
| 44 | baseline_no_image | 5.0e-03 | 0.7244 | all | 0.5263 | 0.7262 |
| 44 | baseline_no_image | 5.0e-03 | 0.7244 | has_image | 0.5306 | 0.7287 |
| 44 | baseline_no_image | 5.0e-03 | 0.7244 | no_image | 0.5219 | 0.7238 |
| 44 | real | 1.0e-03 | 0.7295 | all | 0.5289 | 0.7293 |
| 44 | real | 1.0e-03 | 0.7295 | has_image | 0.5266 | 0.7310 |
| 44 | real | 1.0e-03 | 0.7295 | no_image | 0.5312 | 0.7277 |
| 44 | zeroed | 5.0e-04 | 0.7179 | all | 0.5244 | 0.7211 |
| 44 | zeroed | 5.0e-04 | 0.7179 | has_image | 0.5182 | 0.7201 |
| 44 | zeroed | 5.0e-04 | 0.7179 | no_image | 0.5307 | 0.7221 |

## Paired differences, mean +- std across seeds

- real_minus_baseline [all]: dR5=-0.0046+-0.0090  dR10=+0.0019+-0.0055
- real_minus_baseline [has_image]: dR5=-0.0042+-0.0015  dR10=+0.0031+-0.0042
- real_minus_baseline [no_image]: dR5=-0.0049+-0.0191  dR10=+0.0005+-0.0073
- real_minus_zeroed [all]: dR5=+0.0117+-0.0090  dR10=+0.0112+-0.0102
- real_minus_zeroed [has_image]: dR5=+0.0130+-0.0106  dR10=+0.0117+-0.0065
- real_minus_zeroed [no_image]: dR5=+0.0104+-0.0091  dR10=+0.0106+-0.0140
