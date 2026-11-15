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
| 42 | baseline_no_image | 1.0e-03 | 0.7198 | all | 0.5384 | 0.7224 |
| 42 | baseline_no_image | 1.0e-03 | 0.7198 | has_image | 0.5321 | 0.7208 |
| 42 | baseline_no_image | 1.0e-03 | 0.7198 | no_image | 0.5448 | 0.7240 |
| 42 | real | 1.0e-02 | 0.7298 | all | 0.5252 | 0.7291 |
| 42 | real | 1.0e-02 | 0.7298 | has_image | 0.5289 | 0.7302 |
| 42 | real | 1.0e-02 | 0.7298 | no_image | 0.5213 | 0.7280 |
| 42 | zeroed | 1.0e-02 | 0.7224 | all | 0.5156 | 0.7220 |
| 42 | zeroed | 1.0e-02 | 0.7224 | has_image | 0.5150 | 0.7215 |
| 42 | zeroed | 1.0e-02 | 0.7224 | no_image | 0.5161 | 0.7225 |
| 43 | baseline_no_image | 5.0e-03 | 0.7168 | all | 0.5257 | 0.7200 |
| 43 | baseline_no_image | 5.0e-03 | 0.7168 | has_image | 0.5248 | 0.7219 |
| 43 | baseline_no_image | 5.0e-03 | 0.7168 | no_image | 0.5267 | 0.7181 |
| 43 | real | 1.0e-02 | 0.7191 | all | 0.5298 | 0.7238 |
| 43 | real | 1.0e-02 | 0.7191 | has_image | 0.5274 | 0.7263 |
| 43 | real | 1.0e-02 | 0.7191 | no_image | 0.5323 | 0.7211 |
| 43 | zeroed | 1.0e-02 | 0.7133 | all | 0.5204 | 0.7187 |
| 43 | zeroed | 1.0e-02 | 0.7133 | has_image | 0.5220 | 0.7206 |
| 43 | zeroed | 1.0e-02 | 0.7133 | no_image | 0.5187 | 0.7169 |
| 44 | baseline_no_image | 1.0e-03 | 0.7018 | all | 0.5222 | 0.7000 |
| 44 | baseline_no_image | 1.0e-03 | 0.7018 | has_image | 0.5206 | 0.7016 |
| 44 | baseline_no_image | 1.0e-03 | 0.7018 | no_image | 0.5238 | 0.6984 |
| 44 | real | 1.0e-03 | 0.7304 | all | 0.5286 | 0.7309 |
| 44 | real | 1.0e-03 | 0.7304 | has_image | 0.5261 | 0.7324 |
| 44 | real | 1.0e-03 | 0.7304 | no_image | 0.5310 | 0.7293 |
| 44 | zeroed | 5.0e-03 | 0.7224 | all | 0.5278 | 0.7271 |
| 44 | zeroed | 5.0e-03 | 0.7224 | has_image | 0.5210 | 0.7281 |
| 44 | zeroed | 5.0e-03 | 0.7224 | no_image | 0.5347 | 0.7261 |

## Paired differences, mean +- std across seeds -- REVISED, AUTHORITATIVE VERSION

**Revision note:** the original version of this section (below the original script's own
`np.array(...).std()` call) used population std (ddof=0). This revision uses SAMPLE std
(ddof=1, ddof=n-1), the conventional choice for "mean +- std across a small number of seeds".
Means are unchanged (std convention doesn't affect the mean); only the +- figures differ from
the first pass. An earlier chat message also mislabeled three real_minus_baseline[has_image]
per-seed values (+0.0094/+0.0044/+0.0308) as belonging to real_minus_zeroed -- that was a
reporting error in that message, not in this file or the underlying script; the explicit
per-seed table below removes any ambiguity about which number belongs to which comparison.

### Explicit per-seed breakdown (unambiguous: condition values + both diffs, side by side)

**group = has_image**

| seed | baseline R5 | real R5 | zeroed R5 | R-baseline (R5) | R-zeroed (R5) | baseline R10 | real R10 | zeroed R10 | R-baseline (R10) | R-zeroed (R10) |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.5321 | 0.5289 | 0.5150 | -0.0032 | +0.0139 | 0.7208 | 0.7302 | 0.7215 | +0.0094 | +0.0087 |
| 43 | 0.5248 | 0.5274 | 0.5220 | +0.0026 | +0.0054 | 0.7219 | 0.7263 | 0.7206 | +0.0044 | +0.0057 |
| 44 | 0.5206 | 0.5261 | 0.5210 | +0.0055 | +0.0051 | 0.7016 | 0.7324 | 0.7281 | +0.0308 | +0.0043 |

### Mean +- SAMPLE std (ddof=1), all groups

- real_minus_baseline [all]: dR5=-0.0009+-0.0107  dR10=+0.0138+-0.0149
- real_minus_baseline [has_image]: dR5=+0.0016+-0.0044  dR10=+0.0149+-0.0140
- real_minus_baseline [no_image]: dR5=-0.0036+-0.0173  dR10=+0.0126+-0.0158
- real_minus_zeroed [all]: dR5=+0.0066+-0.0050  dR10=+0.0053+-0.0017
- **real_minus_zeroed [has_image]: dR5=+0.0081+-0.0050  dR10=+0.0062+-0.0022**
- real_minus_zeroed [no_image]: dR5=+0.0050+-0.0087  dR10=+0.0043+-0.0012

### Paper-usable phrasing (agreed)

> 在材料关系恢复任务中，真实图片相对同结构置零对照，在有图企业上的 Recall@5／@10 平均提高 0.81／0.62 个百分点，三个训练 seed 的差值均为正。相对纯文本 baseline，Recall@5 基本持平，Recall@10 的提升波动较大（受 seed44 基线训练偏弱影响），尚不足以支持稳定优越性的结论。

This is a **bounded, reproducible multimodal usage example** for the resource -- not escalated
into "images are proven effective" as a core contribution.
