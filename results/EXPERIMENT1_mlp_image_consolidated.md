# Experiment 1 (consolidated): MLP, attribute-only vs +real vs +zeroed image

Research question: without graph message passing, does website-image content
provide an increment for recorded-material-relation recovery?

## Protocol identity, verified not assumed
`material_recovery_mlp_image_baseline.py` imports `material_recovery_mlp_baseline.py`
as `mb` and calls `mb.select_lr_mlp(feat, device, edges, seed)` directly — the
literal same function object, not a reimplementation. Both scripts therefore
share, by construction: the same `LR_GRID` (5 values), the same early-stopping
`PATIENCE`/`MAX_EPOCHS`, the same `FeatureMLP` architecture, the same
negative-sampling-with-exclusion rule, the same 56-category frozen vocabulary,
the same manufacturer-level split, and the same `recall_at_k` evaluation. The
only thing that differs between the attrs-only run and the +real/+zeroed runs
is the input feature tensor's width (64 -> 97, same concat-style construction
Design A's GNN uses: 64-dim attrs + 1-dim has_image flag + 32-dim SVD image
slice). No material-derived feature, no product image, no graph-derived
embedding anywhere in any condition (verified in the prior round, §5.2 of the
paper).

## Full results, mean +/- sample std (ddof=1) across seeds 42/43/44

| Group | Condition | Recall@5 | Recall@10 |
|---|---|---|---|
| all | attrs-only MLP | 0.5344 +/- 0.0014 | 0.7292 +/- 0.0011 |
| all | MLP + real image | 0.5403 +/- 0.0033 | 0.7334 +/- 0.0020 |
| all | MLP + zeroed image | 0.5347 +/- 0.0018 | 0.7273 +/- 0.0009 |
| has_image | attrs-only MLP | 0.5312 +/- 0.0008 | 0.7284 +/- 0.0020 |
| has_image | MLP + real image | 0.5418 +/- 0.0033 | 0.7349 +/- 0.0027 |
| has_image | MLP + zeroed image | 0.5328 +/- 0.0006 | 0.7268 +/- 0.0006 |
| no_image | attrs-only MLP | 0.5375 +/- 0.0027 | 0.7299 +/- 0.0027 |
| no_image | MLP + real image | 0.5387 +/- 0.0043 | 0.7318 +/- 0.0021 |
| no_image | MLP + zeroed image | 0.5367 +/- 0.0028 | 0.7277 +/- 0.0023 |

## Paired per-seed differences, mean +/- sample std (ddof=1)

| Group | Comparison | dRecall@5 | dRecall@10 |
|---|---|---|---|
| all | REAL - ZEROED | +0.0056 +/- 0.0044 (pos. all 3 seeds) | +0.0061 +/- 0.0029 (pos. all 3 seeds) |
| all | REAL - attrs-only MLP | +0.0059 +/- 0.0046 (pos. all 3 seeds) | +0.0042 +/- 0.0016 (pos. all 3 seeds) |
| has_image | REAL - ZEROED | +0.0091 +/- 0.0037 (pos. all 3 seeds) | +0.0081 +/- 0.0023 (pos. all 3 seeds) |
| has_image | REAL - attrs-only MLP | +0.0106 +/- 0.0036 (pos. all 3 seeds) | +0.0065 +/- 0.0022 (pos. all 3 seeds) |
| no_image | REAL - ZEROED | +0.0020 +/- 0.0055 (sign flips) | +0.0042 +/- 0.0043 (sign flips) |
| no_image | REAL - attrs-only MLP | +0.0012 +/- 0.0066 (sign flips) | +0.0019 +/- 0.0045 (sign flips) |

## Reading the no_image group (a subgroup analysis, not a strict null check)
**Revision note (round 22 mock-review correction)**: this section originally
called the no_image row a "null check" and argued its sign-flipping
differences prove the has_image row's consistency is a real content effect
rather than training noise. That overstates what this row establishes --
corrected below; the absolute numbers and diffs above are unchanged.

For a no-image manufacturer, the image slice of its input feature is already
zero in BOTH the "real" and "zeroed" conditions (TruncatedSVD, unlike PCA,
maps an all-zero input to an all-zero output) -- so its own input is
literally identical across conditions. Any nonzero "REAL-ZEROED" difference
observed for a no_image manufacturer is therefore not caused by that
manufacturer's own features. It does NOT follow that such a difference must
be pure training-realization noise: the has_image manufacturers' inputs
genuinely differ during training, which changes the shared MLP weights the
no_image manufacturers are then scored with -- an indirect
training-composition effect, not noise unrelated to the image manipulation.
We do not claim the no_image row's small, sign-flipping differences prove
the has_image row's consistency is a real content effect rather than noise
-- three seeds cannot establish that contrast on its own -- only that the
no_image pattern is consistent with (not proof of) the has_image effect
being concentrated where the image content actually differs.

## Conclusion
The image signal under the no-graph MLP is real-vs-zeroed positive in all 3
seeds on both metrics, has_image group, with magnitude (+0.91pp R@5,
+0.81pp R@10) at least as large as the GNN's own real-vs-zeroed signal
(+0.81pp, +0.62pp, from the paper's Section 5.2). The REAL vs. attrs-only-MLP
comparison is also positive in all 3 seeds on both metrics (+1.06pp R@5,
+0.65pp R@10) -- a cleaner result than the GNN's own REAL-vs-text-only
comparison, which sign-flipped on Recall@5 (paper Section 5.2). We do not
interpret the MLP/GNN gap itself as a pure structural causal effect (per
instruction) -- multiple explanations remain plausible and were not isolated
(paper Section 5.2). This result is written into the paper's Section 5.2
2x3 table and discussion; this file is the full consolidated backing report.
