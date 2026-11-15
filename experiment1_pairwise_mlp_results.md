# Task A: pairwise no-graph MLP baseline

Scores each (manufacturer, product) test pair from a concatenation of their own raw 64-dim features ALONE -- no edge_index, no message passing, nothing graph-structural. Same frozen split, same FocalLoss, same early-stopping-on-val-AUC discipline as the SAGE baseline. LR selected from a validation-only grid [0.0001, 0.0005, 0.001, 0.005, 0.01] (selected lr=5.0e-03, val_AUC=0.7029, stop_epoch=310) -- a fresh selection for this model class, not SAGE's reused value.

| Bucket | Protocol | Pairwise MLP (no graph) AUC |
|---|---|---|
| Overall | Random (full test set, n_pos=11259) | 0.6953 |
| Overall | Degree-matched (mean, 3 seeds) | 0.6140 |
| Singleton | Random (full test set) | 0.6887 |
| Singleton | Degree-matched (mean, 3 seeds) | 0.5986 |

Overall PR-AUC (random negatives, full test set): 0.6894

Per-seed degree-matched AUCs -- overall: [0.6134928761911886, 0.6139197937213761, 0.6147137975858796]; singleton: [0.5984958407768031, 0.5965456611120357, 0.6007035881930152]

## Population-matched re-score for Table 2 (same 11214-positive population as the archived SAGE=0.6012 row, seed=1 degree-matching identity exclusion, random negatives)

| Bucket | Protocol | Pairwise MLP | SAGE (archived) | Degree | Shared-space text (corrected method) |
|---|---|---|---|---|---|
| Overall | Random (population-matched) | 0.6952 | 0.6012 (recomputed; archived table value 0.6012) | 0.4154 | 0.5228 |
| Singleton | Random (population-matched) | 0.6887 | 0.7346 (recomputed; archived full-population value 0.7346) | -- | -- |

**Single run (seed=42 only) for the pairwise MLP -- not averaged over multiple training seeds. Report as a single-run result, not as statistically robust across seeds.**
