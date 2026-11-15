# Task B: test positives grouped by whether the target material is text-visible

Per-POSITIVE-LABEL grouping (not per-manufacturer): each test (manufacturer, material) positive pair is tagged by whether that specific material's name is word-boundary-mentioned in the manufacturer's own visible text (product strings, business description, NAICS description) -- never the masked Material_Capability field itself. Metric is per-label hit rate (is this one material within the model's top-K for that manufacturer), not the usual per-manufacturer Recall@K.

Total test positives: 7830. Mentioned in visible text: 1096 (14.0%). Not mentioned: 6734 (86.0%).

Attributes-only MLP: single seed=42, lr=1.0e-03 (archived-selected config, not a fresh grid search), val_Recall@10=0.7207, stop_epoch=212.

| Group | n | Frequency hit@5 | Frequency hit@10 | MLP hit@5 | MLP hit@10 |
|---|---|---|---|---|---|
| mentioned_in_text | 1096 | 0.5328 | 0.7591 | 0.6505 | 0.8066 |
| not_mentioned_in_text | 6734 | 0.3463 | 0.5609 | 0.3861 | 0.6041 |
