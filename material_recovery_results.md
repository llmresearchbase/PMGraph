# Recorded-material-relation recovery pilot (single seed=42)

Negatives during training are PSEUDO-negatives (unrecorded, not confirmed absent). Recall@K measures recovery of DIRECTORY-RECORDED materials only.

Negative sampling (fixed this run): rejection-sampled to exclude a manufacturer's own known positives from training negatives.

| Condition | selected LR | val Recall@10 | group | n_test | Recall@5 | Recall@10 |
|---|---|---|---|---|---|---|
| frequency_baseline | n/a | n/a | all | 1200 | 0.4799 | 0.6766 |
| baseline_no_image | 1.0e-02 | 0.7103 | all | 1200 | 0.5169 | 0.7128 |
| baseline_no_image | 1.0e-02 | 0.7103 | has_image | 605 | 0.5211 | 0.7144 |
| baseline_no_image | 1.0e-02 | 0.7103 | no_image | 595 | 0.5126 | 0.7112 |
| real | 1.0e-02 | 0.7181 | all | 1200 | 0.5116 | 0.7201 |
| real | 1.0e-02 | 0.7181 | has_image | 605 | 0.5161 | 0.7209 |
| real | 1.0e-02 | 0.7181 | no_image | 595 | 0.5070 | 0.7193 |
| zeroed | 1.0e-02 | 0.7101 | all | 1200 | 0.5064 | 0.7167 |
| zeroed | 1.0e-02 | 0.7101 | has_image | 605 | 0.5078 | 0.7169 |
| zeroed | 1.0e-02 | 0.7101 | no_image | 595 | 0.5049 | 0.7165 |
