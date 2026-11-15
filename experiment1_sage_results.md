# Experiment 1: SAGE on corrected Graph_A, frozen split (seed=42)

Overall test AUC=0.6945  PR-AUC=0.6610  (lr=0.005, hidden=128, dropout=0.3, loss=focal, stop_epoch=21)

| Bucket | n | n_pos | n_neg | AUC | PR-AUC |
|---|---|---|---|---|---|
| 1_singleton | 16041 | 6242 | 9799 | 0.8155 | 0.6625 |
| 2-5 | 3287 | 2123 | 1164 | 0.5668 | 0.7023 |
| 6-19 | 1672 | 1431 | 241 | 0.5384 | 0.8715 |
| high_frequency_20+ | 1518 | 1463 | 55 | 0.5893 | 0.9709 |
