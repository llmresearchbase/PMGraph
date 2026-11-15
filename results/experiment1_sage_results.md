# Experiment 1: SAGE on corrected Graph_A, frozen split (seed=42)

Overall test AUC=0.5998  PR-AUC=0.5805  (lr=0.005, hidden=128, dropout=0.3, loss=focal, stop_epoch=21)

| Bucket | n | n_pos | n_neg | AUC | PR-AUC |
|---|---|---|---|---|---|
| 1_singleton | 16041 | 6242 | 9799 | 0.7346 | 0.5660 |
| 2-5 | 3287 | 2123 | 1164 | 0.5624 | 0.7042 |
| 6-19 | 1672 | 1431 | 241 | 0.5219 | 0.8550 |
| high_frequency_20+ | 1518 | 1463 | 55 | 0.5022 | 0.9621 |
