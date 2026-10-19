# Abnormal detection, resnet1d_wang_s5, PTB-XL test fold

Thresholds chosen on the validation fold. high_sens targets 90% sensitivity. Mean ± spread over seeds; [95% bootstrap interval, seed 0].

| Lead | n test (pos) | AUROC | AUPRC | Balanced sens / spec | High-sens sens / spec | High-sens PPV |
|---|---|---|---|---|---|---|
| V2-V1 | 2158 (1246) | 0.852 ± 0.001 [0.837, 0.867] | 0.904 ± 0.001 [0.891, 0.915] | 0.70 / 0.86 | 0.89 / 0.49 [spec 0.46 to 0.53] | 0.71 |
| V3 | 2158 (1246) | 0.865 ± 0.002 [0.852, 0.881] | 0.912 ± 0.001 [0.902, 0.924] | 0.68 / 0.90 | 0.89 / 0.55 [spec 0.51 to 0.57] | 0.73 |
