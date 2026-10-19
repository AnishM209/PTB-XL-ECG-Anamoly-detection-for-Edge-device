# AF detection, resnet1d_wang_s5, PTB-XL test fold

Thresholds chosen on the validation fold. high_sens targets 90% sensitivity. Mean ± spread over seeds; [95% bootstrap interval, seed 0].

| Lead | n test (pos) | AUROC | AUPRC | Balanced sens / spec | High-sens sens / spec | High-sens PPV |
|---|---|---|---|---|---|---|
| V2-V1 | 2098 (157) | 0.964 ± 0.002 [0.944, 0.982] | 0.839 ± 0.010 [0.757, 0.901] | 0.89 / 0.96 | 0.84 / 0.98 [spec 0.98 to 0.99] | 0.80 |
| V3 | 2098 (157) | 0.968 ± 0.001 [0.948, 0.984] | 0.852 ± 0.011 [0.791, 0.903] | 0.90 / 0.96 | 0.86 / 0.97 [spec 0.97 to 0.98] | 0.73 |
