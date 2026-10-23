# Evaluation code

| File | Purpose |
|---|---|
| `score_batch.py` | runs the exported models on stored signals (`score`), labels from the PTB-XL csv files, threshold metrics, paired bootstrap of the AUROC difference; self-check against the Stage 1 results |
| `recalibrate.py` | re-chooses thresholds on fold 9 the same way Stage 1 did (balanced = max(TPR minus FPR), high_sens = first TPR of at least 0.90) |
| `score_batch.sbatch`, `chain_test.sbatch` | job files for the Prajna cluster |
| `chain_test.py` | the chain test (filters with and without the team pipeline, clean, DC offset and noisy) and `score_wm`, the window-mean scorer used by pipeline B and the live demo |

Paths are fixed to `~/data/ptb-xl` (PTB-XL) and `~/export` (models), see the top-level README.
