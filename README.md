# Single-lead ECG anomaly detection on PTB-XL (Stage 1)

ML workstream of the EE 603 Extraordinary Effort project *Live Prediction of ECG
Anomalies over Edge Computing in Wearables* (Group 3). The goal is a small neural
network that classifies a single-lead ECG from a two-electrode chest device and runs
on the ARM processor of a PYNQ-Z2, after the project's DSP cleaning pipeline.

Stage 1 benchmarks models on PTB-XL to choose the model, task and lead. Stage 2
(on-board latency and memory) is still to come.

## Results so far

All numbers are on the official PTB-XL test fold (fold 10), mean of 3 seeds, using
`resnet1d_wang_s5` (a 1D ResNet with a learned stride-5 input layer).

| Task | Lead | AUROC | Notes |
|---|---|---|---|
| Superclass, 5 classes (benchmark) | 12-lead | 0.931 | matches the published PTB-XL benchmark (about 0.93) |
| Superclass, 5 classes (benchmark) | V2-V1 | 0.819 | V3 0.830, V1 0.818, V2 0.817 |
| Normal vs abnormal | V2-V1 | 0.852 | balanced sens / spec 0.70 / 0.86 |
| Normal vs abnormal | V3 | 0.865 | balanced sens / spec 0.68 / 0.90 |
| AF (AFIB or AFLT) | V2-V1 | 0.964 | balanced sens / spec 0.89 / 0.96 |
| AF (AFIB or AFLT) | V3 | 0.968 | balanced sens / spec 0.90 / 0.96 |

Main findings:

- The strided ResNet beats the plain `resnet1d_wang` on every lead (+0.010 AUROC with
  12 leads, +0.02 to +0.04 with one lead).
- V2-V1 (V2 minus V1) is the bipolar lead between the V1 and V2 electrode sites, the
  closest PTB-XL match to our two-electrode device. It performs as well as V1 or V2.
- AF detection works reliably from one chest lead; normal vs abnormal is moderate.
- First end-to-end test with the DSP pipeline: AF predictions are unchanged by the
  cleaning, but the current DSP attenuates T waves, which turns a normal record into
  "abnormal". The DSP needs to preserve ST/T morphology before the binary model is
  used on cleaned signals.

Detailed tables and figures are in `results/`.

## Repository layout

| Path | Purpose |
|---|---|
| `env_check.py` | checks the conda environment and GPU |
| `ptbxl_data.py` | labels, signal cache, official folds |
| `inspect_dataset.py` | dataset inspection |
| `models.py` | `resnet1d_wang`, `resnet1d_wang_s5` and the model registry |
| `train.py` | trains and tests one configuration |
| `train.sbatch`, `train_seeds.sbatch` | Slurm jobs (one run, or several seeds in one job) |
| `submit_batch1b.sh` | submits the chest-lead batch |
| `collect_results.py` | summarises all runs (mean and spread over seeds) |
| `evaluate.py`, `eval.sbatch` | operating points, bootstrap intervals and figures |
| `export_model.py` | packages trained models for laptop inference |
| `inference/infer.py` | runs the exported models on one ECG file (CPU) |
| `inference/compare.py` | compares a raw record with its DSP-cleaned version |
| `results/` | summary tables and figures |

Data, caches, checkpoints and logs are not tracked (see `.gitignore`).

## Reproducing on the cluster

```bash
conda activate ecg
python env_check.py
# train one configuration
sbatch train.sbatch --task af --leads V2-V1 --model resnet1d_wang_s5 --win_sec 5.0
# three seeds in one job
sbatch train_seeds.sbatch "0 1 2" --task binary --leads V2-V1 --model resnet1d_wang_s5 --win_sec 2.5
# summarise and evaluate
python collect_results.py --task binary
sbatch eval.sbatch --task binary --leads V2-V1 V3
# package models for the laptop
python export_model.py
```

`--leads` accepts `all`, one lead (`V3`), a list (`V1,V2,V3`) or a bipolar lead
written `A-B` (`V2-V1`).

## Inference on a laptop

Copy the folder written by `export_model.py` (weights, `models.py`, `config.json`),
put `inference/infer.py` in it, then:

```bash
pip install -r inference/requirements.txt
python infer.py path/to/record.dat        # WFDB record with its .hea
python infer.py cleaned.csv               # CSV with V1 and V2, or one V2-V1 column
```

The input must be in mV at 500 Hz, about 10 s long.

## Data

PTB-XL v1.0.3 (500 Hz): Wagner et al., *PTB-XL, a large publicly available
electrocardiography dataset*, Scientific Data, 2020. Available from PhysioNet under
CC BY 4.0. Benchmark and base architecture: Strodthoff et al., *Deep learning for ECG
analysis: benchmarks and insights from PTB-XL*, IEEE JBHI, 2021.
