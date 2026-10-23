# Models and thresholds

Two `resnet1d_wang_s5` models (1D ResNet with a stride-5 first layer, about 536k parameters each), trained on
PTB-XL folds 1 to 8, lead V2 minus V1, 500 Hz, seed 0. Thresholds were chosen on fold 9 (validation).

| File | Contents |
|---|---|
| `models.py` | `build_model(name, n_leads, n_classes)`, the model definitions |
| `config.json` | per task: model name, weights file, sampling rate, window length (binary 2.5 s, AF 5 s), training mean/sd, thresholds for raw 500 Hz input |
| `binary_V2-V1_seed0.pt` | normal vs abnormal (any of MI, STTC, CD, HYP), state dict |
| `af_V2-V1_seed0.pt` | atrial fibrillation or flutter, state dict |
| `config_device_q5.json` | thresholds for pipeline B (binary 0.5750, AF 0.3286) |
| `config_device_q30.json` | thresholds for pipeline B with a Q30 notch (not recommended, see the mains stress test) |
| `config_team_TC_q5.json` | thresholds for pipeline A (binary 0.6857, AF 0.3084) |
| `config_team_TA_q5.json`, `config_team_TB_q5.json` | thresholds of the other team variants tested (no notch; notch only) |
| `config_notch_lp40.json` | thresholds for notch Q30 + LP40 at 500 Hz (filter test) |
| `chain_thresholds.json` | fold-9 thresholds of all chains in the chain test |
| `config_clean06.json` | thresholds for PTB-XL cleaned by the team pipeline at 500 Hz (recalibration) |

Each input chain shifts the model outputs slightly, so always use the threshold file that matches the chain.

Inference: signal in mV at 500 Hz, (x minus mu) / sd, windows with 50 % overlap over 10 s, sigmoid, mean over
windows, compare with `threshold_balanced`. Pipeline B also subtracts each window's own mean first.
