# Results of pipeline A tests

Summary outputs copied from `~/dsp_test/batch` on Prajna, and plots of the real recordings.

| File | From |
|---|---|
| `team_on_device_q5.txt` | `team_on_device.py thresholds`: AUROC, dAUROC and thresholds per variant |
| `still_01_team_q5.png`, `move_01_team_q5.png` (+ `.csv`) | `team_on_device.py recording` on the still and movement recordings |
| `ns_orig_summary.csv`, `ns_reset_summary.csv` | `noise_stress.py` summaries per noise type, SNR and pipeline (T-wave and QRS retention, output SNR, R-peak sensitivity and PPV, locked detector count), with the original detector and with the reset fix |
| `noise_auroc.png` | `score_noise.py`: AUROC against SNR per noise type, PTB-XL fold 10, V2-V1 seed-0 models |
| `noise_operating.png` | `score_noise.py`: sensitivity, specificity and false positives against SNR at thresholds chosen once on clean fold 9 |
| `score_summary.txt`, `recalibration_summary.txt` | team pipeline on clean PTB-XL and recalibrated thresholds |

Recording plots show one team member's ECG (healthy subject, consent given for this use).
