# Validation of pipeline A

| Script | What it tested |
|---|---|
| `team_on_device.py` (+ `.sbatch`) | the team pipeline in the device setting: PTB-XL at 250 Hz through variants D (pipeline B), TA (no notch), TB (notch), TC (notch + LP40 = pipeline A), thresholds on fold 9, then the real recordings. Writes `config_team_T*_q5.json` |
| `noise_stress.py` (+ `.sbatch`) | NSTDB noise (bw, em, ma, mix at 24 to 0 dB) added to fold 10, cleaned by the team pipeline (0.6 s, 0.3 s median, 0.5 Hz high-pass variants), signal metrics (T/QRS retention, output SNR, R-peak detection) |
| `score_noise.py` (+ `.sbatch`) | scores the noise test outputs with the models (thresholds per pipeline on clean fold 9) |
| `recalibrate.sbatch` | job file for `../../shared/evaluation/recalibrate.py` (thresholds for team-cleaned signals) |
| `pack_cleaned.py`, `team06.sbatch` | ran the teammate's `clean_dataset.py` on PTB-XL and scored fold 10 |
| `dsp_chain.py`, `check_dsp.py`, `batch_clean.py` (+ `.sbatch`) | first batch test of the original repo settings (found the T-wave loss) |

Run from one folder with the shared scripts on the path, as described in the top-level README, e.g.

```
sbatch team_on_device.sbatch        # thresholds + recordings in ~/recordings
```
