# DSP pipelines for single-lead ECG anomaly detection on the PYNQ-Z2

This folder holds the two signal-processing pipelines we built and compared for the EE 603 Extraordinary
Effort project (Group 3, "Live Prediction of ECG Anomalies over Edge Computing in Wearables"), together with
the code that validates them, the live demo and the acquisition script for the board.

Both pipelines sit between the ADS1299 analogue front end and two 1D ResNet models (normal vs abnormal, and
atrial fibrillation), which were trained on the PTB-XL lead V2 minus V1.

| Pipeline | Signal chain | Status |
|---|---|---|
| [A: team pipeline with notch](pipeline_a_team_notch/) | notch 50 Hz (Q5) + low-pass 40 Hz, then the team's First Phase cleaning (median, LPF 150 Hz, SWT, Christov, thresholding, ISWT) | works on still recordings; no accuracy gain over B |
| [B: notch + low-pass (chain D3)](pipeline_b_notch_lp40/) | notch 50 Hz (Q5) + low-pass 40 Hz, upsample to 500 Hz, per-window mean removal | validated default of the live demo |

The full explanation of both pipelines, every test and the conclusions is in
[docs/ecg_dsp_pipelines.html](docs/) (open it in a browser).

## Folder map

| Folder | Contents |
|---|---|
| `pipeline_a_team_notch/` | Pipeline A: the team's DSP code, its device-setting front end, validation scripts, results |
| `pipeline_b_notch_lp40/` | Pipeline B: validation scripts (device chain, mains stress test, filter tests), results |
| `shared/models/` | Exported models and the threshold files for each pipeline |
| `shared/evaluation/` | Scoring and threshold code used by the validation scripts of both pipelines |
| `live_demo/` | Real-time screen: raw and clean signal, verdicts, heart rate, signal quality |
| `acquisition/` | PYNQ notebook that reads the ADS1299 and streams over UART, and a simple live plotter |
| `docs/` | The written summary of the pipelines and results |

## Quick start (live demo on a laptop)

```
pip install -r requirements.txt
cd live_demo
python live_demo.py --models ../shared/models --replay <recording.npz>          # pipeline B, replay
python live_demo.py COM37 --scale 1000 --models ../shared/models                # pipeline B, live
python live_demo.py COM37 --scale 1000 --models ../shared/models \
       --pipeline team --team-dir ../pipeline_a_team_notch/team_dsp             # pipeline A, live
```

The board must run `acquisition/ads1299_uart_record.ipynb` with `STREAM_RAW = True`.

## Running the validation scripts

The validation scripts were run on the IITB Prajna cluster from one flat folder (`~/dsp_test`). They import
each other by module name and expect this layout in the home directory:

| Path | What |
|---|---|
| `~/data/ptb-xl` | PTB-XL v1.0.3 (`records500/`, `ptbxl_database.csv`, `scp_statements.csv`) |
| `~/data/nstdb` | MIT-BIH Noise Stress Test Database (`bw`, `em`, `ma`) |
| `~/export` | the contents of `shared/models/` (models, weights, threshold files) |
| `~/test_cleaning_reset` | the contents of `pipeline_a_team_notch/team_dsp/` |
| `~/dsp_test/batch` | intermediate signal files (`raw.npz`, `ns_*.npz`), created by the scripts |

To run them from this repository instead of a flat folder, put the needed folders on the Python path, e.g.

```
export PYTHONPATH=$PWD/shared/evaluation:$PWD/pipeline_b_notch_lp40/validation:$PWD/pipeline_a_team_notch/validation
ln -s $PWD/shared/models ~/export
ln -s $PWD/pipeline_a_team_notch/team_dsp ~/test_cleaning_reset
```

Data (PTB-XL, NSTDB, recordings, intermediate `.npz` files) is not part of the repository.

## Credits

- Team DSP pipeline (First Phase, after Berwal et al., IEEE Sensors Journal 2019): Shriyans Chavan and Balaji Srinivasan.
- Acquisition and hardware: Ayush Gupta, Md Aariz Mozammil, Ankit Raj, Husain Motiwala.
- Models, device-chain design, validation and live demo: Anish Mayanache, with Karan Patel on model development.
