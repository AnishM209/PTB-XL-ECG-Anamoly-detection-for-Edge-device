# ECG Motion-Artefact Removal Pipeline — Project Guide

*DSP project · IIT Bombay · study notes for the presentation*

---

## 1. What the project is about

**Goal:** a wearable ECG device that records a 2-electrode ECG, removes baseline wander, motion artefact
and other noise in real time, and feeds the cleaned signal to a CNN that classifies each 4-second window
as **normal or abnormal**.

```
 electrodes ──► analog front end ──► ADC 500 Hz ──► FPGA: median filter + LPF + SWT
                                                      │  (coefficients written to DDR, ping-pong banks)
                                                      ▼
                                   Processor (Python): R-peak detection → QRS removal → thresholds
                                   → multiresolution thresholding → ISWT → clean ECG → CNN inference
```

| Item | Value |
|---|---|
| Sampling rate | 500 Hz |
| Lead | V2 − V1 (emulates a 2-electrode wearable) |
| Processing window | 2048 samples = 4.096 s (power of 2 for the SWT) |
| Output | Clean ECG window → CNN (normal / abnormal) |
| Training data | PTB-XL (500 Hz records), converted to V2 − V1 and cleaned with the same pipeline |

---

## 2. Reference paper and how we use it

**Berwal et al., "Motion Artifact Removal in Ambulatory ECG Signal for Heart Rate Variability
Analysis", IEEE Sensors Journal, vol. 19, no. 24, 2019** — the "SWT-LT" (stationary wavelet transform
with level thresholding) method.

- **Paper's goal:** clean R-to-R intervals for **heart-rate variability (HRV)**. Evaluated with
  correlation coefficient (avg 0.9337) and NMSE (0.012) on 20 s recordings of 5 healthy subjects and on
  PhysioNet's Motion Artifact Contaminated ECG Database, plus HRV metrics (SDNN, RMSSD, LF/HF).
- **Two phases:** Phase 1 removes low-frequency motion artefact (this is what we implemented).
  Phase 2 (second SWT, zero D1–D5, Hermite approximation of the QRS, QRS scaling) removes
  high-frequency artefact but replaces the real QRS by a fitted model and smooths P/T.
- **Our goal is different:** anomaly *classification* needs the P wave, QRS shape, **ST segment and
  T wave** preserved, not just R-peak timing. This is why some settings were changed (Section 6).

| Aspect | Paper | Our pipeline | Why |
|---|---|---|---|
| Median filter | 0.3 s (chosen from 0.1–0.4 s by correlation) | **0.6 s** | 0.3 s distorts ST/T (Section 6) |
| A7 band | thresholded (outliers only) | thresholded (same) | — |
| QRS removal window | ±50 ms on D1–D6 | ±(50 ms + 2^j samples) per level | ±50 ms left most QRS energy in D5/D6 |
| Thresholds | median ± median-abs-dev of 1 s maxima/minima | same | — |
| SWT | Haar, 7 levels | same | — |
| R-peak detector | Christov on D4 | same + peak refinement | crossing point was 30–40 ms early |
| Phase 2 | yes | **not used** | Hermite QRS model would hide abnormal QRS shapes |

---

## 3. ECG and noise basics (for the presentation)

| Component | Duration / frequency |
|---|---|
| P wave | 80–100 ms, ~0.5–10 Hz |
| QRS complex | 80–120 ms, ~10–40 Hz (most energy) |
| ST segment / T wave | T ≈ 150–250 ms, ~0.5–10 Hz; ST level shifts of 0.05–0.1 mV are clinically significant |
| Diagnostic ECG bandwidth | 0.05–150 Hz |

| Noise (paper Table I) | Frequency | Amplitude |
|---|---|---|
| Baseline wander | 0.15–0.3 Hz | ~15% of peak-to-peak ECG |
| EMG | 5–500 Hz | ~10% |
| Powerline | 50/60 Hz | ~50% |
| **Motion artefact** | **1–10 Hz (in-band!)** | up to 500% |

Motion artefact **overlaps the ECG band**, so ordinary filters cannot remove it without removing ECG.
The SWT approach removes only the *outlier* (unusually large) parts of each frequency band.

---

## 4. The pipeline, block by block

### Block 1 — Median filter (baseline removal) · `median_filter.py`
- `out = x − medfilt(x, 0.6 s)` (301 samples). A median longer than the QRS ignores it and tracks only
  the slow baseline, which is then subtracted.
- **On the FPGA** the median must be *centred*: delay the signal by 150 samples and subtract
  `median(x[n−300 … n])`. A past-only median (zero delay) dropped artefact-removal SNR from 6.1 to 2.1 dB.

### Block 2 — Low-pass filter 150 Hz · `lpf_150hz.py`
- 4th-order Butterworth, zero-phase (`filtfilt`) in Python; causal on the FPGA (~1 sample = 2 ms delay).
- Removes noise above 150 Hz; passband flat to 100 Hz.

### Block 3 — Stationary Wavelet Transform · `swt_haar.py`
- Haar wavelet, 7 levels, **no decimation** (every level has as many samples as the input → shift
  invariant). "À trous": level j uses filters with 2^(j−1) − 1 zeros inserted.
- `A_j[n] = (A_{j−1}[n] + A_{j−1}[n+s]) / √2`, `D_j[n] = (A_{j−1}[n] − A_{j−1}[n+s]) / √2`, s = 2^(j−1).
- Verified identical to PyWavelets, perfect reconstruction (error ~1e−16).

| Band | Frequency at 500 Hz | Contains |
|---|---|---|
| D1 | 125–250 Hz | noise |
| D2 | 62.5–125 Hz | noise, QRS edges |
| D3 | 31–62 Hz | QRS |
| **D4** | **15.6–31 Hz** | **QRS (used for R-peak detection)** |
| D5 | 7.8–15.6 Hz | QRS, P/T edges, motion |
| D6 | 3.9–7.8 Hz | P, T, motion |
| D7 | 1.95–3.9 Hz | T, motion |
| A7 | 0–1.95 Hz | baseline, ST level, slow motion |

### Block 4 — R-peak detection (Christov) · `christov_r_peak_detector.py`
- Runs on |D4|. Adaptive threshold **M + F + R**:
  - **M** (steep slope): 0.6 × average of the last 5 beat maxima; decays 100% → 60% between 200 ms and 1200 ms after a beat.
  - **F** (integrating): tracks high-frequency noise over the last 350 ms.
  - **R** (beat expectation): lowers the threshold when the next beat is due (2/3 RR … RR).
- 200 ms refractory period. Start-up: M = 0.6 × max of the first 5 s.
- Detection is moved from the threshold crossing to the |D4| maximum within 100 ms (true R peak).

### Block 5 — QRS removal · `QRS_removal.py`
- Zeroes D1–D6 around every R peak so the QRS is never treated as artefact (D7 untouched).
- Window half-width = 50 ms + 2^j samples (Haar support grows with level):

| D1 | D2 | D3 | D4 | D5 | D6 |
|---|---|---|---|---|---|
| ±54 ms | ±58 ms | ±66 ms | ±82 ms | ±114 ms | ±178 ms |

### Block 6 — Threshold calculation · `threshold_calculator.py`
- Split each band into 1 s segments; take each segment's max and min.
- `t_max = median(max) + median|median(max) − max_i|`, `t_min = median(min) − median|median(min) − min_i|`.
- Median/MAD instead of mean/std so that artefact segments do not inflate the thresholds.

### Block 7 — Multiresolution thresholding · `multiresolution_thresholding.py`
- Coefficients **inside** [t_min, t_max] = normal ECG → set to 0.
- Coefficients **outside** = outliers = **motion artefact** → kept.
- Applied to D1–D7 and A7.

### Block 8 — Inverse SWT · `iswt_haar.py`
- Reconstructs the **artefact signal** from the kept (outlier) coefficients (average-basis inverse).

### Block 9 — Subtraction · `artefact_subtraction.py`
- `clean = preprocessed − artefact`.

---

## 5. Problems found and fixed during testing

| # | Problem in the first version | Effect | Fix |
|---|---|---|---|
| 1 | Median filter returned `medfilt(x)` (the baseline) | R peaks erased (0.63 → −0.08 mV) | return `x − medfilt(x)` |
| 2 | Thresholding zeroed the outliers and kept normal ECG | ECG subtracted, artefact left in | inverted mask: keep outliers |
| 3 | SWT zero-padding (10000 → 16384) included in thresholds and detector | thresholds too tight, fake beats in padding | use only the real N samples |
| 4 | QRS window ±50 ms on all levels | D5/D6 kept 21× / 9× background QRS energy | level-dependent window |
| 5 | Christov start-up used a running max | false beats in the first second | max of the first 5 s |
| 6 | Christov reported the threshold crossing | R position 30–40 ms early | refine to |D4| maximum |
| 7 | Christov F divisor 150 (meant for 250 Hz) | F twice too large at 500 Hz | scale with fs (300) |
| 8 | SciencePlots style required at import | crash on machines without it | optional |
| 9 | Remove all of A7 + 0.3 s median | T wave 52–76%, ST shifted ≤0.1 mV → normal beats looked abnormal | A7 thresholded, 0.6 s median |
| 10 | PTB-XL records end with ~0.1 s of constant value | false "artefact" at the end | trim flat tail, mirror-pad edges |

---

## 6. Key results

**Signal quality (Phase-1 pipeline):**
- R-peak detection: all beats found with no false detections on the test records (24/24 at 73 bpm,
  11/11 at 65 bpm, 26/26 at **155 bpm**). Median window (0.3 / 0.45 / 0.6 s) does not change detection.
- Synthetic motion-artefact test: input SNR −7.0 dB → output ≈ +6 dB (0.3 s settings).

**Why the settings were changed (morphology vs. artefact removal):**

| Setting | T wave kept | ST level shift | Motion-artefact SNR (synthetic) |
|---|---|---|---|
| 0.3 s median, A7 removed (old) | 52–76% | up to 0.10 mV | 6.2 dB |
| 0.3 s median, A7 thresholded (paper) | 73–83% | up to 0.08 mV | 6.1 dB |
| **0.6 s median, A7 thresholded (now)** | **97–100%** | **≤ 0.01 mV** | 0.5 dB |

- The paper's correlation metric cannot see this: correlation stays 0.94–0.99 for all settings because
  it is dominated by the QRS.
- **Trade-off:** the longer window keeps ST/T but removes less slow motion artefact by itself →
  must be validated on real motion noise (MIT-BIH Noise Stress Test DB, PhysioNet motion-artefact DB,
  or own hardware recordings).
- ML team: ~87% normal/abnormal accuracy after the longer window (to be confirmed with patient-wise
  PTB-XL folds and sensitivity/specificity).

---

## 7. Hardware implementation

### FPGA / processor split with ping-pong DDR banks
- **FPGA:** median filter, LPF and the SWT run **continuously** on the stream; D1–D7 and A7 are written
  to DDR, 2048 samples per bank, alternating bank A / bank B.
- **Processor:** when a bank is full, Python runs blocks 4–9 on it and the CNN infers on the window.
- Because the SWT is continuous, there is **no wrap-around at bank edges** (cutting the signal first
  and running the SWT per window caused errors up to 0.5–1 mV at window boundaries).

### What crosses a bank boundary (`bank_pipeline.py`)
| State kept by the processor | Size |
|---|---|
| Last coefficients of every band (ISWT look-back 128 + not-yet-final 160) | 288 × 8 values |
| R-peak detector thresholds + |D4| history (350 ms) | ~15 values + 176 samples |
| Threshold history: per-second max/min, last 16 s | ≤ 256 values |

- Each cycle outputs 2048 final clean samples **shifted 160 samples (0.32 s) earlier**:
  bank k → samples [2048k − 160, 2048k + 1888).
- Result: within 0.01–0.09 mV of the full-signal pipeline, same SNR, all beats found.
- **Warm-up:** the detector accepts no beats in the first 5 s after power-on → first bank not valid.

### FPGA checklist
- Median: 301-sample window, centred (signal delayed 150 samples).
- SWT coefficients must match `swt_haar.py` exactly: same **time alignment** (D_j[n] uses x[n … n+2^j−1])
  and **scaling** (no normalisation; DC gain √2 per level, ≈11× at A7 → size fixed-point widths).
  Test vectors: `swt_test_input.txt` → `swt_expected_output.txt`.
- Processor must finish a bank within 4.096 s and copy the 288-sample history before the bank is reused.

### Sample-by-sample reference (`streaming_pipeline.py`)
- Proves the minimum buffer sizes (any buffer one sample smaller fails): e.g. median 301, |D4| 176,
  D1–D6 123–154, D7/A7 91 samples; ≈ 8 KB total as float32. Latency 303 samples (0.6 s).

---

## 8. Dataset preparation for the CNN (`clean_dataset.py`)

```
python clean_dataset.py <ptb-xl>/records500 <output>/records500 --workers 4
```
For each record: read V1, V2 → V2 − V1 (mV) → trim flat tail → mirror-pad ~1 s at both ends →
bank pipeline → cut back → write single-lead WFDB record (format 16, 1000 units/mV, 500 Hz,
lead name "V2-V1"). Same folder structure and record names as the input, so `ptbxl_database.csv`
labels still match. Log: `clean_dataset_log.csv`; settings: `clean_dataset_settings.txt`.
Speed ≈ 0.15 s/record (~1 h for 21,800 records on one core).

Files the ML team needs: `clean_dataset.py, pipeline_config.py, bank_pipeline.py,
streaming_pipeline.py, txt_to_dat.py, ecg_io.py, median_filter.py, lpf_150hz.py, swt_haar.py,
iswt_haar.py, multiresolution_thresholding.py, artefact_subtraction.py`.

---

## 9. File map

| File | Purpose |
|---|---|
| `pipeline_config.py` | the two key settings: `MEDIAN_WINDOW_S = 0.6`, `A7_INCLUDED = False` |
| `median_filter.py` … `artefact_subtraction.py` | the 9 blocks; each runnable with its own plot |
| `first_phase_pipeline.py` | whole pipeline on a full recording (`--stages` shows every block) |
| `bank_pipeline.py` | hardware design: continuous SWT + 2048-sample ping-pong banks |
| `streaming_pipeline.py` | sample-by-sample reference with proven minimum buffer sizes |
| `clean_dataset.py` | converts a PTB-XL folder into cleaned V2 − V1 WFDB records |
| `txt_to_dat.py` | txt ↔ WFDB .dat/.hea |
| `ecg_io.py`, `plot_raw_ecg.py` | input loading, shared plot style, raw ECG plots |

---

## 10. Open issues / future work

1. **Validate on real motion noise** (NSTDB / own device) to quantify the artefact-removal loss of the
   0.6 s window; consider adaptive processing (gentle when still, stronger during motion, e.g. accelerometer).
2. **Thresholds are tight** (median + 1×MAD): unusually tall T waves are sometimes trimmed as
   "artefact" (2 of 11 beats on record 00057). Wider thresholds (e.g. 3×MAD) are a possible change.
3. **Christov start-up / amplitude alternation:** missed beats on leads with alternating QRS height
   (lead III of 00057); V2 − V1 was fine.
4. **R-peak position** can be ~26 ms before the R apex on wide QRS (D4 maximum is on the upstroke);
   harmless for cleaning, matters if beats are segmented around R for the CNN.
5. **Phase 2 of the paper** not implemented — only add it if it does not reduce classification accuracy.
6. Fixed-point implementation on FPGA and timing on the processor still to be verified.

---

## 11. Likely questions

- **Why SWT instead of DWT?** No decimation → shift invariant; every level has full time resolution,
  so the artefact can be located and subtracted sample by sample.
- **Why Haar?** Simplest wavelet (2 taps), cheapest on FPGA; the paper found the wavelet choice has
  little effect for SWT.
- **Why remove the QRS before thresholding?** The QRS is the largest feature in D1–D6; without removal
  it would be flagged as an outlier and subtracted.
- **Why median/MAD thresholds?** Robust to artefact segments; mean/std would be inflated by them.
- **Why 0.6 s and not the paper's 0.3 s?** The paper optimised correlation for HRV; correlation is
  QRS-dominated and blind to ST/T changes. For classification, 0.6 s preserves ST/T (T 97–100%)
  while 0.3 s introduced ST depression-like distortion.
- **Why 2048 samples?** Power of 2 for the SWT, ~4 s window for the CNN; continuous SWT on the FPGA
  avoids edge effects.
- **Latency?** ≈ 4.1 s window + 0.32 s finalization delay + 0.3 s median delay.
