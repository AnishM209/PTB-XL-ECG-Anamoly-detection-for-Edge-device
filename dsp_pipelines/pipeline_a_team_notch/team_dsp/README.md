# Team DSP pipeline (First Phase)

Code by Shriyans Chavan and Balaji Srinivasan, following Berwal et al., "Motion Artifact Removal in Ambulatory
ECG Signal for Heart Rate Variability Analysis", IEEE Sensors Journal 19(24), 2019. This copy is the version
used in all our tests (`~/test_cleaning_reset` on Prajna): the teammate's revised code with

- `MEDIAN_WINDOW_S = 0.6` and `A7_INCLUDED = False` in `pipeline_config.py` (the original 0.3 s median and
  full A7 removal cut about 45 % of the T wave),
- the R-peak detector reset fix (`../patches/detector_reset.patch`, marked `[FIX]` in `streaming_pipeline.py`),
- a Python 3.11 f-string fix in `streaming_pipeline.py`.

| File | Block |
|---|---|
| `median_filter.py` | baseline removal: x minus running median (0.6 s) |
| `lpf_150hz.py` | 150 Hz low-pass |
| `swt_haar.py`, `iswt_haar.py` | stationary wavelet transform and inverse, Haar, 7 levels |
| `christov_r_peak_detector.py` | R-peak detection on D4 |
| `QRS_removal.py` | zeroes the coefficients around each beat in D1 to D6 |
| `threshold_calculator.py`, `multiresolution_thresholding.py` | per-band limits and inverted thresholding |
| `artefact_subtraction.py` | cleaned = preprocessed minus artefact |
| `first_phase_pipeline.py` | full-signal driver |
| `bank_pipeline.py` | 2048-sample ping-pong bank version (planned FPGA/processor split); `bank_first_phase` is what pipeline A calls |
| `streaming_pipeline.py` | sample-by-sample reference implementation |
| `clean_dataset.py` | cleans a whole PTB-XL folder into single-lead V2-V1 records |
| `pipeline_config.py` | shared settings |
| `ecg_io.py` | loading ECG text files and shared plotting helpers |
| `plot_raw_ecg.py` | plots a raw ECG file |
| `txt_to_dat.py` | writes single-lead WFDB records (used by `clean_dataset.py`) |

The pipeline itself does not remove mains: pipeline A puts a notch in front of it.

