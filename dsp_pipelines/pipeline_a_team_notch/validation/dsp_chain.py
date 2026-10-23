"""
dsp_chain.py: chains the DSP blocks from ~/dsp/python_models (repo pyCoder03/ECG_Anomaly_Detection_DSP,
commit a9cd487) in the order given in the progress report, plus a beat-level amplitude measurement.

Chain (teammate default):
  raw -> median_filter (0.3 s, returns raw minus median baseline) -> lpf_150hz (order 4, filtfilt)  = "pre"
  pre -> SWT_Haar (7 levels) -> Christov on D4 -> QRS masking on D1..D6 -> thresholds on D1..D7, A7
      -> inverted thresholding (A7 kept whole if a7_included) -> ISWT = "artifact"
  cleaned = pre - artifact

ASSUMPTIONS (no driver script in the repo, confirm with the DSP teammate):
  * 7 SWT levels, A7_included=True (the function default).
  * Thresholds computed on the QRS-removed coefficients (as the threshold_calculator docstring says).
  * Subtraction done in the time domain after ISWT. ISWT is linear with perfect reconstruction, so this
    equals subtracting in the coefficient domain and then inverting.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, medfilt

DSP_DIR = Path.home() / "dsp" / "python_models"
sys.path.insert(0, str(DSP_DIR))

from median_filter import median_filter                      # noqa: E402
from lpf_150hz import lpf_150hz                              # noqa: E402
from SWT_python import SWT_Haar, ISWT_Average_Basis_Haar     # noqa: E402
from christov_r_peak_detector import christov_r_peak_detector  # noqa: E402
from QRS_removal import remove_qrs_complex                   # noqa: E402
from threshold_calculator import calculate_thresholds        # noqa: E402
from multiresolution_thresholding import multiresolution_thresholding  # noqa: E402

LEVELS = 7


def _odd(n):
    n = max(1, int(round(n)))
    return n if n % 2 else n + 1


def baseline_two_stage(x, fs, w1=0.2, w2=0.6):
    """Standard two-stage median baseline (200 ms then 600 ms). Returns x minus baseline."""
    b = medfilt(x, _odd(w1 * fs))
    b = medfilt(b, _odd(w2 * fs))
    return x - b


def highpass(x, fs, fc=0.5, order=2):
    """Zero-phase Butterworth high-pass (0.5 Hz = ECG monitoring-grade baseline removal)."""
    b, a = butter(order, fc / (0.5 * fs), btype="highpass")
    return filtfilt(b, a, x)


def clean(x, fs=500, median_mode="single", median_win=0.3, a7_included=True):
    """Run the chain. median_mode: 'single' (teammate code), 'two_stage' (200 + 600 ms) or 'hp05' (0.5 Hz high-pass)."""
    x = np.asarray(x, dtype=float)
    N = len(x)

    if median_mode == "single":
        m = median_filter(x, fs, window_size=median_win)
    elif median_mode == "two_stage":
        m = baseline_two_stage(x, fs)
    elif median_mode == "hp05":
        m = highpass(x, fs)
    else:
        raise ValueError(f"unknown median_mode {median_mode}")

    pre = lpf_150hz(m, fs)

    approx, details = SWT_Haar(pre, LEVELS)
    r_peaks, *_ = christov_r_peak_detector(details, fs=fs, n_samples=N)
    mod = remove_qrs_complex(details, r_peaks, fs=fs)
    thr = calculate_thresholds(mod, approx, fs=fs, n_samples=N)
    art_d, art_a = multiresolution_thresholding(mod, approx, thr, A7_included=a7_included)
    artifact = ISWT_Average_Basis_Haar((art_a, art_d), LEVELS)[:N]

    cleaned = pre - artifact
    return cleaned, {"pre": pre, "artifact": artifact, "r_peaks": np.asarray(r_peaks), "thresholds": thr}


def refine_r(sig, r_peaks, fs, half_ms=40):
    """Move each detected index to the largest absolute deflection within +/- half_ms of it."""
    h = int(half_ms * 1e-3 * fs)
    out = []
    for r in r_peaks:
        a, b = max(0, r - h), min(len(sig), r + h + 1)
        out.append(a + int(np.argmax(np.abs(sig[a:b]))))
    return np.unique(np.asarray(out, dtype=int))


def beat_amplitudes(sig, r_idx, fs):
    """
    Per beat, relative to an isoelectric level taken as the median of [R-100 ms, R-60 ms]:
      qrs = largest absolute deflection in R +/- 50 ms
      t   = signed value of the largest absolute deflection in [R+150 ms, min(R+450 ms, next R-100 ms)]
    Returns median |T|, median |QRS|, median |T|/|QRS|, number of beats used.
    """
    s = lambda ms: int(ms * 1e-3 * fs)  # noqa: E731
    t_list, q_list = [], []
    for k, r in enumerate(r_idx):
        if r - s(100) < 0 or r + s(450) > len(sig):
            continue
        iso = np.median(sig[r - s(100): r - s(60)])
        qrs = np.max(np.abs(sig[r - s(50): r + s(50)] - iso))
        t_end = r + s(450)
        if k + 1 < len(r_idx):
            t_end = min(t_end, r_idx[k + 1] - s(100))
        if t_end - (r + s(150)) < s(50):
            continue
        seg = sig[r + s(150): t_end] - iso
        t_list.append(seg[np.argmax(np.abs(seg))])
        q_list.append(qrs)
    if not t_list:
        return np.nan, np.nan, np.nan, 0
    t_abs = np.abs(np.asarray(t_list))
    q = np.asarray(q_list)
    return float(np.median(t_abs)), float(np.median(q)), float(np.median(t_abs / q)), len(t_list)
