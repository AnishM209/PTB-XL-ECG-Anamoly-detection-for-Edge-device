"""
First Phase of the ECG cleaning pipeline (wires the blocks exactly as in the block diagram):

Raw ECG -> Median filter -> LPF 150 Hz -> SWT (7 levels, Haar)            [swt_haar.py]
                                           |-> Christov R-peak detection on D4 --> QRS peak positions
                                           |-> QRS removal (D1-D6) -> Threshold calculation
                                                                   -> Multiresolution thresholding
                                                                   -> ISWT                     [iswt_haar.py]
clean ECG = preprocessed ECG - ISWT output (low-frequency motion artefact)   [artefact_subtraction.py]

Usage:
    python first_phase_pipeline.py                         # default ECG file, overview plot
    python first_phase_pipeline.py my_ecg.txt --fs 500     # any txt file
    python first_phase_pipeline.py --stages                # also show the plot of every block
    python first_phase_pipeline.py --save                  # save the plots to test_results/
    python first_phase_pipeline.py --no-plot               # only write <file>_clean.txt
"""
import argparse
import os

import numpy as np

from ecg_io import load_ecg_txt, DEFAULT_ECG
from pipeline_config import MEDIAN_WINDOW_S, A7_INCLUDED
from median_filter import median_filter
from lpf_150hz import lpf_150hz
from swt_haar import SWT_Haar                      # [UPDATE] SWT and ISWT now live in separate files
from iswt_haar import ISWT_Average_Basis_Haar
from christov_r_peak_detector import christov_r_peak_detector
from QRS_removal import remove_qrs_complex
from threshold_calculator import calculate_thresholds
from multiresolution_thresholding import multiresolution_thresholding
from artefact_subtraction import subtract_artefact, plot_artefact_subtraction

LEVELS = 7  # D1..D7 + A7 (A7 = 0..~2 Hz at 500 Hz)


def first_phase(raw_ecg, fs=500, A7_included=A7_INCLUDED, median_window=MEDIAN_WINDOW_S):
    raw_ecg = np.asarray(raw_ecg, dtype=float)
    N = len(raw_ecg)  # original length -- the SWT pads to a power of 2, everything is cut back to N

    # Preprocessing
    median_out = median_filter(raw_ecg, fs, median_window)
    pre = lpf_150hz(median_out, fs)

    # SWT
    A7, D = SWT_Haar(pre, LEVELS)

    # QRS peak detection (on D4, padding ignored)
    r_peaks, M_h, F_h, R_h, MFR_h = christov_r_peak_detector(D, fs, n_samples=N)

    # QRS removal -> thresholds -> thresholding (artefact coefficients)
    D_mod = remove_qrs_complex(D, r_peaks, fs)
    thresholds = calculate_thresholds(D_mod, A7, fs, n_samples=N)
    D_art, A_art = multiresolution_thresholding(D_mod, A7, thresholds, A7_included=A7_included)

    # ISWT -> low-frequency motion artefact, then subtract
    artefact = ISWT_Average_Basis_Haar((A_art, D_art), LEVELS)[:N]
    clean = subtract_artefact(pre, artefact)          # clean = preprocessed - artefact

    return {
        # final outputs
        "preprocessed": pre, "artefact": artefact, "clean": clean, "r_peaks": r_peaks, "thresholds": thresholds,
        # every intermediate stage (used by the plots)
        "raw": raw_ecg, "median_out": median_out, "A7": A7, "D": D, "D_mod": D_mod, "D_art": D_art, "A_art": A_art,
        "christov": (M_h, F_h, R_h, MFR_h), "fs": fs, "N": N, "A7_included": A7_included, "median_window": median_window,
    }


# ============================================================================ plotting
def plot_overview(out, start=0.0, dur=None, save=False, show=True):
    """Raw -> preprocessed -> estimated artefact -> clean ECG, with the detected R peaks."""
    from ecg_io import setup_style, plot_window, finish, C_BLUE, C_ORANGE, C_AQUA, C_GREY, C_INK
    plt = setup_style()
    fs, N = out["fs"], out["N"]
    sl, t = plot_window(N, fs, start, dur if dur is not None else N / fs)
    pk = out["r_peaks"]; pk = pk[(pk >= sl.start) & (pk < sl.stop)]
    rows = [("raw ECG", out["raw"], C_GREY),
            ("preprocessed (median + LPF)", out["preprocessed"], C_BLUE),
            ("estimated motion artefact (ISWT of thresholded coefficients)", out["artefact"], C_ORANGE),
            ("clean ECG = preprocessed - artefact", out["clean"], C_AQUA)]
    fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for a, (title, sig, col) in zip(ax, rows):
        a.plot(t, sig[sl], color=col)
        a.set_title(title)
    ax[3].plot(pk / fs, out["clean"][pk], "v", color=C_INK, ms=6, label=f"R peaks ({len(out['r_peaks'])} total)")
    ax[3].set_ylim(top=1.3 * out["clean"][sl].max()); ax[3].legend(loc="upper right"); ax[3].set_xlabel("time (s)")
    return finish(fig, "0_pipeline_overview", save, show)


def plot_all_stages(out, start=0.0, dur=6.0, save=False):
    """Opens the plot of every block, in pipeline order."""
    from median_filter import plot_median_filter
    from lpf_150hz import plot_lpf
    from swt_haar import plot_swt
    from christov_r_peak_detector import plot_christov
    from QRS_removal import plot_qrs_removal
    from threshold_calculator import plot_thresholds
    from multiresolution_thresholding import plot_thresholding
    fs, N = out["fs"], out["N"]
    plot_median_filter(out["raw"], out["median_out"], fs, start, dur, save=save, window_size=out["median_window"])
    plot_lpf(out["median_out"], out["preprocessed"], fs, start, dur, save=save)
    plot_swt(out["A7"], out["D"], fs, n_samples=N, start=start, dur=dur, input_sig=out["preprocessed"], save=save)
    plot_christov(out["preprocessed"], out["D"][3], out["r_peaks"], *out["christov"], fs, start, dur, save=save)
    plot_qrs_removal(out["D"], out["D_mod"], out["r_peaks"], fs, n_samples=N, start=start, dur=dur, save=save)
    plot_thresholds(out["D_mod"], out["A7"], out["thresholds"], fs, n_samples=N, save=save)
    plot_thresholding(out["D_mod"], out["A7"], out["thresholds"], out["D_art"], out["A_art"], fs, n_samples=N,
                      artefact=out["artefact"], A7_included=out["A7_included"], save=save)
    plot_artefact_subtraction(out["preprocessed"], out["artefact"], out["clean"], fs, out["r_peaks"], start, dur,
                              save=save)
    plot_overview(out, save=save)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="First Phase ECG cleaning pipeline")
    ap.add_argument("file", nargs="?", default=DEFAULT_ECG, help="ECG txt file (default: %(default)s)")
    ap.add_argument("--fs", type=int, default=500)
    ap.add_argument("--remove-A7", action="store_true", help="remove all of A7 (old behaviour) instead of thresholding it")
    ap.add_argument("--median-window", type=float, default=MEDIAN_WINDOW_S, help="median filter window in s (default %(default)s)")
    ap.add_argument("--stages", action="store_true", help="show the plot of every block")
    ap.add_argument("--start", type=float, default=0.0, help="zoom window start (s) for the stage plots")
    ap.add_argument("--dur", type=float, default=10.0, help="zoom window length (s) for the stage plots")
    ap.add_argument("--save", action="store_true", help="save the plots to test_results/")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    ecg = load_ecg_txt(args.file)
    out = first_phase(ecg, fs=args.fs, A7_included=args.remove_A7 or A7_INCLUDED, median_window=args.median_window)
    rr = np.diff(out["r_peaks"]) / args.fs
    print(f"samples: {len(ecg)}, R peaks: {len(out['r_peaks'])}, mean HR: {60 / rr.mean():.1f} bpm")
    out_path = os.path.splitext(args.file)[0] + "_clean.txt"
    np.savetxt(out_path, out["clean"], fmt="%.6f")
    print("clean ECG written to", out_path)

    if not args.no_plot:
        if args.stages:
            plot_all_stages(out, args.start, args.dur, save=args.save)
        else:
            plot_overview(out, save=args.save)
