"""
Final block of the First Phase -- the summing junction in the block diagram:

        clean ECG = preprocessed ECG (median + LPF output)  -  ISWT output (low-frequency motion artefact)

Run on its own to get the cleaned ECG: it runs every block before it, plots preprocessed / artefact /
clean, and writes the cleaned signal to <input>_clean.txt (one value per line, same units as the input).

    python artefact_subtraction.py                      # default ECG file (set in ecg_io.py)
    python artefact_subtraction.py my_ecg.txt --save    # other file, also save the PNG to test_results/
    python artefact_subtraction.py --start 2 --dur 4    # zoom 2..6 s
"""
import os

import numpy as np


def subtract_artefact(preprocessed_ecg, artefact):
    """clean = preprocessed - artefact (the ISWT output is cut to the ECG length first,
    because the SWT pads the signal to a power of 2)."""
    pre = np.asarray(preprocessed_ecg, dtype=float)
    art = np.asarray(artefact, dtype=float)[:len(pre)]
    return pre - art


# ============================================================================ plotting
def plot_artefact_subtraction(preprocessed_ecg, artefact, clean_ecg, fs, r_peaks=None, start=0.0, dur=10.0,
                              save=False, show=True):
    from ecg_io import setup_style, plot_window, finish, C_BLUE, C_ORANGE, C_AQUA, C_GREY, C_INK
    plt = setup_style()
    n = len(clean_ecg)
    sl, t = plot_window(n, fs, start, dur)
    art = np.asarray(artefact)[:n]
    fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1.3]})
    ax[0].plot(t, preprocessed_ecg[sl], color=C_BLUE); ax[0].set_title("Preprocessed ECG (median filter + LPF)")
    ax[1].plot(t, art[sl], color=C_ORANGE); ax[1].set_title("Estimated motion artefact (ISWT of thresholded coefficients)")
    ax[2].plot(t, clean_ecg[sl], color=C_AQUA); ax[2].set_title("Clean ECG = preprocessed - artefact")
    if r_peaks is not None:
        pk = np.asarray(r_peaks, dtype=int); pk = pk[(pk >= sl.start) & (pk < sl.stop)]
        ax[2].plot(pk / fs, clean_ecg[pk], "v", color=C_INK, ms=6, label="R peaks"); ax[2].legend(loc="upper right")
    ax[3].plot(t, preprocessed_ecg[sl], color=C_GREY, lw=2.2, label="preprocessed")
    ax[3].plot(t, clean_ecg[sl], color=C_AQUA, lw=1.1, label="clean")
    ax[3].set_title("Overlay: what the subtraction changed"); ax[3].legend(loc="upper right"); ax[3].set_xlabel("time (s)")
    for a in ax: a.set_ylabel("mV")
    return finish(fig, "9_artefact_subtraction", save, show)


if __name__ == "__main__":
    from ecg_io import parse_args
    from median_filter import median_filter
    from lpf_150hz import lpf_150hz
    from swt_haar import SWT_Haar
    from iswt_haar import ISWT_Average_Basis_Haar
    from christov_r_peak_detector import christov_r_peak_detector
    from QRS_removal import remove_qrs_complex
    from threshold_calculator import calculate_thresholds
    from multiresolution_thresholding import multiresolution_thresholding

    args = parse_args("Artefact subtraction block: produces the final clean ECG")
    fs = args.fs
    pre = lpf_150hz(median_filter(args.ecg, fs), fs)                       # blocks 1-2
    N = len(pre)
    A7, D = SWT_Haar(pre, 7)                                                # block 3
    r_peaks = christov_r_peak_detector(D, fs, n_samples=N)[0]               # block 4
    D_mod = remove_qrs_complex(D, r_peaks, fs)                              # block 5
    th = calculate_thresholds(D_mod, A7, fs, n_samples=N)                   # block 6
    D_art, A_art = multiresolution_thresholding(D_mod, A7, th)        # block 7 (A7 setting: pipeline_config.py)
    artefact = ISWT_Average_Basis_Haar((A_art, D_art), 7)[:N]              # ISWT
    clean = subtract_artefact(pre, artefact)                                # this block

    out_path = os.path.splitext(args.file)[0] + "_clean.txt"
    np.savetxt(out_path, clean, fmt="%.6f")
    print(f"{len(r_peaks)} R peaks | artefact range {artefact.min():+.3f} .. {artefact.max():+.3f} | "
          f"clean ECG written to {out_path}")
    plot_artefact_subtraction(pre, artefact, clean, fs, r_peaks, args.start, args.dur, save=args.save)
