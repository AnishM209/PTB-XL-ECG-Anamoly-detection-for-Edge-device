import numpy as np

def remove_qrs_complex(coeff_detail, r_peaks, fs=500):
    """
    Remove QRS complexes from SWT detail coefficients D1-D6.

    Input parameters:
    coeff_detail -- detail coefficients from SWT D1 - D7
    r_peaks -- Sample indices of detected R-peaks from Christov detector
               (now refined to the true R peak, see christov_r_peak_detector.py).
    fs -- Sampling frequency in Hz.

    Return objects:
        modified_coeff_detail -- QRS complex removed detail coefficients D1 to D6 with D7 unchanged
    """

    # Make copies so that the original SWT coefficients are not modified
    modified_coeff_detail = [np.array(detail, dtype=float, copy=True) for detail in coeff_detail]

    # Calculate the samples corresponding to a 50ms second window on either side of the detected R peak indices.
    window_samples = int(0.05 * fs)

    for r_peak in r_peaks:
        # Set QRS samples to zero in D1-D6
        for level in range(6):

            # Calculate the half-width of the window for the current level
            # Window size is scaled by 2^(level + 1) to account for the downsampling in SWT
            half_width = window_samples + 2 ** (level + 1)

            # Create QRS removal window around R peak
            start = max(0, r_peak - half_width)
            end = min(len(modified_coeff_detail[level]), r_peak + half_width + 1)

            modified_coeff_detail[level][start:end] = 0.0

    return modified_coeff_detail


# ============================================================================ plotting
def plot_qrs_removal(coeff_detail, modified_coeff_detail, r_peaks, fs, n_samples=None, start=0.0, dur=6.0,
                     save=False, show=True):
    """D1..D6 before (grey) and after (colour) QRS removal, with the removal windows shaded."""
    from ecg_io import setup_style, plot_window, finish, C_BLUE, C_ORANGE, C_GREY
    plt = setup_style()
    n = n_samples if n_samples is not None else len(coeff_detail[0])
    sl, t = plot_window(n, fs, start, dur)
    w = int(0.05 * fs)
    pk = np.asarray(r_peaks, dtype=int)
    fig, ax = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
    for level, a in enumerate(ax):
        hw = w + 2 ** (level + 1)
        for p in pk[(pk + hw >= sl.start) & (pk - hw < sl.stop)]:
            a.axvspan((p - hw) / fs, (p + hw) / fs, color=C_ORANGE, alpha=0.12, lw=0)
        a.plot(t, np.asarray(coeff_detail[level])[sl], color=C_GREY, lw=0.9)
        a.plot(t, np.asarray(modified_coeff_detail[level])[sl], color=C_BLUE)
        a.set_ylabel(f"D{level + 1}\n±{1000 * hw / fs:.0f} ms", rotation=0, ha="right", va="center", fontsize=9)
    ax[0].set_title("5 - QRS removal: grey = before, blue = after, shaded = zeroed window around each R peak")
    ax[-1].set_xlabel("time (s)")
    return finish(fig, "5_qrs_removal", save, show)


if __name__ == "__main__":
    from ecg_io import parse_args
    from median_filter import median_filter
    from lpf_150hz import lpf_150hz
    from swt_haar import SWT_Haar
    from christov_r_peak_detector import christov_r_peak_detector

    args = parse_args("QRS removal block (zeroes D1-D6 around each R peak)")
    pre = lpf_150hz(median_filter(args.ecg, args.fs), args.fs)
    A7, D = SWT_Haar(pre, 7)
    r_peaks = christov_r_peak_detector(D, args.fs, n_samples=len(pre))[0]
    D_mod = remove_qrs_complex(D, r_peaks, args.fs)
    plot_qrs_removal(D, D_mod, r_peaks, args.fs, n_samples=len(pre), start=args.start, dur=args.dur, save=args.save)
