import numpy as np


# [FIX] NEW optional argument n_samples = length of the ORIGINAL ECG (before SWT zero-padding).
#       The SWT pads 10000 samples to 16384, so the old code split 33 one-second segments of which
#       12 were pure zero padding. Their max = min = 0 dragged the medians towards zero and gave
#       thresholds that were far too tight (e.g. D2 upper 0.16 instead of 0.23). Pass len(ecg).
def calculate_thresholds(coeff_detail, coeff_approx_A7, fs=500, n_samples=None):
    """
    Calculate upper and lower thresholds t_max and t_min for multi-resolution
    thresholding block.

    Input parameters:
    coeff_detail : list of numpy arrays
        SWT detail coefficients after QRS removal: [D1, D2, D3, D4, D5, D6, D7]
    coeff_approx_A7 : numpy array
        SWT approximation coefficient A7.
    fs : int
        Sampling frequency in Hz.
    n_samples : int or None 
        Number of real ECG samples. Only coeff[:n_samples] is used.

    Return objects:
    thresholds : Dictionary containing upper and lower thresholds for D1 to D7 and A7.
    """

    segment_length = fs
    # Combine D1-D7 and A7
    coefficients = list(coeff_detail) + [coeff_approx_A7]
    coefficient_names = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "A7"]

    # Create threshold dictionary to hold upper and lower thresholds for each coefficient
    thresholds = {}

    for name, coeff in zip(coefficient_names, coefficients):

        coeff = np.asarray(coeff, dtype=float)
        if n_samples is not None:
            coeff = coeff[:n_samples]     

        segment_maxima = []
        segment_minima = []

        for start in range(0, len(coeff), segment_length):
            # Do the maxima and minima calculation for each segment of 1 second for each coefficient
            end = min(start + segment_length, len(coeff))

            # Reject very short segments that bias the overall min max and median to fall.
            if end - start < segment_length // 2:
                continue
            segment = coeff[start:end]

            segment_maxima.append(np.max(segment))
            segment_minima.append(np.min(segment))

        segment_maxima = np.asarray(segment_maxima)
        segment_minima = np.asarray(segment_minima)

        # Now find the upper and lower thresholds based on the calculated segment maxima and minima
        # First find the median of the segment maxima and minima
        # Then calculate the median absolute deviation (MAD) for both maxima and minima
        # Then obain the upper threshold t_max and lower threshold t_min using the median and MAD values
        median_max = np.median(segment_maxima)
        mad_maxima = np.median(np.abs(median_max - segment_maxima))

        t_max = median_max + mad_maxima

        median_min = np.median(segment_minima)
        mad_minima = np.median(np.abs(median_min - segment_minima))

        t_min = median_min - mad_minima

        # Store the t_max and t_min values for the each coefficient in the thresholds dictionary
        thresholds[name] = {"upper": t_max, "lower": t_min}

    return thresholds


# ============================================================================ plotting
def plot_thresholds(coeff_detail, coeff_approx_A7, thresholds, fs=500, n_samples=None, save=False, show=True):
    """For every band: the max and min of each 1 s segment, and the resulting upper/lower thresholds."""
    from ecg_io import setup_style, finish, C_BLUE, C_ORANGE, C_INK
    plt = setup_style()
    names = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "A7"]
    coeffs = list(coeff_detail) + [coeff_approx_A7]
    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    for a, name, c in zip(axes.T.ravel(), names, coeffs):
        c = np.asarray(c, dtype=float)
        if n_samples is not None:
            c = c[:n_samples]
        segs = [c[s:s + fs] for s in range(0, len(c), fs) if len(c[s:s + fs]) >= fs // 2]
        k = np.arange(len(segs))
        a.plot(k, [s.max() for s in segs], "^", color=C_BLUE, ms=5, label="segment max")
        a.plot(k, [s.min() for s in segs], "v", color=C_ORANGE, ms=5, label="segment min")
        a.axhline(thresholds[name]["upper"], color=C_BLUE, lw=1.2, ls="--", label="upper threshold")
        a.axhline(thresholds[name]["lower"], color=C_ORANGE, lw=1.2, ls="--", label="lower threshold")
        a.axhline(0, color=C_INK, lw=0.5)
        a.set_title(f"{name}:  [{thresholds[name]['lower']:+.3f}, {thresholds[name]['upper']:+.3f}]")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4)
    for a in axes[-1]: a.set_xlabel("1-second segment")
    fig.suptitle("6 - Threshold calculation: median ± MAD of the per-second maxima / minima",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig._layout_done = True
    return finish(fig, "6_thresholds", save, show)


if __name__ == "__main__":
    from ecg_io import parse_args
    from median_filter import median_filter
    from lpf_150hz import lpf_150hz
    from swt_haar import SWT_Haar
    from christov_r_peak_detector import christov_r_peak_detector
    from QRS_removal import remove_qrs_complex

    args = parse_args("Threshold calculation block")
    pre = lpf_150hz(median_filter(args.ecg, args.fs), args.fs)
    N = len(pre)
    A7, D = SWT_Haar(pre, 7)
    r_peaks = christov_r_peak_detector(D, args.fs, n_samples=N)[0]
    D_mod = remove_qrs_complex(D, r_peaks, args.fs)
    th = calculate_thresholds(D_mod, A7, args.fs, n_samples=N)
    for k, v in th.items():
        print(f"  {k}: lower {v['lower']:+.4f}   upper {v['upper']:+.4f}")
    plot_thresholds(D_mod, A7, th, args.fs, n_samples=N, save=args.save)
