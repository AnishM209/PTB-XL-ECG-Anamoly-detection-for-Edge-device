import numpy as np

from pipeline_config import A7_INCLUDED

# [UPDATE] default A7_included True -> False (set in pipeline_config.py): removing all of A7 (0-2 Hz) acted like
#          a 2 Hz high-pass and distorted the ST segment / T wave the classifier needs.
def multiresolution_thresholding(mod_coeff_detail, coeff_approx, thresholds, A7_included=A7_INCLUDED):
    """
    Apply multi-resolution thresholding to D1-D7 and A7 to build the coefficients of the
    LOW-FREQUENCY MOTION ARTEFACT signal.

    Input parameters:
    mod_coeff_detail : list of arrays conaining Detail coefficients [D1, D2, D3, D4, D5, D6, D7] after QRS removal.
    coeff_approx : array containing Approximation coefficient A7.
    thresholds : Threshold dictionary returned by calculate_thresholds().
    A7_included : bool -- True: all of A7 is treated as artefact and removed.
                          False (default): A7 is thresholded like D1-D7 (only its outliers are removed).

    Return objects:
    thresholded_detail : list of arrays containing the artefact part of D1-D7.

    thresholded_approx : array containing the artefact part of A7.
    """

    # Combine D1-D7 and A7 for processing
    coefficient_names = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "A7"]
    coefficients = ( list(mod_coeff_detail) + [coeff_approx] )

    # Create a list to hold the thresholded coefficients
    thresholded_coefficients = []

    # Iterate through each coefficient and apply thresholding
    for name, coeff in zip(coefficient_names, coefficients):

        # Copy the coefficient array to avoid changing original samples
        coeff = np.array(coeff, dtype=float, copy=True)

        # [FIX] see A7_included above
        if name == "A7" and A7_included:
            thresholded_coefficients.append(coeff)
            continue

        # Retrieve the thresholds for this coefficient from the thresholds dictionary given by the threshold_calculator.py
        upper = thresholds[name]["upper"]
        lower = thresholds[name]["lower"]

        # Make the values inside the thresholds zero as it is the normal ECG signal
        # We will take the outlier noise values to take its ISWT and subtract it from the preprocessed ECG to get clean signal
        inlier_mask = (coeff <= upper) & (coeff >= lower)
        coeff[inlier_mask] = 0.0

        # Append the thresholded coefficient to the list
        thresholded_coefficients.append(coeff)

    # Separate detail coefficients from approximation coefficients
    thresholded_detail = thresholded_coefficients[:7]
    thresholded_approx = thresholded_coefficients[7]

    return thresholded_detail, thresholded_approx


# ============================================================================ plotting
def plot_thresholding(mod_coeff_detail, coeff_approx, thresholds, thresholded_detail, thresholded_approx, fs,
                      n_samples=None, start=0.0, dur=None, artefact=None, A7_included=A7_INCLUDED, save=False, show=True):
    """Each band with its threshold band; coefficients kept as artefact are highlighted.
    If `artefact` (the ISWT of the kept coefficients) is given, it is shown in the last row."""
    from ecg_io import setup_style, plot_window, finish, C_BLUE, C_ORANGE, C_AQUA
    plt = setup_style()
    names = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "A7"]
    ins = list(mod_coeff_detail) + [coeff_approx]
    outs = list(thresholded_detail) + [thresholded_approx]
    n = n_samples if n_samples is not None else len(coeff_approx)
    sl, t = plot_window(n, fs, start, dur if dur is not None else n / fs)
    rows = len(names) + (artefact is not None)
    fig, ax = plt.subplots(rows, 1, figsize=(12, 1.25 * rows + 1), sharex=True)
    for a, name, c_in, c_out in zip(ax, names, ins, outs):
        c_in, c_out = np.asarray(c_in)[sl], np.asarray(c_out)[sl]
        a.plot(t, c_in, color=C_BLUE, lw=0.8)
        if name == "A7" and A7_included:
            a.plot(t, c_out, color=C_ORANGE, lw=1.2)
            kept_txt = "all kept"
        else:
            a.axhspan(thresholds[name]["lower"], thresholds[name]["upper"], color=C_BLUE, alpha=0.08, lw=0)
            k = c_out != 0
            a.plot(t[k], c_in[k], ".", color=C_ORANGE, ms=3)
            kept_txt = f"{100 * k.mean():.1f}% kept"
        a.set_ylabel(f"{name}\n{kept_txt}", rotation=0, ha="right", va="center", fontsize=8.5)
    if artefact is not None:
        ax[-1].plot(t, np.asarray(artefact)[sl], color=C_AQUA)
        ax[-1].set_ylabel("ISWT\n= artefact", rotation=0, ha="right", va="center", fontsize=8.5)
    ax[0].set_title("7 - Multiresolution thresholding: shaded = normal range (set to 0), orange = kept as motion artefact")
    ax[-1].set_xlabel("time (s)")
    return finish(fig, "7_thresholding", save, show)


if __name__ == "__main__":
    from ecg_io import parse_args
    from median_filter import median_filter
    from lpf_150hz import lpf_150hz
    from swt_haar import SWT_Haar
    from iswt_haar import ISWT_Average_Basis_Haar
    from christov_r_peak_detector import christov_r_peak_detector
    from QRS_removal import remove_qrs_complex
    from threshold_calculator import calculate_thresholds

    args = parse_args("Multiresolution thresholding block")
    pre = lpf_150hz(median_filter(args.ecg, args.fs), args.fs)
    N = len(pre)
    A7, D = SWT_Haar(pre, 7)
    r_peaks = christov_r_peak_detector(D, args.fs, n_samples=N)[0]
    D_mod = remove_qrs_complex(D, r_peaks, args.fs)
    th = calculate_thresholds(D_mod, A7, args.fs, n_samples=N)
    D_art, A_art = multiresolution_thresholding(D_mod, A7, th)
    artefact = ISWT_Average_Basis_Haar((A_art, D_art), 7)[:N]
    dur = args.dur if "--dur" in __import__("sys").argv else None    # default: whole record
    plot_thresholding(D_mod, A7, th, D_art, A_art, args.fs, n_samples=N, start=args.start, dur=dur,
                      artefact=artefact, save=args.save)
