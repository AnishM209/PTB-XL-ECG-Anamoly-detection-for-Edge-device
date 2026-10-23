import numpy as np
from scipy.signal import medfilt

from pipeline_config import MEDIAN_WINDOW_S

# [UPDATE] default window 0.3 s -> 0.6 s (set in pipeline_config.py). The 0.3 s window shrank the T wave to
#          52-76 % and shifted the ST level by up to 0.1 mV, making normal beats look abnormal to the CNN.
def median_filter( raw_ecg_sig, fs, window_size = MEDIAN_WINDOW_S):
    """
    TASK: Remove baseline wander from the raw ECG signal taken as per sample value in array type.
          A sliding-window median (0.6 s by default, see pipeline_config.py) is used to track the baseline. That baseline estimate
          is then subtracted from the raw signal:

    Input parameters:
        raw_ecg_sig : array -- The raw ECG signal to be filtered.
        fs : integer -- The sampling frequency of the ECG signal.
        window_size : float -- The size of the median filter window in seconds. Default MEDIAN_WINDOW_S = 0.6 s.

    Return objects
        median_ecg_sig : array -- The baseline-corrected ECG signal (raw signal minus its median baseline).
    """

    # make sure the input is a float numpy array
    raw_ecg_sig = np.asarray(raw_ecg_sig, dtype=float)

    # Calculate the window size in samples
    window_size_samples = max(1, int(round(window_size * fs)))

    # Ensure the window size is odd for the median filter to work correctly
    if window_size_samples % 2 == 0:
        window_size_samples = window_size_samples + 1

    # Apply median filter to get baseline wander estimate
    baseline = medfilt(raw_ecg_sig, kernel_size=window_size_samples)

    # Subtract the baseline from the raw signal instead of returning the baseline itself
    median_ecg_sig = raw_ecg_sig - baseline

    return median_ecg_sig


# ============================================================================ plotting
def plot_median_filter(raw_ecg_sig, median_ecg_sig, fs, start=0.0, dur=6.0, save=False, show=True,
                       window_size=MEDIAN_WINDOW_S):
    """Raw ECG with the median baseline estimate on top, and the baseline-corrected output below."""
    from ecg_io import setup_style, plot_window, finish, C_BLUE, C_ORANGE, C_GREY
    plt = setup_style()
    raw = np.asarray(raw_ecg_sig, dtype=float)
    sl, t = plot_window(len(raw), fs, start, dur)
    baseline = raw - median_ecg_sig
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(t, raw[sl], color=C_GREY, label="raw ECG")
    ax[0].plot(t, baseline[sl], color=C_ORANGE, lw=1.8, label=f"baseline = medfilt(x, {window_size:g} s)")
    ax[0].set_title("1 - Median filter: baseline estimate"); ax[0].legend(loc="upper right")
    ax[1].plot(t, median_ecg_sig[sl], color=C_BLUE, label="output = x - medfilt(x)")
    ax[1].axhline(0, color=C_GREY, lw=0.8)
    ax[1].set_title("Baseline-corrected ECG (median_filter output)"); ax[1].legend(loc="upper right")
    ax[1].set_xlabel("time (s)")
    for a in ax: a.set_ylabel("amplitude")
    return finish(fig, "1_median_filter", save, show)


if __name__ == "__main__":
    from ecg_io import parse_args
    args = parse_args("Median filter block (baseline wander removal)")
    out = median_filter(args.ecg, args.fs)
    plot_median_filter(args.ecg, out, args.fs, args.start, args.dur, save=args.save)
