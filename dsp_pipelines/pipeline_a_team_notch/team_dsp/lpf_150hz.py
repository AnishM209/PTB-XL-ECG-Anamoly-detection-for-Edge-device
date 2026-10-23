import numpy as np
import scipy.signal as scpsig

def lpf_150hz(median_ecg_sig, fs, order=4):
    """
    TASK: Apply low pass filter to the median filtered ECG signal taken as per sample value in array type.
          Main objective is to remove high frequency noise (above 150 Hz) from the median filtered ECG signal.
    
    Input parameters:
        median_ecg_sig : array -- The median filtered ECG signal to be filtered.
        fs : integer -- The sampling frequency of the ECG signal (must be > 300 Hz for a 150 Hz cut-off).

    Return objects
        lpf_ecg_sig : array -- The filtered ECG signal after applying the low pass filter.
    """

    # Define filter parameters
    fc = 150

    # Normalize the cutoff frequency with respect to the Nyquist frequency
    nyquist_freq = 0.5 * fs
    normal_cutoff = fc / nyquist_freq
    b, a = scpsig.butter(order, normal_cutoff, btype='lowpass', analog=False)

    # Apply the low pass filter to the median filtered ECG signal
    lpf_ecg_sig = scpsig.filtfilt(b, a, median_ecg_sig)

    return lpf_ecg_sig


# ============================================================================ plotting
def plot_lpf(median_ecg_sig, lpf_ecg_sig, fs, start=0.0, dur=6.0, save=False, show=True):
    """Time signal before/after the LPF, and the power spectrum before/after."""
    from ecg_io import setup_style, plot_window, finish, C_BLUE, C_GREY, C_INK
    plt = setup_style()
    sl, t = plot_window(len(median_ecg_sig), fs, start, dur)
    f, P_in = scpsig.welch(median_ecg_sig, fs, nperseg=min(4096, len(median_ecg_sig)))
    _, P_out = scpsig.welch(lpf_ecg_sig, fs, nperseg=min(4096, len(lpf_ecg_sig)))
    fig, ax = plt.subplots(2, 1, figsize=(12, 6.5))
    ax[0].plot(t, median_ecg_sig[sl], color=C_GREY, lw=1.8, label="LPF input (median filter output)")
    ax[0].plot(t, lpf_ecg_sig[sl], color=C_BLUE, label="LPF output")
    ax[0].set_title("2 - Low-pass filter 150 Hz: time domain"); ax[0].legend(loc="upper right")
    ax[0].set_xlabel("time (s)")
    ax[1].semilogy(f, P_in, color=C_GREY, label="input"); ax[1].semilogy(f, P_out, color=C_BLUE, label="output")
    ax[1].axvline(150, color=C_INK, ls=":", lw=1); ax[1].text(152, P_in.max() / 10, "150 Hz", fontsize=8.5)
    ax[1].set_xlim(0, fs / 2); ax[1].set_xlabel("frequency (Hz)"); ax[1].set_ylabel("power spectral density")
    ax[1].set_title("Spectrum before / after"); ax[1].legend(loc="upper right")
    return finish(fig, "2_lpf_150hz", save, show)


if __name__ == "__main__":
    from ecg_io import parse_args
    from median_filter import median_filter
    args = parse_args("150 Hz low-pass filter block")
    med = median_filter(args.ecg, args.fs)
    out = lpf_150hz(med, args.fs)
    plot_lpf(med, out, args.fs, args.start, args.dur, save=args.save)
