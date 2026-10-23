"""
Forward Stationary Wavelet Transform (SWT), Haar wavelet -- block 3 of the First Phase.
(Split out of SWT_python.py; the inverse is in iswt_haar.py. The maths is unchanged.)

Run on its own to see all detail/approximation levels:
    python swt_haar.py [ecg.txt] [--start 0 --dur 6 --save]
"""
import numpy as np

# Forward SWT Function
def SWT_Haar(sig,l):
    """
        Calculates the SWT coefficients of the given array, with Haar Wavelet
        Pads the array with zeros to make length as power of 2

        Parameters
        - The signal "sig", length N, internally updated to the smallest power of 2 greater than or equal to N
        - l, the required detail/approximation level, with l=0 gives the actual signal samples

        Returns an ordered pair, whose first element is the array of N approximation coefficients (N should be a power of 2),
        and the second element is the list of N-sized arrays of detail coefficients

        [NOTE] The output length is the padded power-of-2 length (e.g. 10000 -> 16384). The extra
        samples are zero padding, not ECG. Downstream blocks (R-peak detector, threshold calculator)
        must be given the original length N via their `n_samples` argument so the padding is ignored,
        and the ISWT output must be cut back to [:N].

        Verified: identical to pywt.swt(..., 'haar', norm=False) and perfect reconstruction (error ~1e-16).
    """
    # Finding smallest power of 2 greater than or equal to array's length
    sig = np.array(sig,dtype=float)
    N = len(sig)
    N1 = 1<<(N-1).bit_length()
    J = N1.bit_length()-1

    sig = np.pad(sig,(0,N1-N))
    N = N1

    if l == 0:
        return sig, []

    # Number of approximation and detail coefficients in the output
    N_appx = N/pow(2,l)
    N_detail = N-N_appx

    # Haar high pass and low pass filters
    h_phi = np.array([1/np.sqrt(2),1/np.sqrt(2)])
    h_psi = np.array([1/np.sqrt(2),-1/np.sqrt(2)])

    l_h_phi = len(h_phi)
    l_h_psi = len(h_psi)

    # Time Reversing (for Forward Transform) and Shifting for Causality
    h_lpf = h_phi[::-1]
    h_hpf = h_psi[::-1]

    # Coefficients array
    coeff_appx = sig
    coeff_detail = []

    stride = 1
    for i in range(l):
        h_lpf_upsampled = np.zeros((len(h_lpf)-1)*stride+1)
        h_lpf_upsampled[::stride] = h_lpf

        h_hpf_upsampled = np.zeros((len(h_hpf)-1)*stride+1)
        h_hpf_upsampled[::stride] = h_hpf

        # Applying the Low Pass Filter
        coeff_appx_next = np.pad(coeff_appx,(0,len(h_lpf_upsampled)-1),mode="wrap")
        coeff_appx_next = np.convolve(h_lpf_upsampled,coeff_appx_next,mode="valid")
        # Decimating by factor of 2
        # coeff_appx = coeff_appx[l_h_phi-1::2]

        # Applying the High Pass Filter
        coeff_detail_next = np.pad(coeff_appx,(0,len(h_hpf_upsampled)-1),mode="wrap")
        coeff_detail_next = np.convolve(h_hpf_upsampled,coeff_detail_next,mode="valid")
        # Decimating by factor of 2
        # coeff_detail = coeff_detail[l_h_psi-1::2]

        coeff_appx = coeff_appx_next
        coeff_detail.append(coeff_detail_next)

        stride*=2

    return coeff_appx, coeff_detail


# ============================================================================ plotting
def plot_swt(coeff_appx, coeff_detail, fs, n_samples=None, start=0.0, dur=6.0, input_sig=None,
             save=False, show=True):
    """One row per band: (input), D1 ... DL, AL. Each row has its own y-scale."""
    from ecg_io import setup_style, plot_window, finish, C_BLUE, C_ORANGE, C_GREY
    plt = setup_style()
    L = len(coeff_detail)
    n = n_samples if n_samples is not None else len(coeff_appx)
    sl, t = plot_window(n, fs, start, dur)
    bands = ([("input", input_sig)] if input_sig is not None else []) + \
            [(f"D{j + 1}", coeff_detail[j]) for j in range(L)] + [(f"A{L}", coeff_appx)]
    fig, ax = plt.subplots(len(bands), 1, figsize=(12, 1.15 * len(bands) + 1), sharex=True)
    for a, (name, c) in zip(ax, bands):
        lo_hz, hi_hz = (fs / 2 ** (int(name[1:]) + 1), fs / 2 ** int(name[1:])) if name[0] == "D" else \
                       ((0, fs / 2 ** (int(name[1:]) + 1)) if name[0] == "A" else (None, None))
        col = C_GREY if name == "input" else (C_ORANGE if name == "D4" else C_BLUE)
        a.plot(t, np.asarray(c)[sl], color=col)
        label = name if lo_hz is None else f"{name}  {lo_hz:.3g}-{hi_hz:.3g} Hz"
        a.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    ax[0].set_title(f"SWT (Haar, {L} levels) -- D4 (orange) is used for R-peak detection")
    ax[-1].set_xlabel("time (s)")
    return finish(fig, "3_swt", save, show)


if __name__ == "__main__":
    from ecg_io import parse_args
    from median_filter import median_filter
    from lpf_150hz import lpf_150hz

    args = parse_args("SWT block: plots D1..D7 and A7 of the preprocessed ECG")
    pre = lpf_150hz(median_filter(args.ecg, args.fs), args.fs)
    A7, D = SWT_Haar(pre, 7)
    print(f"SWT: {len(D)} detail levels, each {len(A7)} long (input padded from {len(pre)})")
    plot_swt(A7, D, args.fs, n_samples=len(pre), start=args.start, dur=args.dur, input_sig=pre, save=args.save)
