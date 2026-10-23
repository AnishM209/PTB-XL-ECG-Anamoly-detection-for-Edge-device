"""
Inverse Stationary Wavelet Transform (ISWT, average-basis), Haar wavelet -- the ISWT block of
the First Phase. (Split out of SWT_python.py; the forward transform is in swt_haar.py.
The maths is unchanged.)

Run on its own to check perfect reconstruction on your ECG:
    python iswt_haar.py [ecg.txt] [--start 0 --dur 6 --save]
"""
import numpy as np

# Inverse SWT Function (Average Basis Inverse)
def ISWT_Average_Basis_Haar(coeff,l):
    """
        Calculates the Average Basis Inverse SWT (ISWT) for the SWT coefficients, assuming starting level l

        Parameters
        - coeff, an ordered pair, whose first element is the array of N approximation coefficients (N should be a power of 2),
          and the second element is the list of N-sized arrays of detail coefficients
    """

    # Finding smallest power of 2 greater than or equal to array's length
    coeff_appx, coeff_detail = coeff

    coeff_appx = np.array(coeff_appx,dtype=float)
    coeff_detail = np.array(coeff_detail,dtype=float)
    N = len(coeff_appx)

    assert N > 0 and (N & (N-1)) == 0, "Input Approximation Coefficient array not of length of a power of 2!"

    J = N.bit_length()-1

    if l == 0:
        return coeff_appx

    # Haar high pass and low pass filters
    h_phi = np.array([1/np.sqrt(2),1/np.sqrt(2)])
    h_psi = np.array([1/np.sqrt(2),-1/np.sqrt(2)])

    l_h_phi = len(h_phi)
    l_h_psi = len(h_psi)

    j = J-l
    stride = pow(2,l-1)
    for i in range(-1,-l-1,-1):
        h_lpf_upsampled = np.zeros((len(h_phi)-1)*stride+1)
        h_lpf_upsampled[::stride] = h_phi

        h_hpf_upsampled = np.zeros((len(h_psi)-1)*stride+1)
        h_hpf_upsampled[::stride] = h_psi

        # Applying the Low Pass Filter
        # Interpolating (Inserting zeros)
        # Padding in front to aid circular convolution, instead of at the back for forward transform
        coeff_appx_next1 = np.pad(coeff_appx,(len(h_lpf_upsampled)-1,0),mode="wrap")
        coeff_appx_next1 = np.convolve(h_lpf_upsampled,coeff_appx_next1,mode="valid")

        # Applying the High Pass Filter
        # Interpolating (Inserting zeros)
        # Padding in front to aid circular convolution, instead of at the back for forward transform
        coeff_appx_next = np.pad(coeff_detail[i],(len(h_hpf_upsampled)-1,0),mode="wrap")
        coeff_appx_next = np.convolve(h_hpf_upsampled,coeff_appx_next,mode="valid")

        coeff_appx = (coeff_appx_next + coeff_appx_next1)/2

        stride//=2

    return coeff_appx


# ============================================================================ plotting
def plot_iswt(original, reconstructed, fs, start=0.0, dur=6.0, title=None, save=False, show=True, name="iswt",
              labels=("SWT input", "ISWT output")):
    """Input vs ISWT output and their difference."""
    from ecg_io import setup_style, plot_window, finish, C_BLUE, C_ORANGE, C_GREY
    plt = setup_style()
    n = len(original)
    rec = np.asarray(reconstructed)[:n]
    sl, t = plot_window(n, fs, start, dur)
    fig, ax = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(t, np.asarray(original)[sl], color=C_GREY, lw=2.2, label=labels[0])
    ax[0].plot(t, rec[sl], color=C_BLUE, label=labels[1])
    ax[0].set_title(title or "ISWT -- reconstruction of the SWT input"); ax[0].legend(loc="upper right")
    ax[1].plot(t, (rec - original)[sl], color=C_ORANGE)
    ax[1].set_ylabel("difference"); ax[1].set_xlabel("time (s)")
    return finish(fig, name, save, show)


if __name__ == "__main__":
    from ecg_io import parse_args
    from median_filter import median_filter
    from lpf_150hz import lpf_150hz
    from swt_haar import SWT_Haar

    args = parse_args("ISWT block: checks that ISWT(SWT(x)) == x on the preprocessed ECG")
    pre = lpf_150hz(median_filter(args.ecg, args.fs), args.fs)
    rec = ISWT_Average_Basis_Haar(SWT_Haar(pre, 7), 7)[:len(pre)]
    err = np.max(np.abs(rec - pre))
    print(f"max |ISWT(SWT(x)) - x| = {err:.2e}  ({'perfect reconstruction' if err < 1e-9 else 'CHECK!'})")
    plot_iswt(pre, rec, args.fs, args.start, args.dur,
              title=f"ISWT(SWT(x)) vs x -- max error {err:.1e}", save=args.save)
