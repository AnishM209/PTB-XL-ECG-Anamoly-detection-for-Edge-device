import numpy as np

class ChristovRPeakDetector:

    # Sample rate 500 Hz consistent with the ambulatory ECG data paper.
    def __init__(self, fs=500):
        self.fs = fs

        # Convert time windows to corresponding sample counts
        self.win_50ms = int(0.05 * fs)
        self.win_200ms = int(0.20 * fs)
        self.win_350ms = int(0.35 * fs)
        self.win_1200ms = int(1.20 * fs)

        self.F_divisor = 2.0 * (self.win_350ms - self.win_50ms)

        # According to Christov's method, between 200ms and 1200ms after the R-peak, the threshold decays linearly from 100% to 60% of the average M value
        self.decay_curve = np.linspace(1.0, 0.6, self.win_1200ms - self.win_200ms)

        # Initialize thresholds and buffers for M that is the steep slope threshold
        self.M = 0.0
        # List to store the last 5 M estimates for averaging
        self.M_buffer = []
        self.pending_M = 0.0
        self.running_max = -np.inf

        # Initialize F, that is the integrating noise threshold
        self.F = 0.0

        # Initialize buffers for R, R to R intervals, and expected RR interval which is the beat expectation threshold
        self.R = 0.0
        # Buffer to store the last 5 RR intervals
        self.RR_buffer = []
        # Mean of the last 5 RR intervals to estimate the expected RR interval
        self.expected_rr = 0

        # Initialize list to store detected beat sample indices
        self.beats = []

    # Push an estimate of the M value to the buffer
    def push_M_estimate(self, value):
        self.M_buffer.append(value)
        if len(self.M_buffer) > 5:
            self.M_buffer.pop(0)
    # Update the M threshold based on the current signal and detected beats
    def update_M(self, i, x_abs):
        # Calculate the number of samples since the last detected beat
        samples_since_last = (i - self.beats[-1]) if self.beats else None

        # If we are within the first 5 seconds of the signal, we can set M to 60% of the running maximum of the signal
        if i < 5 * self.fs:
            self.M = 0.6 * self.running_max
            self.push_M_estimate(self.M)

        # If we are within 200ms of the last detected beat,
        # we can calculate a new candidate M value based on the maximum of the signal in that window
        elif self.beats and samples_since_last < self.win_200ms:
            # If we are within 200ms of the last detected beat, we can calculate a new candidate M value
            candidate = 0.6 * np.max(x_abs[self.beats[-1]:i + 1])
            # Prevent the candidate from being too high compared to the last M estimate
            if candidate > 1.5 * self.M_buffer[-1]:
                candidate = 1.1 * self.M_buffer[-1]
            self.pending_M = candidate

        # If we are exactly 200ms after the last detected beat,
        # we can push the pending M value to the buffer and update M to the average of the last 5 estimates
        elif self.beats and samples_since_last == self.win_200ms:
            estimate = self.pending_M if self.pending_M != 0 else self.M_buffer[-1]
            self.push_M_estimate(estimate)
            # Update M to the average of the last 5 estimates
            self.M = np.mean(self.M_buffer)

        # If we are between 200ms and 1200ms after the last detected beat, we apply a decay to the average M value
        elif self.beats and self.win_200ms < samples_since_last < self.win_1200ms:
            self.M = np.mean(self.M_buffer) * self.decay_curve[samples_since_last - self.win_200ms]

        # If we are more than 1200ms after the last detected beat, we can reset M to 60% of the average of the last 5 estimates
        elif self.beats and samples_since_last >= self.win_1200ms:
            self.M = 0.6 * np.mean(self.M_buffer)

        # This above logic updates M based on the signal values and characteristics as defined in Christov's method.

    # Update the F threshold based on the current signal
    def update_F(self, i, x_abs):
        # let 350 ms window be passed for F calculation
        if i <= self.win_350ms:
            return
        # Calculate the maximum values in the latest and earliest 50 ms segments of the 350 ms window
        window = x_abs[i - self.win_350ms:i]
        max_latest = np.max(window[-self.win_50ms:])
        max_earliest = np.max(window[:self.win_50ms])

        # Update F based on the difference between the latest and earliest maximum values, scaled by the F divisor
        self.F += (max_latest - max_earliest) / self.F_divisor

    # Update the R threshold based on the current signal and detected beats
    def update_R(self, i):
        if not self.beats:
            return
        samples_since_last = i - self.beats[-1]
        # Create lower and upper bounds for expected R to R intervals based on the average of the last 5 detected beats
        lower = int((2 / 3) * self.expected_rr)
        upper = self.expected_rr

        # Too soon to expect a beat, set R to 0.0
        if samples_since_last < lower:
            self.R = 0.0
        # We are in the expected range of the next beat so set R to mean of M buffer divided by 1.4
        elif lower <= samples_since_last <= upper:
            self.R = (self.M - np.mean(self.M_buffer)) / 1.4

    # Register a detected beat and update the RR interval buffer
    def register_beat(self, i):
        self.beats.append(i)
        # Update the RR interval buffer with the latest RR interval if there is more than one detected beat
        if len(self.beats) > 1:
            self.RR_buffer.append(self.beats[-1] - self.beats[-2])
            # Maintain only the last 5 RR intervals in the buffer
            if len(self.RR_buffer) > 5:
                self.RR_buffer.pop(0)
            self.expected_rr = int(np.mean(self.RR_buffer))

    # Testing fix: refine the detected beat indices to the actual R peak by searching for
    # the maximum in a 100 ms window after each detected beat
    # This is necessary because the Christov method detects the threshold crossing, not the actual peak.
    # Otherwise the QRS removal block was removing the wrong portion of the R peak, leading to false negatives in the next beat detection.
    def refine_to_peak(self, x_abs, search_ms=100):
        search = int(search_ms * 1e-3 * self.fs)
        refined = []
        for b in self.beats:
            seg = x_abs[b:min(len(x_abs), b + search)]
            refined.append(b + int(np.argmax(seg)))
        return refined

    # Take the abs of D4 coefficient and run the R peak detection algorithm
    def run(self, x_abs):
        self.running_max = np.max(x_abs[:5 * self.fs])

        # To plot and check M,F,R,MFR initialize lists
        M_hist, F_hist, R_hist, MFR_hist = [], [], [], []

        for i in range(len(x_abs)):
            self.update_M(i, x_abs)
            self.update_F(i, x_abs)
            self.update_R(i)

            # Calculate the combined threshold MFR
            MFR = self.M + self.F + self.R

            M_hist.append(self.M)
            F_hist.append(self.F)
            R_hist.append(self.R)
            MFR_hist.append(MFR)

            samples_since_last = (i - self.beats[-1]) if self.beats else None

            # If no beat has been detected yet, check if the current detail coefficient value exceeds the combined threshold and register a beat.
            if not self.beats:
                if x_abs[i] > MFR:
                    self.register_beat(i)
            # If we are more than 200ms after the last detected beat, check if the current value exceeds the combined threshold and register a beat.
            elif samples_since_last > self.win_200ms:
                if x_abs[i] > MFR:
                    self.register_beat(i)

        detected_beats = self.refine_to_peak(x_abs)

        # Return the detected beats and the history of M, F, R, and MFR for analysis or plotting
        return (
            np.array(detected_beats, dtype=int),
            np.array(M_hist),
            np.array(F_hist),
            np.array(R_hist),
            np.array(MFR_hist),
        )

# Take D4 coefficients from the previous block of SWT and run R peak detection algorithm based on Christov's method
# Returns the detected R peak indices, and additionally the history of M, F, R, and MFR for analysis or plotting.
def christov_r_peak_detector(coeff_detail, fs=500, n_samples=None):
    D4 = np.asarray(coeff_detail[3], dtype=float)

    # Remove the zero-padding from D4 if n_samples is provided, to avoid false detections in the padded region
    if n_samples is not None:
        D4 = D4[:n_samples]     
    D4_abs = np.abs(D4)
    R_peak_detector = ChristovRPeakDetector(fs=fs)
    return R_peak_detector.run(D4_abs)


# ============================================================================ plotting
def plot_christov(ecg, D4, r_peaks, M_hist, F_hist, R_hist, MFR_hist, fs, start=0.0, dur=6.0,
                  save=False, show=True):
    """|D4| with the adaptive thresholds M, F, R and M+F+R, and the detected R peaks on the ECG."""
    from ecg_io import setup_style, plot_window, finish, C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_GREY, C_INK
    plt = setup_style()
    n = len(MFR_hist)
    D4_abs = np.abs(np.asarray(D4, dtype=float)[:n])
    ecg = np.asarray(ecg, dtype=float)[:n]
    sl, t = plot_window(n, fs, start, dur)
    pk = np.asarray(r_peaks, dtype=int)
    pk_w = pk[(pk >= sl.start) & (pk < sl.stop)]
    fig, ax = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2, 2]})
    ax[0].plot(t, D4_abs[sl], color=C_BLUE, label="|D4|")
    ax[0].plot(t, MFR_hist[sl], color=C_ORANGE, lw=1.6, label="threshold M + F + R")
    ax[0].plot(pk_w / fs, D4_abs[pk_w], "v", color=C_INK, ms=7, label="detected R peak")
    ax[0].set_ylim(top=1.4 * max(D4_abs[sl].max(), MFR_hist[sl].max()))
    ax[0].set_title(f"4 - Christov R-peak detection on |D4|: {len(pk)} beats"); ax[0].legend(loc="upper right", ncol=3)
    ax[1].plot(t, M_hist[sl], color=C_AQUA, label="M (steep-slope)")
    ax[1].plot(t, F_hist[sl], color=C_YELLOW, label="F (integrating)")
    ax[1].plot(t, R_hist[sl], color=C_GREY, label="R (beat expectation)")
    ax[1].set_title("Threshold components"); ax[1].legend(loc="upper right", ncol=3)
    ax[2].plot(t, ecg[sl], color=C_GREY, label="ECG (preprocessed)")
    ax[2].plot(pk_w / fs, ecg[pk_w], "o", mfc="none", mec=C_ORANGE, ms=8, mew=1.5, label="R peaks")
    ax[2].set_title("R peaks on the ECG"); ax[2].legend(loc="upper right"); ax[2].set_xlabel("time (s)")
    if len(pk) > 1:
        rr = np.diff(pk) / fs
        ax[2].text(0.01, 0.95, f"mean HR {60 / rr.mean():.1f} bpm, RR {rr.min():.2f}-{rr.max():.2f} s",
                   transform=ax[2].transAxes, va="top", fontsize=8.5)
    return finish(fig, "4_christov_r_peaks", save, show)


if __name__ == "__main__":
    from ecg_io import parse_args
    from median_filter import median_filter
    from lpf_150hz import lpf_150hz
    from swt_haar import SWT_Haar

    args = parse_args("Christov R-peak detector block (on SWT D4)")
    pre = lpf_150hz(median_filter(args.ecg, args.fs), args.fs)
    A7, D = SWT_Haar(pre, 7)
    r_peaks, M_h, F_h, R_h, MFR_h = christov_r_peak_detector(D, args.fs, n_samples=len(pre))
    print(f"Detected {len(r_peaks)} R peaks: {r_peaks.tolist()}")
    plot_christov(pre, D[3], r_peaks, M_h, F_h, R_h, MFR_h, args.fs, args.start, args.dur, save=args.save)
