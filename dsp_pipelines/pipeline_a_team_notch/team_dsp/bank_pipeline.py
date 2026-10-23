"""
Simulation of the planned hardware split (ping-pong DDR banks):

  FPGA  : raw -> median filter -> LPF -> SWT (7 levels, Haar), computed CONTINUOUSLY on the stream.
          D1..D7 and A7 are written to DDR, 2048 samples (4.096 s) per bank, alternating bank A / bank B.
  CPU   : when a bank is full, Python processes it:
          R-peak detection on D4 -> QRS removal -> thresholds -> thresholding -> ISWT -> subtraction
          and the CNN gives one inference per 2048-sample window.

Because the SWT runs continuously on the FPGA, the coefficients in a bank are exactly the coefficients
of the continuous signal -- there is no wrap-around at the bank edges (that was the main problem of
cutting the signal first and running the SWT per window). What still has to cross the bank edge:

  1. ISWT look-back: the ISWT of sample n needs coefficients n-127 .. n, so the first 127 samples of a
     bank need the last 127 coefficients of the previous bank.
  2. Beats near the end of a bank: a beat is confirmed by the detector 100 ms after its threshold
     crossing, and its QRS-removal window reaches +-178 ms (D6). So the last ~140 samples of a bank
     cannot be finalized until the first samples of the next bank are known.
  3. Detector state (thresholds M, F, R, RR history) and the threshold statistics (per-second max/min
     of the last 16 s) must carry from one bank to the next -- they are small.

  => The processor keeps a HISTORY of the last 288 coefficient samples of every band (128 for the ISWT
     look-back + 160 not-yet-final samples) in its own RAM, because the old bank is overwritten by the
     FPGA during the next cycle. Each cycle then outputs 2048 final clean samples, delayed by
     DELAY = 160 samples (0.32 s): cycle k outputs samples [2048k - 160, 2048k + 1888).

    python bank_pipeline.py                                 # default ECG file, compare with full-signal pipeline
    python bank_pipeline.py 15526_hr_V2_minus_V1.txt --save
"""
import os

import numpy as np

from median_filter import median_filter
from lpf_150hz import lpf_150hz
from swt_haar import SWT_Haar
from iswt_haar import ISWT_Average_Basis_Haar
from multiresolution_thresholding import multiresolution_thresholding
from artefact_subtraction import subtract_artefact
from streaming_pipeline import Ring, _ChristovStream
from pipeline_config import MEDIAN_WINDOW_S, A7_INCLUDED

BANK = 2048           # samples per DDR bank
LEVELS = 7
LOOKBACK = 2 ** LEVELS            # 128: ISWT look-back
DELAY = 160                       # samples held back each cycle (>= 15 + 49 + 89 = 153 needed)
HIST = LOOKBACK + DELAY           # 288 coefficient samples per band kept in processor RAM
NAMES = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "A7"]


def _qrs_half_width(level, fs):
    return int(0.05 * fs) + 2 ** (level + 1)       # same as QRS_removal.py


class BankProcessor:
    """Everything the processor keeps between bank cycles."""

    def __init__(self, fs=500, A7_included=A7_INCLUDED, K_segments=16, warmup=True, oracle_running_max=None):
        self.fs, self.A7_included, self.K = fs, A7_included, K_segments
        self.detector = _ChristovStream(fs, None if warmup else oracle_running_max)
        self.absD4 = Ring(176, "|D4| history")          # 350 ms, as in streaming_pipeline.py
        self.beats = []
        self.all_beats = []                              # only for plotting/evaluation
        self.hist = None                                 # last HIST samples of every band
        self.hist_start = 0                              # absolute index of hist[0]
        self.done = 0                                    # absolute index up to which output is final
        self.seg_hist = {n: [] for n in NAMES}           # carried per-second (max, min)

    def process_bank(self, D_bank, A7_bank, pre_bank, pre_hist, bank_start, last=False):
        fs = self.fs
        n_b = len(A7_bank)
        # ---- 1. R-peak detection continues over this bank's D4 (state carried)
        for i in range(n_b):
            g = bank_start + i
            self.absD4.push(g, abs(D_bank[3][i]))
            r = self.detector.step(g, self.absD4)
            if r is not None:
                self.beats.append(r); self.all_beats.append(r)

        # ---- 2. working buffer = carried history + this bank (all 8 bands)
        bands = [np.asarray(D_bank[j], float) for j in range(LEVELS)] + [np.asarray(A7_bank, float)]
        if self.hist is None:
            buf = [b.copy() for b in bands]; buf_start = bank_start
            pre_buf = np.asarray(pre_bank, float)
        else:
            buf = [np.r_[h, b] for h, b in zip(self.hist, bands)]; buf_start = self.hist_start
            pre_buf = np.r_[pre_hist, pre_bank]
        L = len(buf[0])

        # ---- 3. QRS removal for every confirmed beat that touches the buffer (D1..D6)
        for r in self.beats:
            for lev in range(6):
                hw = _qrs_half_width(lev, fs)
                a, b = r - hw - buf_start, r + hw + 1 - buf_start
                if b > 0 and a < L:
                    buf[lev][max(0, a):min(L, b)] = 0.0

        # ---- 4. which part becomes final this cycle
        out_from = self.done
        out_to = bank_start + n_b if last else bank_start + n_b - DELAY
        lo, hi = out_from - buf_start, out_to - buf_start

        # thresholds: per-second stats of the newly finalized coefficients + carried history
        current = {}
        for name, c in zip(NAMES, buf):
            current[name] = [(float(c[s:min(s + fs, hi)].max()), float(c[s:min(s + fs, hi)].min()))
                             for s in range(lo, hi, fs) if min(s + fs, hi) - s >= fs // 2]
        th = {}
        for name in NAMES:
            pairs = (self.seg_hist[name] + current[name])[-self.K:] or current[name]
            mx = np.array([p[0] for p in pairs]); mn = np.array([p[1] for p in pairs])
            mmx, mmn = np.median(mx), np.median(mn)
            th[name] = {"upper": mmx + np.median(np.abs(mmx - mx)), "lower": mmn - np.median(np.abs(mmn - mn))}
            self.seg_hist[name] = (self.seg_hist[name] + current[name])[-self.K:]

        # ---- 5. thresholding + ISWT (only look-back is used, so zero-padding the end is harmless)
        D_art, A_art = multiresolution_thresholding(buf[:LEVELS], buf[LEVELS], th, A7_included=self.A7_included)
        P = 1 << (L - 1).bit_length()
        pad = lambda v: np.r_[v, np.zeros(P - L)]
        art = ISWT_Average_Basis_Haar((pad(A_art), [pad(d) for d in D_art]), LEVELS)[:L]
        clean = subtract_artefact(pre_buf, art)

        # ---- 6. keep history for the next cycle (the old bank will be overwritten by the FPGA)
        self.hist = [c[-HIST:].copy() for c in bands] if self.hist is None or n_b >= HIST \
            else [np.r_[h, c][-HIST:] for h, c in zip(self.hist, bands)]
        self.hist_start = bank_start + n_b - len(self.hist[0])
        # forget beats that can no longer touch the buffer (keeps the list short on the device)
        self.beats = [r for r in self.beats if r + 2 * DELAY >= self.hist_start]
        self.done = out_to
        return out_from, out_to, clean[lo:hi], art[lo:hi], th


def bank_first_phase(raw_ecg, fs=500, A7_included=A7_INCLUDED, warmup=True, median_window=MEDIAN_WINDOW_S):
    """Runs the FPGA part on the whole stream, then feeds the processor bank by bank."""
    raw_ecg = np.asarray(raw_ecg, float)
    N = len(raw_ecg)
    pre = lpf_150hz(median_filter(raw_ecg, fs, median_window), fs)   # FPGA (continuous, centred median)
    A7, D = SWT_Haar(pre, LEVELS)                             # FPGA (continuous SWT)
    proc = BankProcessor(fs, A7_included, warmup=warmup,
                         oracle_running_max=float(np.max(np.abs(D[3][:5 * fs]))))
    clean = np.full(N, np.nan); artefact = np.full(N, np.nan); cycles = []
    for s in range(0, N, BANK):
        e = min(s + BANK, N)
        h0 = max(0, s - HIST)
        a, b, c, art, th = proc.process_bank([d[s:e] for d in D], A7[s:e], pre[s:e], pre[h0:s], s, last=(e == N))
        clean[a:b] = c; artefact[a:b] = art; cycles.append((s, a, b))
    return dict(preprocessed=pre, clean=clean, artefact=artefact, r_peaks=np.array(proc.all_beats),
                cycles=cycles, fs=fs, N=N, mode="ping-pong banks")


# ============================================================================ plotting
def plot_banks(out, reference_clean=None, save=False, show=True):
    from ecg_io import setup_style, finish, C_BLUE, C_ORANGE, C_AQUA, C_GREY, C_INK
    plt = setup_style()
    fs, N = out["fs"], out["N"]; t = np.arange(N) / fs
    rows = 3 if reference_clean is None else 4
    fig, ax = plt.subplots(rows, 1, figsize=(12, 2.4 * rows + 1), sharex=True)
    ax[0].plot(t, out["preprocessed"], color=C_BLUE); ax[0].set_title("Preprocessed ECG (FPGA)")
    ax[1].plot(t, out["artefact"], color=C_ORANGE); ax[1].set_title("Estimated artefact (processor, per bank)")
    ax[2].plot(t, out["clean"], color=C_AQUA, label="clean (bank processing)")
    pk = out["r_peaks"]; ax[2].plot(pk / fs, np.nan_to_num(out["clean"][pk]), "v", color=C_INK, ms=5, label="R peaks")
    if reference_clean is not None:
        ax[2].plot(t, reference_clean, color=C_GREY, lw=0.8, label="full-signal pipeline")
        ax[3].plot(t, out["clean"] - reference_clean, color=C_INK, lw=0.8)
        ax[3].set_title("Difference: bank processing - full-signal"); ax[3].set_ylabel("mV")
    ax[2].set_title("Clean ECG"); ax[2].legend(loc="upper right", ncol=3)
    for a in ax:
        a.axvspan(0, 5, color=C_GREY, alpha=0.12, lw=0)          # detector warm-up (output not valid)
        for (s, ka, kb) in out["cycles"]:
            a.axvline(s / fs, color=C_GREY, ls=":", lw=0.8)
    ax[-1].set_xlabel("time (s)   (dotted lines = DDR bank boundaries every 2048 samples)")
    return finish(fig, "bank_pingpong", save, show)


if __name__ == "__main__":
    from ecg_io import parse_args
    from first_phase_pipeline import first_phase
    args = parse_args("Ping-pong DDR bank simulation of the First Phase")
    full = first_phase(args.ecg, args.fs)
    out = bank_first_phase(args.ecg, args.fs)
    # the device does not accept beats during the first 5 s (R-peak detector warm-up), so compare after 6 s
    warm = 6 * args.fs
    sl = slice(warm, out["N"] - 256)
    d = (out["clean"] - full["clean"])[sl]
    n_full = int(np.sum(full["r_peaks"] >= warm)); n_bank = int(np.sum(out["r_peaks"] >= warm))
    print(f"{len(out['cycles'])} bank cycles | after the 6 s warm-up: {n_bank} R peaks (full-signal: {n_full}), "
          f"difference to full-signal clean ECG: max {np.nanmax(np.abs(d)):.3f} mV, RMS {np.sqrt(np.nanmean(d**2)):.4f} mV")
    np.savetxt(os.path.splitext(args.file)[0] + "_clean_banks.txt", out["clean"], fmt="%.6f")
    plot_banks(out, full["clean"], save=args.save)
