"""
Sample-by-sample (streaming) implementation of the First Phase pipeline, written to find and
PROVE the minimum amount of past data every block must keep on a wearable/processor.

Every buffer is a fixed-size ring (class Ring). A ring raises an error if any block ever asks
for a sample that has already been overwritten, so if the pipeline runs to the end, the ring
sizes below are sufficient. Running with a ring one sample smaller makes it fail, so they are
also the minimum.

Time base: p = index of the newest preprocessed sample (median + LPF output).
Everything below is at fs = 500 Hz, 7 SWT levels, Haar wavelet.

Block           what it needs from the past                                   samples
-------------   -----------------------------------------------------------   -------
median 0.6 s    301-sample window centred on the output sample                 301  (151 at 0.3 s)
LPF 150 Hz      causal Butterworth (sosfilt) state                             4 (2 biquads x 2)
SWT level j     A_{j-1}[n] and A_{j-1}[n + 2^(j-1)]                            2^(j-1)+1
Christov        |D4| over the last 350 ms (F) + 100 ms refine look-ahead       176
QRS removal +   D_j from (output index - 2^(j-1)) up to newest D_j            155 - 2^(j-1)
  ISWT, D1..D6
D7, A7          not modified, only delayed for the ISWT                         91
ISWT level j    reconstructed A_j[n] and A_j[n - 2^(j-1)]                      2^(j-1)+1
pre delay       pre[n] to subtract the artefact                                154
thresholds      max/min of the last K one-second segments per band              2*K per band

Output latency: clean[n] is produced when pre[n + 153] arrives, plus 150 samples of median
look-ahead => 303 samples = 606 ms behind the raw input (0.6 s median; 228 samples with 0.3 s).
"""
import numpy as np
import scipy.signal as ss

from pipeline_config import MEDIAN_WINDOW_S, A7_INCLUDED

SQ2 = np.sqrt(2.0)


class Ring:
    """Fixed-size circular buffer indexed by ABSOLUTE sample index. Raises if an evicted
    (overwritten) sample is requested -> proves the size is sufficient."""

    def __init__(self, size, name, fill=0.0):
        self.size, self.name = size, name
        self.buf = np.full(size, fill, dtype=float)
        self.newest = -1                     # indices < 0 read as `fill` (zero padding at start)

    def push(self, idx, val):
        assert idx == self.newest + 1, f"{self.name}: out-of-order push {idx}"
        self.buf[idx % self.size] = val
        self.newest = idx

    def _check(self, idx):
        if not (self.newest - self.size < idx <= self.newest):
            raise IndexError(f"{self.name} (size {self.size}): sample {idx} not in buffer "
                             f"(newest {self.newest}) -> ring too small")

    def get(self, idx):
        self._check(idx)
        return self.buf[idx % self.size]

    def set(self, idx, val):
        self._check(idx)
        self.buf[idx % self.size] = val

    def window(self, a, b):                  # values for indices a..b-1
        return np.array([self.get(i) for i in range(a, b)])


def ring_sizes(fs=500, levels=7, median_s=MEDIAN_WINDOW_S, qrs_ms=50, refine_ms=100, f_win_ms=350):
    """Analytic minimum ring sizes (see module docstring)."""
    med = int(round(median_s * fs)) | 1
    w = int(qrs_ms * 1e-3 * fs)                              # 25
    r = int(refine_ms * 1e-3 * fs)                           # 50 (refine search length)
    hw = {j: w + 2 ** j for j in range(1, 7)}                # QRS half-widths (QRS_removal.py)
    d4_lag = 2 ** 4 - 1                                      # D4[n] needs pre[n+15]
    fin = {j: d4_lag + (r - 1) + hw[j] for j in range(1, 7)}  # D_j final when pre index = n + fin
    out_lag = max(max(fin.values()), 2 ** levels - 1)        # 153
    sizes = {"median": med, "christov_absD4": int(f_win_ms * 1e-3 * fs) + 1, "pre_delay": out_lag + 1}
    for j in range(1, levels + 1):
        sizes[f"swt_A{j-1}"] = 2 ** (j - 1) + 1
        sizes[f"iswt_A{j}"] = 2 ** (j - 1) + 1 if j < levels else None
    for j in range(1, 7):
        sizes[f"D{j}"] = out_lag + 2 ** (j - 1) - (2 ** j - 1) + 1          # = 155 - 2^(j-1)
    sizes["D7"] = out_lag + 2 ** (levels - 1) - (2 ** levels - 1) + 1       # 91
    sizes["A7"] = sizes["D7"]
    sizes = {k: v for k, v in sizes.items() if v is not None}
    return sizes, out_lag, fin, hw


class StreamingFirstPhase:
    def __init__(self, fs=500, levels=7, A7_included=A7_INCLUDED,
                 oracle_thresholds=None, oracle_running_max=None,
                 K_segments=16, min_segments=3, size_scale=0, sizes_override=None):
        assert levels == 7
        self.fs, self.L, self.A7_included = fs, levels, A7_included
        sizes, self.out_lag, self.fin, self.hw = ring_sizes(fs, levels)
        if sizes_override:
            sizes.update(sizes_override)
        sizes = {k: v + size_scale for k, v in sizes.items()}
        self.sizes = sizes
        self.half_med = ring_sizes(fs, levels)[0]["median"] // 2     # 150 samples look-ahead/behind at 0.6 s

        # --- preprocessing
        self.raw = Ring(sizes["median"], "raw/median")
        self.sos = ss.butter(4, 150 / (0.5 * fs), btype="low", output="sos")
        self.zi = np.zeros((self.sos.shape[0], 2))
        # --- SWT forward delay lines (A0 = preprocessed signal)
        self.A = [Ring(sizes[f"swt_A{j}"], f"swt_A{j}") for j in range(levels)]
        self.pre = Ring(sizes["pre_delay"], "pre_delay")
        # --- coefficient rings (raw -> QRS removed -> thresholded, in place)
        self.D = {j: Ring(sizes[f"D{j}"], f"D{j}") for j in range(1, levels + 1)}
        self.A7 = Ring(sizes["A7"], "A7")
        self.finalized = {j: -1 for j in range(1, levels + 1)}
        self.finalized["A7"] = -1
        self.zero_intervals = {j: [] for j in range(1, 7)}
        # --- ISWT reconstruction rings
        self.Ar = {j: Ring(sizes[f"iswt_A{j}"], f"iswt_A{j}") for j in range(1, levels)}
        # --- Christov
        self.absD4 = Ring(sizes["christov_absD4"], "christov_|D4|")
        self.ch = _ChristovStream(fs, oracle_running_max)
        self.beats = []
        # --- thresholds
        self.oracle_th = oracle_thresholds
        self.K, self.min_seg = K_segments, min_segments
        self.seg = {}            # name -> [current segment index, running max, running min, count]
        self.hist = {}           # name -> list of (max, min) for completed segments (last K)
        self.p = -1
        self.out = []            # (index, clean value, artefact value, pre value)

    # ------------------------------------------------------------------ thresholds
    def _threshold_value(self, name, n, v):
        # update causal segment statistics with this finalized value
        s = n // self.fs
        st = self.seg.get(name)
        if st is None or st[0] != s:
            if st is not None and st[3] >= self.fs // 2:
                h = self.hist.setdefault(name, [])
                h.append((st[1], st[2]))
                if len(h) > self.K:
                    h.pop(0)
            st = [s, -np.inf, np.inf, 0]
            self.seg[name] = st
        st[1], st[2], st[3] = max(st[1], v), min(st[2], v), st[3] + 1

        if name == "A7" and self.A7_included:
            return v
        if self.oracle_th is not None:
            up, lo = self.oracle_th[name]["upper"], self.oracle_th[name]["lower"]
        else:
            h = self.hist.get(name, [])
            if len(h) < self.min_seg:
                return 0.0                       # warm-up: no artefact estimate yet
            mx = np.array([a for a, _ in h]); mn = np.array([b for _, b in h])
            mmx, mmn = np.median(mx), np.median(mn)
            up = mmx + np.median(np.abs(mmx - mx))
            lo = mmn - np.median(np.abs(mmn - mn))
        return 0.0 if lo <= v <= up else v

    # ------------------------------------------------------------------ main step
    def push(self, x):
        t = self.raw.newest + 1
        self.raw.push(t, x)
        n = t - self.half_med
        if n < 0:
            return
        med = np.median(self.raw.window(n - self.half_med, n + self.half_med + 1))
        m = self.raw.get(n) - med
        y, self.zi = ss.sosfilt(self.sos, [m], zi=self.zi)
        self._push_pre(float(y[0]))

    def _push_pre(self, v):
        self.p += 1
        p = self.p
        self.pre.push(p, v)
        self.A[0].push(p, v)
        # SWT cascade: when A_{j-1}[m] arrives, A_j[m-s] and D_j[m-s] become computable
        for j in range(1, self.L + 1):
            s = 2 ** (j - 1)
            m = self.A[j - 1].newest
            nidx = m - s
            if nidx < 0:
                return
            a_n, a_m = self.A[j - 1].get(nidx), self.A[j - 1].get(m)
            d = (a_n - a_m) / SQ2
            a = (a_n + a_m) / SQ2
            # QRS zeroing of samples arriving inside an already confirmed beat window
            if j <= 6 and any(lo <= nidx <= hi for lo, hi in self.zero_intervals[j]):
                d = 0.0
            self.D[j].push(nidx, d)
            if j < self.L:
                self.A[j].push(nidx, a)
            else:
                self.A7.push(nidx, a)
            if j == 4:
                self._christov_step(nidx, abs(d))
        self._finalize_and_reconstruct()

    def _christov_step(self, i, val):
        self.absD4.push(i, val)
        beat = self.ch.step(i, self.absD4)
        if beat is not None:
            self.beats.append(beat)
            for j in range(1, 7):
                hw = self.hw[j]
                lo, hi = beat - hw, beat + hw
                for k in range(max(0, lo), min(hi, self.D[j].newest) + 1):
                    assert k > self.finalized[j], f"D{j}[{k}] already finalized before beat {beat}"
                    self.D[j].set(k, 0.0)
                self.zero_intervals[j].append((lo, hi))
                self.zero_intervals[j] = [iv for iv in self.zero_intervals[j] if iv[1] >= self.D[j].newest]

    def _finalize_and_reconstruct(self):
        p = self.p
        # a D_j sample is final once no unconfirmed beat can still zero it
        for j in range(1, self.L + 1):
            target = (p - self.fin[j]) if j <= 6 else self.D[j].newest
            while self.finalized[j] < min(target, self.D[j].newest):
                k = self.finalized[j] + 1
                self.D[j].set(k, self._threshold_value(f"D{j}", k, self.D[j].get(k)))
                self.finalized[j] = k
        while self.finalized["A7"] < self.A7.newest:
            k = self.finalized["A7"] + 1
            self.A7.set(k, self._threshold_value("A7", k, self.A7.get(k)))
            self.finalized["A7"] = k
        # ISWT of index n once every band is final at n
        n = p - self.out_lag
        if n < 0:
            return
        assert all(self.finalized[j] >= n for j in range(1, self.L + 1)) and self.finalized["A7"] >= n
        get = lambda r, idx: r.get(idx) if idx >= 0 else 0.0
        s = 2 ** (self.L - 1)
        a = 0.5 * ((get(self.A7, n) + get(self.A7, n - s)) / SQ2 + (get(self.D[self.L], n) - get(self.D[self.L], n - s)) / SQ2)
        for j in range(self.L - 1, 0, -1):
            self.Ar[j].push(n, a)
            s = 2 ** (j - 1)
            a = 0.5 * ((self.Ar[j].get(n) + get(self.Ar[j], n - s)) / SQ2 + (get(self.D[j], n) - get(self.D[j], n - s)) / SQ2)
        self.out.append((n, self.pre.get(n) - a, a, self.pre.get(n)))


class _ChristovStream:
    """Same rules as christov_r_peak_detector.py, but sample-by-sample with only a ring of |D4|."""

    def __init__(self, fs, oracle_running_max=None):
        self.fs = fs
        self.w50, self.w200, self.w350, self.w1200 = int(.05 * fs), int(.2 * fs), int(.35 * fs), int(1.2 * fs)
        self.F_div = 2.0 * (self.w350 - self.w50)
        self.decay = np.linspace(1.0, 0.6, self.w1200 - self.w200)
        self.search = int(0.1 * fs)
        self.oracle = oracle_running_max
        self.running_max = -np.inf
        self.M, self.F, self.R = 0.0, 0.0, 0.0
        self.Mbuf, self.RRbuf, self.exp_rr = [], [], 0
        self.pending_M = 0.0
        self.crossings = []
        self.pending = None
        # [FIX] detector reset: if no beat is found for 2 s after the warm-up, restart the detector from the
        #       current noise/beat level (one large start-up artefact otherwise freezes M and F for good)
        self.recover = int(2.0 * fs)
        self.reset_at = 5 * fs          # time of warm-up end or of the last reset
        self.f_start = 0                # F integrates only over windows that start after this sample
        self.sec_max, self._cur_max = [], 0.0

    def _reset(self, i):
        lvl = float(np.median(self.sec_max[-3:])) if self.sec_max else self.M
        self.M = 0.6 * lvl
        self.Mbuf = [self.M] * 5
        self.pending_M = 0.0
        self.F, self.R = 0.0, 0.0
        self.f_start = i
        self.crossings, self.RRbuf, self.exp_rr = [], [], 0
        self.reset_at = i

    def _pushM(self, v):
        self.Mbuf.append(v)
        if len(self.Mbuf) > 5:
            self.Mbuf.pop(0)

    def step(self, i, X):
        x = X.get(i)
        # per-second maxima of |D4| (used only by a reset)
        self._cur_max = max(self._cur_max, x)
        if (i + 1) % self.fs == 0:
            self.sec_max = (self.sec_max + [self._cur_max])[-5:]; self._cur_max = 0.0
        if i >= 5 * self.fs:
            ref = max(self.crossings[-1] if self.crossings else 0, self.reset_at)
            if i - ref >= self.recover:
                self._reset(i)
        last = self.crossings[-1] if self.crossings else None
        ssl = (i - last) if last is not None else None
        warm = i < 5 * self.fs
        # ---- M
        if warm:
            if self.oracle is not None:
                self.running_max = self.oracle
            else:
                self.running_max = max(self.running_max, x)   # causal: max so far
            self.M = 0.6 * self.running_max
            self._pushM(self.M)
        elif last is not None and ssl < self.w200:
            cand = 0.6 * max(X.get(k) for k in range(last, i + 1))
            if cand > 1.5 * self.Mbuf[-1]:
                cand = 1.1 * self.Mbuf[-1]
            self.pending_M = cand
        elif last is not None and ssl == self.w200:
            self._pushM(self.pending_M if self.pending_M != 0 else self.Mbuf[-1])
            self.M = np.mean(self.Mbuf)
        elif last is not None and self.w200 < ssl < self.w1200:
            self.M = np.mean(self.Mbuf) * self.decay[ssl - self.w200]
        elif last is not None and ssl >= self.w1200:
            self.M = 0.6 * np.mean(self.Mbuf)
        # ---- F
        if i > self.f_start + self.w350:
            self.F += (max(X.get(k) for k in range(i - self.w50, i)) -
                       max(X.get(k) for k in range(i - self.w350, i - self.w350 + self.w50))) / self.F_div
        # ---- R
        if last is not None:
            lo, up = int(2 / 3 * self.exp_rr), self.exp_rr
            if ssl < lo:
                self.R = 0.0
            elif lo <= ssl <= up:
                self.R = (self.M - np.mean(self.Mbuf)) / 1.4
        MFR = self.M + self.F + self.R
        # ---- detection (in causal mode no beats are accepted during the 5 s warm-up)
        allowed = (self.oracle is not None) or not warm
        if allowed and (last is None or ssl > self.w200) and x > MFR:
            self.crossings.append(i)
            if len(self.crossings) > 1:
                self.RRbuf.append(self.crossings[-1] - self.crossings[-2])
                if len(self.RRbuf) > 5:
                    self.RRbuf.pop(0)
                self.exp_rr = int(np.mean(self.RRbuf))
            self.pending = i
        # ---- refine: R peak = argmax |D4| in [crossing, crossing + 100 ms)
        if self.pending is not None and i == self.pending + self.search - 1:
            c = self.pending
            self.pending = None
            seg = [X.get(k) for k in range(c, c + self.search)]
            return c + int(np.argmax(seg))
        return None


# ============================================================================ plotting
def plot_streaming(raw, stream, batch_clean=None, beats=None, fs=500, start=0.0, dur=None, save=False, show=True):
    """Raw input, streaming clean output (placed at its true sample index) and, optionally, the batch output."""
    from ecg_io import setup_style, plot_window, finish, C_BLUE, C_ORANGE, C_GREY, C_INK
    plt = setup_style()
    N = len(raw)
    idx = stream[:, 0].astype(int)
    clean = np.full(N, np.nan); clean[idx] = stream[:, 1]
    sl, t = plot_window(N, fs, start, dur if dur is not None else N / fs)
    fig, ax = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    ax[0].plot(t, raw[sl], color=C_GREY); ax[0].set_title("Raw input (arrives one sample at a time)")
    if batch_clean is not None:
        ax[1].plot(t, batch_clean[sl], color=C_GREY, lw=2.2, label="batch pipeline (whole record)")
    ax[1].plot(t, clean[sl], color=C_BLUE, label="streaming output (fixed-size buffers, causal)")
    if beats:
        b = np.array([k for k in beats if sl.start <= k < sl.stop and k < N], dtype=int)
        ax[1].plot(b / fs, np.nan_to_num(clean[b]), "v", color=C_INK, ms=6, label="beats detected while streaming")
    ax[1].axvspan(sl.start / fs, min(5.0, sl.stop / fs), color=C_ORANGE, alpha=0.08, lw=0)
    ax[1].text(0.05 + sl.start / fs, ax[1].get_ylim()[1] * 0.85 if batch_clean is not None else 0, "warm-up", fontsize=8.5)
    ax[1].set_title(f"Clean ECG -- streaming output is {ring_sizes()[1] + ring_sizes()[0]['median'] // 2 + 1} samples behind the input")
    ax[1].set_ylim(top=1.35 * np.nanmax(clean[sl])); ax[1].legend(loc="upper right", ncol=3); ax[1].set_xlabel("time (s)")
    return finish(fig, "streaming_vs_batch", save, show)


if __name__ == "__main__":
    from ecg_io import parse_args
    from first_phase_pipeline import first_phase

    args = parse_args("Streaming (sample-by-sample) version of the First Phase")
    s = StreamingFirstPhase(args.fs)                     # fully causal: running thresholds, 5 s warm-up
    for v in args.ecg:
        s.push(v)
    out = np.array(s.out)
    print(f"streamed {len(args.ecg)} samples -> {len(out)} clean samples, {len(s.beats)} beats; "
          f"ring sizes: {s.sizes}")
    batch = first_phase(args.ecg, args.fs)["clean"]
    dur = args.dur if "--dur" in __import__("sys").argv else None
    plot_streaming(args.ecg, out, batch, s.beats, args.fs, args.start, dur, save=args.save)
