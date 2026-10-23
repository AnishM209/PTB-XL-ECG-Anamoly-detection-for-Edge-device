"""
live_demo_mn.py: TEST VERSION of live_demo.py with the notch moved inside the team pipeline:
    raw (250 Hz) -> x2 to 500 Hz -> median filter (0.6 s) -> notch 50 Hz Q5 + LP 40 Hz -> LPF 150 Hz -> SWT, Christov,
    QRS removal, thresholding, ISWT -> models            (default --pipeline team_mn)
Thresholds: no fold-9 recalibration exists for this order yet, so the notch-before-median values
(config_team_TC_q5.json) are used as an approximation. Use only for testing.
Everything else is as in live_demo.py:

Shows, updated live:
  * raw ECG as received from the board
  * clean ECG after DSP filtering (50 Hz notch + 0.5 to 40 Hz, causal) with detected beats
  * verdict of both models (Normal / Abnormal, AF / No AF) from the last 10 s
  * heart rate with a beating heart, and a signal-quality indicator

Model path = the validated device chain D3: causal notch 50 Hz Q5 + Butterworth LP 40 Hz (order 2) at 250 Hz
-> x2 to 500 Hz (resample_poly) -> 10 s buffer -> each model window minus its own mean -> (x - mu) / sd ->
binary (2.5 s windows) and AF (5 s windows), 50 % overlap, sigmoid mean, thresholds from config_device_q5.json.
If the signal quality is low (mains, movement, drift, no beats), no verdict is shown.
Electrode check: in the training orientation (+ at V2, - at V1) the T wave of V2-V1 is upright; a clearly
inverted T wave on every beat shows a note that the leads may be swapped. Press I to flip the polarity live.

Usage (PowerShell, in the folder with models.py, config.json and the two .pt files, e.g. ...\\AgriRAG\\export):
    pip install pyserial matplotlib numpy scipy torch
    python live_demo.py --replay still_01.npz              # rehearsal / fallback: plays a recording as if live
    python live_demo.py COM37 --scale 1000                 # live; --scale = stream units per mV (1000 = uV)
Options: --invert (swap polarity), --fs 250, --config config_device_q5.json, --save demo_rec.npz, --speed 1
The stream must be the UNFILTERED signal, one number per line, 250 samples/s.
"""
import argparse
import json
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, iirnotch, lfilter, lfilter_zi, resample_poly

NUM = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
DEFAULT_THR = {"binary": 0.5750, "af": 0.3286}          # config_device_q5.json (PTB-XL fold 9, device chain)
DEFAULT_THR_TEAM = {"binary": 0.6857, "af": 0.3084}     # config_team_TC_q5.json (notch + LP40 -> team pipeline)
TEAM_PAD = 512                                          # mirror padding, as clean_dataset.py / team_on_device.py
# signal-quality limits (still recordings: mains 0.3 to 1 mV, 0.5-5 Hz 0.14 mV rms, 5-40 Hz 0.10 mV rms)
NO_DATA_S = 1.5                                         # no samples for this long: show "No signal"
Q_MAINS_MV, Q_SLOW_MV, Q_FAST_MV, Q_DRIFT_MV = 3.0, 0.40, 0.30, 0.30   # drift: < 0.5 Hz rms over 10 s

BG, INK, MUTED, RULE = "#ffffff", "#1f2328", "#6b7280", "#e5e7eb"
C_RAW, C_CLEAN, C_OK, C_BAD, C_WARN = "#9ca3af", "#1f4e79", "#2f6f4f", "#a33a3a", "#9a6700"


# ----------------------------------------------------------------------------- filters
class Causal:
    """Cascade of causal IIR sections with state kept between chunks."""

    def __init__(self, sections):
        self.sec = sections
        self.z = None

    def __call__(self, x):
        x = np.asarray(x, float)
        if len(x) == 0:
            return x
        if self.z is None:
            self.z = [lfilter_zi(b, a) * x[0] for b, a in self.sec]
        for k, (b, a) in enumerate(self.sec):
            x, self.z[k] = lfilter(b, a, x, zi=self.z[k])
        return x


def mains_amp(x, fs):
    x = np.asarray(x, float) - np.mean(x)
    n = np.arange(len(x))
    return 2 * abs(np.sum(x * np.exp(-2j * np.pi * 50 * n / fs))) / max(1, len(x))


def band_rms(x, fs, lo, hi):
    b, a = butter(2, [lo, hi], btype="band", fs=fs)
    return float(np.std(filtfilt(b, a, x)))


def detect_beats(y, fs):
    """R/S peaks on the clean display signal (dominant polarity found automatically)."""
    if len(y) < 2 * fs:
        return np.array([], int)
    sgn = 1 if np.percentile(y, 99.5) >= -np.percentile(y, 0.5) else -1
    s = sgn * y
    cand, _ = find_peaks(s, distance=int(0.33 * fs), prominence=0.1)
    if len(cand) < 3:
        return np.array([], int)
    ref = np.median(np.sort(s[cand])[-8:])
    return cand[s[cand] > 0.45 * ref]


def t_wave_sign(y, beats, fs):
    """Median signed T-wave amplitude (mV) relative to the PR level, and median |QRS| (mV)."""
    s = lambda ms: int(ms * 1e-3 * fs)  # noqa: E731
    T, Q = [], []
    for k, r in enumerate(beats):
        if r - s(100) < 0 or r + s(400) >= len(y):
            continue
        iso = np.median(y[r - s(100): r - s(60)])
        end = r + s(400) if k + 1 >= len(beats) else min(r + s(400), beats[k + 1] - s(100))
        if end - (r + s(150)) < s(50):
            continue
        seg = y[r + s(150): end] - iso
        T.append(seg[np.argmax(np.abs(seg))]); Q.append(np.max(np.abs(y[r - s(50): r + s(50)] - iso)))
    if len(T) < 4:
        return np.nan, np.nan
    return float(np.median(T)), float(np.median(Q))


# ----------------------------------------------------------------------------- models
class Predictor:
    def __init__(self, folder, config_name, defaults=DEFAULT_THR):
        import torch
        self.torch = torch
        folder = Path(folder)
        sys.path.insert(0, str(folder))
        from models import build_model
        cfg = json.loads((folder / "config.json").read_text())
        thr = dict(defaults)
        cpath = folder / config_name
        if cpath.exists():
            dc = json.loads(cpath.read_text())
            thr = {t: dc[t]["threshold_balanced"] for t in ("binary", "af")}
            self.thr_source = cpath.name
        else:
            self.thr_source = f"built-in values ({config_name} not found)"
        self.thr = thr
        self.m, self.c = {}, {}
        for task in ("binary", "af"):
            c = cfg[task]
            m = build_model(c["model"], 1, 1)
            m.load_state_dict(torch.load(folder / c["weights"], map_location="cpu"))
            self.m[task], self.c[task] = m.eval(), c
        torch.set_num_threads(2)

    def __call__(self, x500, demean=True):
        """x500: 10 s model input at 500 Hz (mV). Returns {task: probability}."""
        out = {}
        with self.torch.no_grad():
            for task, m in self.m.items():
                c = self.c[task]
                win = int(round(c["win_sec"] * c["fs"]))
                T = len(x500)
                starts = list(range(0, T - win + 1, max(1, win // 2)))
                if starts[-1] != T - win:
                    starts.append(T - win)
                W = np.stack([x500[s:s + win] - (x500[s:s + win].mean() if demean else 0.0) for s in starts])
                W = ((W - c["mu"][0]) / c["sd"][0]).astype(np.float32)[:, None, :]
                p = self.torch.sigmoid(m(self.torch.from_numpy(W))).numpy().ravel()
                out[task] = float(p.mean())
        return out


class TeamPipeline:
    """The team's First Phase cleaning (bank_pipeline.bank_first_phase: 0.6 s median, LPF 150 Hz, SWT, Christov,
    QRS removal, thresholding, ISWT) on a 10 s, 500 Hz window, mirror-padded like clean_dataset.py."""

    def __init__(self, team_dir, notch_after_median=False):
        team_dir = Path(team_dir).resolve()
        if not (team_dir / "bank_pipeline.py").exists():
            raise FileNotFoundError(f"bank_pipeline.py not found in {team_dir}")
        sys.path.insert(0, str(team_dir))
        import bank_pipeline
        if notch_after_median:
            # insert notch Q5 + LP40 right after the team's median filter (their code is not edited):
            # bank_first_phase computes pre = lpf_150hz(median_filter(x)), so wrapping median_filter gives
            # median -> notch + LP40 -> LPF 150 Hz -> SWT ...
            orig = bank_pipeline.median_filter

            def median_then_notch(x, fs, *a, **k):
                y = np.asarray(orig(x, fs, *a, **k), float)
                for b, c in (iirnotch(50.0, 5.0, fs), butter(2, 40.0, btype="low", fs=fs)):
                    y = lfilter(b, c, y, zi=lfilter_zi(b, c) * y[0])[0]
                return y
            bank_pipeline.median_filter = median_then_notch
        self.f = bank_pipeline.bank_first_phase

    def __call__(self, x500):
        n = len(x500)
        ext = np.r_[x500[TEAM_PAD:0:-1], x500, x500[-2:-TEAM_PAD - 2:-1]]
        out = self.f(ext, 500, warmup=False)
        clean = np.nan_to_num(np.asarray(out["clean"][TEAM_PAD:TEAM_PAD + n], float))
        pk = np.asarray(out["r_peaks"], int) - TEAM_PAD
        return clean, pk[(pk >= 0) & (pk < n)]


# ----------------------------------------------------------------------------- sources
class Source:
    def __init__(self):
        self.lock = threading.Lock()
        self.new, self.all = [], []
        self.error, self.running, self.bad = None, True, 0

    def push(self, vals):
        with self.lock:
            self.new.extend(vals); self.all.extend(vals)

    def take(self):
        with self.lock:
            out, self.new = self.new, []
        return out


def serial_reader(src, port, baud, scale):
    try:
        import serial
        ser = serial.Serial(port, baud, timeout=0.5)
    except ImportError:
        src.error = "pyserial missing: pip install pyserial"; return
    except Exception as e:
        src.error = f"cannot open {port}: {e} (close Arduino IDE / miniterm / other plots)"; return
    ser.reset_input_buffer()
    buf = b""
    while src.running:
        chunk = ser.read(ser.in_waiting or 1)
        if not chunk:
            continue
        buf += chunk
        *lines, buf = buf.split(b"\n")
        vals = []
        for ln in lines:
            m = NUM.findall(ln.decode("ascii", "ignore"))
            if m:
                vals.append(float(m[0]) / scale)
            else:
                src.bad += 1
        if vals:
            src.push(vals)
    ser.close()


def replay_reader(src, path, fs, speed):
    z = np.load(path)
    x = z["code"] * (float(z["vref"]) / (int(z["gain"]) * (2 ** 23 - 1))) * 1e3     # mV
    k, chunk, t0 = 0, max(1, int(fs / 50)), time.time()
    while src.running:
        src.push(list(x[k:k + chunk]))
        k += chunk
        if k >= len(x):
            k = 0                                                                    # loop
        target = t0 + (k if k else len(x)) / (fs * speed)
        time.sleep(max(0.0, target - time.time()))
        if k == 0:
            t0 = time.time()


# ----------------------------------------------------------------------------- app
def main():
    ap = argparse.ArgumentParser(description="Live ECG analysis demo")
    ap.add_argument("port", nargs="?", default="COM37")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--fs", type=float, default=250.0)
    ap.add_argument("--scale", type=float, default=1000.0, help="stream units per mV (1000 = uV)")
    ap.add_argument("--replay", default=None, help="play a recording (.npz from the PYNQ script) instead of the port")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--invert", action="store_true", help="start with the polarity flipped (also: press I)")
    ap.add_argument("--models", default=".", help="folder with models.py, config.json and the .pt files")
    ap.add_argument("--config", default=None, help="thresholds file (default: matches --pipeline)")
    ap.add_argument("--pipeline", choices=["d3", "team", "team_mn"], default="team_mn",
                    help="model input: d3 = notch + LP40; team = notch + LP40 -> team pipeline; "
                         "team_mn = team pipeline with notch + LP40 after its median filter (this test)")
    ap.add_argument("--team-dir", default="test_cleaning_reset", help="folder with the team's pipeline .py files")
    ap.add_argument("--seconds", type=float, default=8.0, help="visible window")
    ap.add_argument("--every", type=float, default=2.0, help="seconds between model runs")
    ap.add_argument("--save", default=None, help="save everything received (mV) to this .npz on exit")
    ap.add_argument("--no-model", action="store_true", help="show signals and heart rate only")
    args = ap.parse_args()

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": MUTED,
                                "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": RULE,
                                "toolbar": "None"})

    fs = args.fs
    pred, team, model_msg = None, None, "Models off"
    if not args.no_model:
        try:
            team_mode = args.pipeline in ("team", "team_mn")
            cfg_name = args.config or ("config_team_TC_q5.json" if team_mode else "config_device_q5.json")
            team = TeamPipeline(args.team_dir, args.pipeline == "team_mn") if team_mode else None
            pred = Predictor(args.models, cfg_name, DEFAULT_THR_TEAM if team_mode else DEFAULT_THR)
            model_msg = (f"Model input: {dict(d3='notch + 40 Hz low-pass', team='notch + LP40 -> team DSP pipeline', team_mn='team DSP pipeline, notch after median (test)')[args.pipeline]}"
                         f"    |    Thresholds: {pred.thr_source}")
        except Exception as e:
            model_msg = f"Models not loaded: {type(e).__name__}: {e}"
    thr = pred.thr if pred else DEFAULT_THR

    model_f = Causal([iirnotch(50.0, 5.0, fs), butter(2, 40.0, btype="low", fs=fs)])          # model path
    disp_f = Causal([iirnotch(50.0, 5.0, fs), butter(2, [0.5, 40.0], btype="band", fs=fs)])   # display path
    n_vis, n_model = int(args.seconds * fs), int(10 * fs)
    nbuf = max(n_vis, n_model)
    raw, clean, mpath = deque(maxlen=nbuf), deque(maxlen=nbuf), deque(maxlen=n_model)
    total = [0]
    st = dict(hist=deque(maxlen=3), last_run=0.0, hr=np.nan, beat_t=0.0, last_beat=-1,
              sign=-1.0 if args.invert else 1.0, flipped_at=-1e9)

    src = Source()
    if args.replay:
        threading.Thread(target=replay_reader, args=(src, args.replay, fs, args.speed), daemon=True).start()
        mode = f"Replay: {Path(args.replay).name}"
    else:
        threading.Thread(target=serial_reader, args=(src, args.port, args.baud, args.scale), daemon=True).start()
        mode = f"Live: {args.port}"

    # ---- layout: white page, two signal panels on the left, a plain read-out column on the right
    fig = plt.figure(figsize=(15, 8.2), facecolor=BG)
    try:
        fig.canvas.manager.set_window_title("ECG analysis")
    except Exception:
        pass
    gs = fig.add_gridspec(3, 2, width_ratios=[3.3, 1.15], height_ratios=[0.32, 1, 1],
                          left=0.05, right=0.97, top=0.95, bottom=0.10, hspace=0.38, wspace=0.10)
    head = fig.add_subplot(gs[0, :]); head.axis("off")
    head.text(0.0, 0.70, "Real-time ECG analysis", fontsize=18, color=INK, va="center")
    head.text(0.0, 0.12, "Single-lead chest ECG, PYNQ-Z2 with ADS1299, on-device signal path and models",
              fontsize=10.5, color=MUTED, va="center")
    head.text(1.0, 0.70, mode, fontsize=10.5, color=MUTED, ha="right", va="center")
    clock_t = head.text(1.0, 0.12, "", fontsize=10, color=MUTED, ha="right", va="center")
    head.axhline(-0.25, color=RULE, lw=1); head.set_ylim(-0.3, 1.0)

    ax_raw = fig.add_subplot(gs[1, 0]); ax_cl = fig.add_subplot(gs[2, 0], sharex=ax_raw)
    for a, title in ((ax_raw, "Raw signal"), (ax_cl, "Clean signal after DSP filtering")):
        a.set_facecolor(BG); a.grid(color="#f1f2f4", lw=0.8)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.set_title(title, loc="left", fontsize=12, color=INK, pad=6)
        a.set_ylabel("mV")
    ax_cl.set_xlabel("time (s)")
    l_raw, = ax_raw.plot([], [], color=C_RAW, lw=0.8)
    l_cl, = ax_cl.plot([], [], color=C_CLEAN, lw=1.2)
    l_pk, = ax_cl.plot([], [], "o", ms=4, mfc="none", mec=C_BAD, mew=1.2)

    side = fig.add_subplot(gs[1:, 1]); side.axis("off"); side.set_xlim(0, 1); side.set_ylim(0, 1)
    rows = {}
    y = 0.98
    for key, label in (("hr", "Heart rate"), ("morph", "Morphology"), ("af", "Atrial fibrillation"),
                       ("q", "Signal quality")):
        side.text(0.04, y, label, fontsize=10.5, color=MUTED, va="top")
        rows[key] = side.text(0.13 if key == "hr" else 0.04, y - 0.055, "", fontsize=22 if key != "q" else 15, color=INK, va="top")
        rows[key + "_sub"] = side.text(0.04, y - 0.135 if key != "q" else y - 0.115, "", fontsize=9.5,
                                       color=MUTED, va="top")
        side.axhline(y - 0.20, xmin=0.04, xmax=0.98, color=RULE, lw=1)
        y -= 0.225
    heart = side.text(0.04, 0.98 - 0.062, "\u2665", fontsize=19, color=C_BAD, va="top", alpha=0.25)
    note_t = side.text(0.04, 0.085, "", fontsize=9.5, color=C_WARN, va="top", linespacing=1.4)
    foot = fig.text(0.05, 0.015, model_msg + "    |    Press I to flip polarity", fontsize=8.5, color=MUTED)
    fig.text(0.97, 0.015, "Demonstration system, not a medical device", fontsize=8.5, color=MUTED, ha="right")
    t_start = time.time()
    st["last_rx"] = t_start
    idle_txt = [a.text(0.5, 0.5, "", transform=a.transAxes, ha="center", va="center", fontsize=12, color=MUTED,
                       bbox=dict(fc=BG, ec="none", alpha=0.85)) for a in (ax_raw, ax_cl)]

    def show_idle(msg, hint):
        """No samples arriving: flat line in both panels and a plain message instead of results."""
        if total[0] == 0:
            ax_raw.set_xlim(0, args.seconds)
        x0, x1 = ax_raw.get_xlim()
        for a, ln, tx, m in ((ax_raw, l_raw, idle_txt[0], msg), (ax_cl, l_cl, idle_txt[1], hint)):
            ln.set_data([x0, x1], [0.0, 0.0]); a.set_ylim(-1, 1); tx.set_text(m)
        l_pk.set_data([], [])
        rows["hr"].set_text("--"); rows["hr"].set_color(MUTED); heart.set_alpha(0.2)
        for key in ("morph", "af"):
            rows[key].set_text("No result"); rows[key].set_color(MUTED); rows[key + "_sub"].set_text("")
        rows["q"].set_text("No signal"); rows["q"].set_color(C_WARN); rows["q_sub"].set_text(hint)
        note_t.set_text("")

    def flip(_ev=None):
        st["sign"] *= -1
        for d in (raw, clean, mpath):
            v = -np.asarray(d); d.clear(); d.extend(v)
        for f in (model_f, disp_f):
            if f.z is not None:
                f.z = [-z for z in f.z]
        st["hist"].clear(); st["last_run"] = 0.0; st["flipped_at"] = time.time()

    fig.canvas.mpl_connect("key_press_event", lambda e: flip() if e.key in ("i", "I") else None)

    def assess(r, yc):
        n5 = int(5 * fs)
        if len(r) < n5:
            return "waiting", ""
        r = np.asarray(r); r5 = r[-n5:] - r[-n5:].mean(); y5 = np.asarray(yc)[-n5:]
        mains = mains_amp(r5[-int(fs):], fs)
        slow, fast = band_rms(r5, fs, 0.5, 5.0), band_rms(r5, fs, 5.0, 40.0)
        r10 = r[-int(10 * fs):]
        drift = float(np.std(filtfilt(*butter(2, 0.5, btype="low", fs=fs), r10 - r10.mean()))) if len(r10) > 3 * fs else 0.0
        if len(detect_beats(y5, fs)) < 2:
            return "bad", "No heartbeat found. Check the electrodes."
        if mains > Q_MAINS_MV:
            return "bad", f"Mains interference ({mains:.1f} mV)."
        if drift > Q_DRIFT_MV:
            return "bad", "Baseline drift. Check electrode contact."
        if slow > Q_SLOW_MV or fast > Q_FAST_MV:
            return "bad", "Movement artefact."
        return "good", f"50 Hz in raw signal {mains:.2f} mV"

    def update(_):
        if src.error:
            hint = "Install it: pip install pyserial" if "pyserial" in src.error else \
                "Close other programs using the port\n(Arduino IDE, miniterm) and restart."
            show_idle("No signal: the serial port could not be opened", hint)
            foot.set_text(src.error); foot.set_color(C_BAD)
            return
        new = np.asarray(src.take(), float) * st["sign"]
        now = time.time()
        if len(new):
            st["last_rx"] = now
            for tx in idle_txt:
                tx.set_text("")
        elif now - st["last_rx"] > NO_DATA_S:
            gap = now - st["last_rx"]
            show_idle(f"No signal from the board for {gap:.0f} s",
                      "Start the streaming cell on the PYNQ\nand check the USB cable.")
            clock_t.set_text(f"{total[0] / fs:.0f} s received, waiting for data")
            return
        if len(new):
            raw.extend(new); clean.extend(disp_f(new)); mpath.extend(model_f(new)); total[0] += len(new)
        el = time.time() - t_start
        clock_t.set_text(f"{total[0] / fs:.0f} s received, {total[0] / max(el, 1e-6):.0f} samples/s")
        n = len(raw)
        if n < 10:
            return
        k = min(n, n_vis)
        r = np.asarray(raw)[-k:]; yc = np.asarray(clean)[-k:]
        t = (total[0] - k + np.arange(k)) / fs
        l_raw.set_data(t, r - np.median(r)); l_cl.set_data(t, yc)
        ax_raw.set_xlim(max(0, t[-1] - args.seconds), max(args.seconds, t[-1]))
        for a, s in ((ax_raw, r - np.median(r)), (ax_cl, yc)):
            lo, hi = np.percentile(s, [0.5, 99.5]); pad = 0.2 * (hi - lo) + 0.05
            a.set_ylim(lo - pad, hi + pad)

        beats = detect_beats(yc, fs)
        l_pk.set_data(t[beats], yc[beats]) if len(beats) else l_pk.set_data([], [])
        if len(beats) >= 3:
            rr = np.diff(beats[-9:]) / fs; rr = rr[(rr > 0.3) & (rr < 2.0)]
            st["hr"] = 60.0 / np.median(rr) if len(rr) else np.nan
            last_abs = total[0] - k + beats[-1]
            if last_abs != st["last_beat"]:
                st["last_beat"] = last_abs; st["beat_t"] = time.time()

        q, reason = assess(raw, clean)
        ok = q == "good"
        if team is not None and not np.isnan(st.get("hr_team", np.nan)):
            st["hr"] = st["hr_team"]                        # team mode: heart rate from the team's Christov detector
        hr_ok = ok and not np.isnan(st["hr"])
        rows["hr"].set_text(f"{st['hr']:.0f} bpm" if hr_ok else "--"); rows["hr"].set_color(INK if hr_ok else MUTED)
        heart.set_alpha(0.9 if hr_ok and time.time() - st["beat_t"] < 0.25 else 0.2)
        rows["q"].set_text({"waiting": "Collecting signal", "good": "Good", "bad": "Low"}[q])
        rows["q"].set_color({"waiting": MUTED, "good": C_OK, "bad": C_WARN}[q])
        rows["q_sub"].set_text(reason)

        # electrode orientation hint: T wave inverted on most beats while the signal is otherwise good
        note = ""
        tw, qa = t_wave_sign(np.asarray(clean)[-n_model:], detect_beats(np.asarray(clean)[-n_model:], fs), fs) \
            if len(clean) >= n_model else (np.nan, np.nan)
        if ok and not np.isnan(tw) and tw < -0.08 and abs(tw) > 0.12 * qa and time.time() - st["flipped_at"] > 12:
            note = ("Note: the T wave is inverted on most beats.\nThe electrodes may be swapped\n"
                    "(+ should be at V2, - at V1).\nSwap the leads or press I.")
        if st["sign"] < 0:
            note = (note + "\n" if note else "") + "Polarity flipped in software\n(press I to undo)."
        note_t.set_text(note)

        if len(mpath) < n_model:
            rows["morph"].set_text("Analysing"); rows["morph"].set_color(MUTED)
            rows["morph_sub"].set_text(f"first result in {max(0, 10 - len(mpath) / fs):.0f} s")
            rows["af"].set_text(""); rows["af_sub"].set_text("")
            return
        if pred is not None and time.time() - st["last_run"] >= args.every and not st.get("busy"):
            st["last_run"] = time.time(); st["busy"] = True
            src_buf = np.asarray(raw, float)[-n_model:] if args.pipeline == "team_mn" else np.asarray(mpath, float)
            x500 = resample_poly(src_buf - src_buf.mean(), 2, 1)      # mean removed first: no resampling edge jump

            def run(x500=x500):
                try:
                    if team is not None:
                        xc, pk = team(x500)
                        rr = np.diff(pk) / 500.0; rr = rr[(rr > 0.3) & (rr < 2.0)]
                        st["hr_team"] = 60.0 / np.median(rr) if len(rr) >= 2 else np.nan
                        st["hist"].append(pred(xc, demean=False))
                    else:
                        st["hist"].append(pred(x500))
                except Exception as e:                       # keep the screen running, show the reason
                    st["run_error"] = f"{type(e).__name__}: {e}"
                finally:
                    st["busy"] = False
            threading.Thread(target=run, daemon=True).start()
        if st.get("run_error"):
            foot.set_text("Model run failed: " + st["run_error"]); foot.set_color(C_BAD)
        if pred is None or not st["hist"]:
            rows["morph"].set_text("Model off" if pred is None else "Analysing"); rows["morph"].set_color(MUTED)
            return
        pb = float(np.mean([h["binary"] for h in st["hist"]]))       # smoothed over the last 3 runs (~6 s)
        pa = float(np.mean([h["af"] for h in st["hist"]]))
        if not ok:
            for key in ("morph", "af"):
                rows[key].set_text("No result"); rows[key].set_color(MUTED)
                rows[key + "_sub"].set_text("waiting for a clean signal")
            return
        ab, af = pb >= thr["binary"], pa >= thr["af"]
        rows["morph"].set_text("Abnormal" if ab else "Normal"); rows["morph"].set_color(C_BAD if ab else C_OK)
        rows["morph_sub"].set_text(f"model probability {pb:.0%} (threshold {thr['binary']:.0%})")
        rows["af"].set_text(f"Suspected ({pa:.0%})" if af else "Not detected"); rows["af"].set_color(C_BAD if af else C_OK)
        rows["af_sub"].set_text(f"model probability {pa:.0%} (threshold {thr['af']:.0%})")

    anim = FuncAnimation(fig, update, interval=60, cache_frame_data=False)  # noqa: F841
    try:
        plt.show()
    finally:
        src.running = False
        if args.save and src.all:
            np.savez(args.save, mv=np.asarray(src.all), fs=fs)
            print(f"saved {len(src.all)} samples to {args.save}")


if __name__ == "__main__":
    main()
