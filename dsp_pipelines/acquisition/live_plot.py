"""
live_plot.py: live plot of the ECG stream the PYNQ sends over UART (one number per line).

Top panel   : the stream as received (mean removed), with the 50 Hz mains amplitude of the last second.
Bottom panel: the model-path filter applied on the laptop: 50 Hz notch (Q5) + 40 Hz low-pass (causal),
              which is what the models get (window mean removed for display).

Usage (PowerShell, close the Arduino IDE / miniterm first, only one program can open COM37):
    pip install pyserial matplotlib numpy scipy
    python live_plot.py COM37
    python live_plot.py COM37 --scale 1000        # if the stream is in uV (1000 units per mV), plot in mV
    python live_plot.py COM37 --save still_02.npz # also save everything received when the window is closed
    python live_plot.py --demo                    # synthetic ECG + 17 mV mains, to test without the board

Options: --baud 115200, --fs 250, --q 5, --lp 40, --seconds 8, --column 0 (if a line has several numbers)
"""
import argparse
import re
import threading
import time
from collections import deque

import numpy as np
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi

NUM = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def parse_line(line, column=0):
    """Return one float from a text line ('123', '12,34', 'ecg:123', '1 2 3'), or None."""
    nums = NUM.findall(line)
    if len(nums) <= column:
        return None
    return float(nums[column])


class CausalChain:
    """Notch (mains) + Butterworth low-pass, causal, state kept between chunks (as on the device)."""

    def __init__(self, fs, q=5.0, lp=40.0, mains=50.0, lp_order=4):
        self.bn, self.an = iirnotch(mains, q, fs=fs)
        self.bl, self.al = butter(lp_order, lp, btype="low", fs=fs)
        self.zn = self.zl = None

    def __call__(self, x):
        x = np.asarray(x, float)
        if len(x) == 0:
            return x
        if self.zn is None:                       # start in steady state at the first value (no start-up jump)
            self.zn = lfilter_zi(self.bn, self.an) * x[0]
            self.zl = lfilter_zi(self.bl, self.al) * x[0]
        y, self.zn = lfilter(self.bn, self.an, x, zi=self.zn)
        y, self.zl = lfilter(self.bl, self.al, y, zi=self.zl)
        return y


def mains_amplitude(x, fs, f=50.0):
    """Peak amplitude of the f Hz component of x (same units as x)."""
    x = np.asarray(x, float)
    if len(x) < fs // 2:
        return np.nan
    x = x - x.mean()
    n = np.arange(len(x))
    return 2.0 * abs(np.sum(x * np.exp(-2j * np.pi * f * n / fs))) / len(x)


# ----------------------------------------------------------------------------- data sources
class Source:
    """Background reader: collects values and receive times so the plot never blocks the port."""

    def __init__(self):
        self.lock = threading.Lock()
        self.new = []
        self.all_vals, self.all_times = [], []
        self.bad_lines = 0
        self.error = None
        self.running = True

    def push(self, v):
        t = time.time()
        with self.lock:
            self.new.append(v)
            self.all_vals.append(v)
            self.all_times.append(t)

    def take(self):
        with self.lock:
            out, self.new = self.new, []
        return out


def serial_reader(src, port, baud, column):
    try:
        import serial
    except ImportError:
        src.error = "pyserial missing: run  pip install pyserial"
        return
    try:
        ser = serial.Serial(port, baud, timeout=0.5)
    except Exception as e:                        # port busy or wrong name
        src.error = f"cannot open {port}: {e}  (close Arduino IDE / miniterm / other plots)"
        return
    ser.reset_input_buffer()
    buf = b""
    while src.running:
        chunk = ser.read(ser.in_waiting or 1)
        if not chunk:
            continue
        buf += chunk
        *lines, buf = buf.split(b"\n")
        for raw in lines:
            v = parse_line(raw.decode("ascii", "ignore"), column)
            if v is None:
                src.bad_lines += 1
            else:
                src.push(v)
    ser.close()


def demo_reader(src, fs):
    """Synthetic: 70 bpm ECG-like beats (~1 mV) + 17 mV 50 Hz mains + slow drift, in uV."""
    k, beat = 0, int(fs * 60 / 70)
    t_next = time.time()
    while src.running:
        n = k % beat
        qrs = -800 * np.exp(-0.5 * ((n - 0.10 * fs) / (0.012 * fs)) ** 2)
        tw = 400 * np.exp(-0.5 * ((n - 0.40 * fs) / (0.050 * fs)) ** 2)
        src.push(qrs + tw + 17000 * np.sin(2 * np.pi * 50 * k / fs) + 300 * np.sin(2 * np.pi * 0.2 * k / fs))
        k += 1
        t_next += 1.0 / fs
        time.sleep(max(0.0, t_next - time.time()))


# ----------------------------------------------------------------------------- plot
def main():
    ap = argparse.ArgumentParser(description="Live plot of the PYNQ ECG UART stream")
    ap.add_argument("port", nargs="?", default="COM37")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--fs", type=float, default=250.0, help="sampling rate of the stream (Hz)")
    ap.add_argument("--q", type=float, default=5.0, help="notch quality factor (5 = wide, 30 = narrow)")
    ap.add_argument("--lp", type=float, default=40.0, help="low-pass cut-off (Hz)")
    ap.add_argument("--seconds", type=float, default=8.0, help="length of the visible window")
    ap.add_argument("--scale", type=float, default=None, help="stream units per mV (e.g. 1000 for uV); plot in mV")
    ap.add_argument("--column", type=int, default=0, help="which number to use if a line has several")
    ap.add_argument("--save", default=None, help="save received values + times to this .npz on exit")
    ap.add_argument("--demo", action="store_true", help="synthetic signal instead of the serial port")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fs = args.fs
    scale = args.scale or 1.0
    unit = "mV" if args.scale else "stream units"
    n_win = int(args.seconds * fs)
    raw_buf, filt_buf = deque(maxlen=n_win), deque(maxlen=n_win)
    chain = CausalChain(fs, args.q, args.lp)

    src = Source()
    target = (lambda: demo_reader(src, fs)) if args.demo else \
             (lambda: serial_reader(src, args.port, args.baud, args.column))
    threading.Thread(target=target, daemon=True).start()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    l1, = ax1.plot([], [], lw=0.8, color="tab:gray")
    l2, = ax2.plot([], [], lw=1.0, color="tab:blue")
    ax2.set_xlabel("time (s), last %.0f s" % args.seconds)
    ax1.set_ylabel(unit); ax2.set_ylabel(unit)
    ax2.set_title(f"model path: notch 50 Hz Q{args.q:g} + low-pass {args.lp:g} Hz (causal), window mean removed")
    status = fig.suptitle("waiting for data ...")
    t_start = time.time()

    def ylim(ax, y):
        lo, hi = np.percentile(y, [0.5, 99.5])
        pad = 0.15 * (hi - lo) + 1e-9
        ax.set_ylim(lo - pad, hi + pad)

    def update(_):
        if src.error:
            status.set_text(src.error); status.set_color("red")
            return l1, l2
        new = np.asarray(src.take(), float) / scale
        if len(new):
            raw_buf.extend(new)
            filt_buf.extend(chain(new))
        n = len(raw_buf)
        if n < 10:
            return l1, l2
        t = np.arange(n) / fs
        r = np.asarray(raw_buf); f = np.asarray(filt_buf)
        l1.set_data(t, r - r.mean()); l2.set_data(t, f - f.mean())
        ax1.set_xlim(0, max(args.seconds, t[-1]))
        ylim(ax1, r - r.mean()); ylim(ax2, f - f.mean())
        mains = mains_amplitude(r[-int(fs):], fs)
        ax1.set_title(f"as received (mean removed): 50 Hz mains amplitude {mains:.3g} {unit} (last 1 s)")
        el = time.time() - t_start
        rate = len(src.all_vals) / el if el > 1 else float("nan")
        status.set_text(f"{len(src.all_vals)} samples, received rate {rate:.1f} Hz (expected {fs:g}), "
                        f"unparsed lines {src.bad_lines}")
        return l1, l2

    anim = FuncAnimation(fig, update, interval=100, cache_frame_data=False)  # noqa: F841 (keep a reference)
    plt.tight_layout()
    try:
        plt.show()
    finally:
        src.running = False
        if args.save and src.all_vals:
            np.savez(args.save, values=np.asarray(src.all_vals), t=np.asarray(src.all_times),
                     fs=fs, scale=scale, port=args.port)
            print(f"saved {len(src.all_vals)} samples to {args.save}")


if __name__ == "__main__":
    main()
