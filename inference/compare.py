"""
Compare a raw PTB-XL record with its DSP-cleaned V2-V1 version.

Shows where the cleaning changes the waveform and which parts of the record
change the models' decisions. Keep this file next to infer.py.

Usage:
    python compare.py RAW.dat CLEAN.dat
Prints model probabilities for every window (raw vs cleaned), heart rate and
beat-to-beat variability, and saves compare_<name>.png next to infer.py.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import find_peaks

import infer
from models import build_model

RAW_DAT = {"channels": None, "dtype": "int16", "gain": 1000.0}


def window_probs(x, c):
    win = int(round(c["win_sec"] * c["fs"]))
    m = build_model(c["model"], 1, 1)
    m.load_state_dict(torch.load(infer.HERE / c["weights"], map_location="cpu"))
    m.eval()
    z = torch.from_numpy(((x - c["mu"][0]) / c["sd"][0]).astype(np.float32)).reshape(1, 1, -1)
    starts = list(range(0, len(x) - win + 1, max(1, win // 2)))
    if starts[-1] != len(x) - win:
        starts.append(len(x) - win)
    with torch.no_grad():
        p = [float(torch.sigmoid(m(z[:, :, s:s + win]))) for s in starts]
    return starts, win, p


def rhythm(x, fs):
    # QRS complexes have the steepest slopes, so detect beats on the slope, not the
    # amplitude; tall T waves then cannot be mistaken for beats.
    y = np.abs(np.gradient(x))
    peaks, _ = find_peaks(y, height=0.4 * np.percentile(y, 99.9), distance=int(0.3 * fs))
    rr = np.diff(peaks) / fs
    return len(peaks), 60 / rr.mean(), rr.std() / rr.mean(), rr


def main(raw_path, clean_path):
    cfg = json.loads((infer.HERE / "config.json").read_text())
    fs = cfg["binary"]["fs"]
    raw, _ = infer.load_v2_minus_v1(raw_path, None, RAW_DAT)
    clean, _ = infer.load_v2_minus_v1(clean_path, None, RAW_DAT)
    n = min(len(raw), len(clean))
    if len(raw) != len(clean):
        print(f"NOTE: lengths differ (raw {len(raw)}, clean {len(clean)}); comparing first {n} samples")
    raw, clean = np.asarray(raw[:n], float), np.asarray(clean[:n], float)
    diff = clean - raw
    lag = int(np.argmax(np.correlate(clean - clean.mean(), raw - raw.mean(), "full")) - (n - 1))
    print(f"Peak-to-peak: raw {np.ptp(raw):.2f} mV, clean {np.ptp(clean):.2f} mV; "
          f"RMS of (clean - raw) {np.sqrt(np.mean(diff ** 2)):.3f} mV; best alignment lag {lag} samples")

    for name, x in (("raw", raw), ("clean", clean)):
        k, hr, cv, rr = rhythm(x, fs)
        print(f"{name:5s}: {k} beats, heart rate {hr:.0f} bpm, RR variability (CV) {cv:.2f}  "
              f"RR (s): {np.round(rr, 2).tolist()}")
    print("      (an RR CV above about 0.10 suggests an irregular rhythm)")

    for task, c in cfg.items():
        starts, win, pr = window_probs(raw, c)
        _, _, pc = window_probs(clean, c)
        t = c["threshold_balanced"]
        print(f"\n{task}: window probabilities (threshold {t:.3f})")
        print("  window (s)     raw    clean")
        for s, a, b in zip(starts, pr, pc):
            print(f"  {s / fs:4.2f}-{(s + win) / fs:5.2f}   {a:.3f}   {b:.3f}")
        print(f"  mean        {np.mean(pr):.3f}   {np.mean(pc):.3f}")

    tt = np.arange(n) / fs
    fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    ax[0].plot(tt, raw, lw=0.8, color="k", label="raw V2-V1")
    ax[0].plot(tt, clean, lw=0.8, color="#d62728", alpha=0.8, label="DSP cleaned")
    ax[0].set_ylabel("mV")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[1].plot(tt, diff, lw=0.8, color="#1f77b4")
    ax[1].set_ylabel("clean - raw (mV)")
    ax[1].set_xlabel("Time (s)")
    for a in ax:
        a.grid(alpha=0.3)
    name = Path(clean_path).stem
    fig.suptitle(name, fontsize=10)
    fig.tight_layout()
    out = infer.HERE / f"compare_{name}.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python compare.py RAW.dat CLEAN.dat")
    main(sys.argv[1], sys.argv[2])
