"""
Plot a raw ECG from a txt file -- the whole recording on top and a zoomed window below drawn on
ECG-paper grid (small squares 40 ms x 0.1 mV, large squares 200 ms x 0.5 mV).

Usage (from the folder with the scripts):
    python plot_raw_ecg.py                                   # default ECG file in this folder
    python plot_raw_ecg.py 00057_hr_V2_minus_V1.txt          # any one-column txt file
    python plot_raw_ecg.py --start 4 --dur 3                 # zoom window 4..7 s
    python plot_raw_ecg.py 00057_hr_12lead.txt --column 1    # one column of a multi-column file (0 = first)
    python plot_raw_ecg.py 00057_hr_12lead.txt --all         # every column, stacked (12-lead view)
    python plot_raw_ecg.py --fs 360 --units uV --save        # other sampling rate / units, save PNG
"""
import argparse
import os

import numpy as np

from ecg_io import DEFAULT_ECG, load_ecg_txt, setup_style, finish, C_BLUE, C_GREY


def read_column_names(path):
    """Column names if the first non-empty line of the file is a text header, else None."""
    with open(path) as f:
        for line in f:
            parts = line.replace(",", " ").split()
            if not parts:
                continue
            try:
                [float(p) for p in parts]
                return None
            except ValueError:
                return parts
    return None


def ecg_paper_grid(ax, t0, t1, y0, y1, units="mV"):
    """Pink ECG-paper grid: 0.04 s / 0.1 mV minor, 0.2 s / 0.5 mV major (scaled for uV)."""
    s = 1000.0 if units.lower() in ("uv", "µv") else 1.0
    ax.grid(False)
    ax.set_xticks(np.arange(np.floor(t0 / 0.2) * 0.2, t1 + 1e-9, 0.2))
    ax.set_xticks(np.arange(np.floor(t0 / 0.04) * 0.04, t1 + 1e-9, 0.04), minor=True)
    ax.set_yticks(np.arange(np.floor(y0 / (0.5 * s)) * 0.5 * s, y1 + 1e-9, 0.5 * s))
    ax.set_yticks(np.arange(np.floor(y0 / (0.1 * s)) * 0.1 * s, y1 + 1e-9, 0.1 * s), minor=True)
    ax.grid(which="major", color="#f0a3a3", lw=0.8)
    ax.grid(which="minor", color="#f8d7d7", lw=0.4)
    ax.tick_params(which="minor", length=0)
    labels = [f"{v:g}" if i % 5 == 0 else "" for i, v in enumerate(ax.get_xticks())]
    ax.set_xticklabels(labels)


def plot_raw_ecg(ecg, fs=500, start=0.0, dur=5.0, title="Raw ECG", units="mV", save=False, show=True):
    plt = setup_style()
    ecg = np.asarray(ecg, dtype=float)
    n = len(ecg)
    t = np.arange(n) / fs
    a = max(0, int(start * fs)); b = min(n, a + int(dur * fs))
    fig, ax = plt.subplots(2, 1, figsize=(13, 7.5), gridspec_kw={"height_ratios": [1, 1.6]})

    ax[0].plot(t, ecg, color=C_GREY, lw=0.8)
    ax[0].axvspan(a / fs, b / fs, color=C_BLUE, alpha=0.08, lw=0)
    ax[0].set_xlim(0, n / fs)
    ax[0].set_title(f"{title} -- whole recording: {n} samples, {n / fs:.1f} s at {fs} Hz  "
                    f"(min {ecg.min():.3g}, max {ecg.max():.3g} {units}; shaded = zoom below)")
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel(units)

    seg = ecg[a:b]
    pad = 0.1 * (seg.max() - seg.min() + 1e-12)
    y0, y1 = seg.min() - pad, seg.max() + pad
    ax[1].plot(t[a:b], seg, color="#1a1a1a", lw=1.0)
    ax[1].set_xlim(a / fs, b / fs); ax[1].set_ylim(y0, y1)
    ecg_paper_grid(ax[1], a / fs, b / fs, y0, y1, units)
    ax[1].set_title(f"Zoom {a / fs:.1f}-{b / fs:.1f} s on ECG paper (small square 40 ms x 0.1 mV)")
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel(units)
    return finish(fig, "raw_ecg", save, show)


def plot_all_columns(data, names, fs=500, start=0.0, dur=None, units="mV", save=False, show=True):
    """Every column of a multi-column file, stacked with a shared time axis (e.g. 12-lead)."""
    plt = setup_style()
    n, m = data.shape
    a = max(0, int(start * fs)); b = n if dur is None else min(n, a + int(dur * fs))
    t = np.arange(a, b) / fs
    fig, ax = plt.subplots(m, 1, figsize=(13, 0.9 * m + 1.5), sharex=True)
    for i in range(m):
        ax[i].plot(t, data[a:b, i], color="#1a1a1a", lw=0.8)
        ax[i].set_ylabel(names[i] if names else f"col {i}", rotation=0, ha="right", va="center")
        ax[i].grid(color="#f2c4c4", lw=0.5)
    ax[0].set_title(f"Raw ECG -- {m} channels, {fs} Hz ({units})")
    ax[-1].set_xlabel("time (s)")
    return finish(fig, "raw_ecg_all_columns", save, show)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot a raw ECG txt file")
    ap.add_argument("file", nargs="?", default=DEFAULT_ECG)
    ap.add_argument("--fs", type=int, default=500, help="sampling frequency in Hz (default 500)")
    ap.add_argument("--start", type=float, default=0.0, help="zoom window start (s)")
    ap.add_argument("--dur", type=float, default=5.0, help="zoom window length (s)")
    ap.add_argument("--column", type=int, default=-1, help="column of a multi-column file (default: last)")
    ap.add_argument("--all", action="store_true", help="plot every column of a multi-column file")
    ap.add_argument("--units", default="mV", help="amplitude units shown on the axis (default mV)")
    ap.add_argument("--save", action="store_true", help="also save the PNG to test_results/")
    args = ap.parse_args()

    names = read_column_names(args.file)
    name = os.path.basename(args.file)
    if args.all:
        width = len(names) if names else next(len(l.replace(',', ' ').split()) for l in open(args.file) if l.strip())
        cols = [load_ecg_txt(args.file, column=i) for i in range(width)]
        data = np.column_stack(cols)
        print(f"{name}: {data.shape[0]} samples x {data.shape[1]} columns")
        plot_all_columns(data, names, args.fs, units=args.units, save=args.save)
    else:
        ecg = load_ecg_txt(args.file, column=args.column)
        label = f"{name}" + (f" [{names[args.column]}]" if names else "")
        print(f"{label}: {len(ecg)} samples, {len(ecg) / args.fs:.1f} s, min {ecg.min():.3f}, max {ecg.max():.3f}")
        plot_raw_ecg(ecg, args.fs, args.start, args.dur, title=f"Raw ECG: {label}", units=args.units, save=args.save)
