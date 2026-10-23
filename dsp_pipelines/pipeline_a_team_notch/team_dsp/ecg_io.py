"""
Input / plotting helpers shared by every block of the ECG pipeline.

    from ecg_io import load_ecg_txt
    ecg = load_ecg_txt("ECG Filtering_python/ecg_raw_500hz.txt")

Every block file can also be run on its own and will plot its output, e.g.

    python median_filter.py                                   # uses the default ECG file
    python christov_r_peak_detector.py my_ecg.txt --fs 500    # any txt file
    python QRS_removal.py --start 4 --dur 6 --save            # zoom 4..10 s, save PNG to test_results/

matplotlib is only imported inside the plotting functions, so the processing code itself
does not need matplotlib (important for running the blocks on the processor).
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# >>> The file every script uses when you run it without a file name (change this to switch files) <<<
DEFAULT_ECG_NAME = "00057_hr_V2_minus_V1.txt"     # PTB-XL record 00057_hr, V2 - V1, 500 Hz, mV, 10 s


def _find_default_ecg():
    """Default input file: the V2 - V1 record next to these scripts, else ecg_raw_500hz.txt here or in
    'ECG Filtering_python/', else the only other .txt file in this folder (ignoring *_clean.txt outputs)."""
    for cand in (os.path.join(HERE, DEFAULT_ECG_NAME),
                 os.path.join(HERE, "ecg_raw_500hz.txt"),
                 os.path.join(HERE, "ECG Filtering_python", "ecg_raw_500hz.txt")):
        if os.path.isfile(cand):
            return cand
    txts = [f for f in os.listdir(HERE) if f.lower().endswith(".txt") and not f.endswith("_clean.txt")]
    return os.path.join(HERE, txts[0]) if len(txts) == 1 else os.path.join(HERE, "ecg_raw_500hz.txt")


DEFAULT_ECG = _find_default_ecg()
RESULTS_DIR = os.path.join(HERE, "test_results")

# colours used in every plot (blue, orange, aqua, yellow, grey, ink)
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_GREY, C_INK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#9a9a95", "#2b2b2a"


# ----------------------------------------------------------------------------- input
def load_ecg_txt(path=DEFAULT_ECG, column=-1):
    """
    Read an ECG recording from a text file and return it as a 1-D float numpy array.

    Works for:
      - one sample per line                      (e.g. ecg_raw_500hz.txt)
      - several columns (time, value / lead1, lead2, ...) separated by spaces, tabs or commas;
        `column` picks which one (default: the last column)
      - header / comment lines and blank lines   (any line that is not numeric is skipped)
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            parts = line.replace(",", " ").replace(";", " ").split()
            if not parts:
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:          # header or comment line
                continue
    if not rows:
        raise ValueError(f"No numeric samples found in {path}")
    width = min(len(r) for r in rows)
    data = np.array([r[:width] for r in rows], dtype=float)
    return data[:, column] if data.ndim == 2 else data


def parse_args(description):
    """Common command line for every block: [file] [--fs] [--start] [--dur] [--save]."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("file", nargs="?", default=DEFAULT_ECG, help="ECG txt file (default: %(default)s)")
    ap.add_argument("--fs", type=int, default=500, help="sampling frequency in Hz (default 500)")
    ap.add_argument("--start", type=float, default=0.0, help="plot window start in s (default 0)")
    ap.add_argument("--dur", type=float, default=10.0, help="plot window length in s (default 10 = whole V2-V1 record)")
    ap.add_argument("--save", action="store_true", help="also save the figure to test_results/")
    args = ap.parse_args()
    args.ecg = load_ecg_txt(args.file)
    print(f"Loaded {len(args.ecg)} samples ({len(args.ecg) / args.fs:.1f} s at {args.fs} Hz) from {args.file}")
    return args


# ----------------------------------------------------------------------------- plotting
def plot_window(n, fs, start=0.0, dur=6.0):
    """Sample slice and time axis (s) for the plotting window."""
    a = max(0, int(start * fs))
    b = min(n, a + int(dur * fs)) if dur > 0 else n
    return slice(a, b), np.arange(a, b) / fs


def setup_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 1.0,
    })
    return plt


def finish(fig, name, save=False, show=True):
    """Tight layout, optionally save to test_results/<name>.png, then show."""
    import matplotlib.pyplot as plt
    if not getattr(fig, "_layout_done", False):
        fig.tight_layout()
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, name + ".png")
        fig.savefig(path, dpi=110)
        print("saved", path)
    if show:
        plt.show()
    return fig
