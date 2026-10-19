"""
Run the exported PTB-XL models (normal vs abnormal, and AF) on one ECG recording,
on a laptop CPU. Keep this file in the same folder as models.py, config.json and
the .pt weight files.

Usage:
    python infer.py path/to/cleaned.csv
    python infer.py path/to/signal.npy --fs 500
    python infer.py path/to/record.dat                 (WFDB: needs the .hea next to it)
    python infer.py cleaned.dat --channels 2           (raw .dat with no .hea: [V1, V2] int16)
    python infer.py data.csv --column ecg_clean        (pick one column by name)

Input:
    WFDB .dat with its .hea: V1 and V2 channels, a V2-V1 channel, or one channel.
    Raw .dat with no .hea: give --channels (12 = PTB-XL order, 2 = [V1, V2],
        1 = V2-V1), and --dtype / --gain if not 16-bit at 1000 counts per mV.
    CSV with a header row: V1 and V2 columns (the model input V2-V1 is computed),
        or a single signal column, which is taken to be V2-V1 already.
    .npy: shape (T,) taken as V2-V1, or (T, 2) taken as [V1, V2].
    The signal must be in mV (use --scale, e.g. 0.001 for microvolts) and should
    be about 10 s long. Other sampling rates are resampled to 500 Hz.
"""
import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from models import build_model

HERE = Path(__file__).resolve().parent


def _bipolar_from_channels(sig, names, where):
    """sig: (T, n) in mV. Use V1 and V2 if named, a V2-V1 channel, or a single channel."""
    low = [str(n).strip().lower() for n in names]
    if "v1" in low and "v2" in low:
        return sig[:, low.index("v2")] - sig[:, low.index("v1")], f"V2 minus V1 from {where}"
    for k in ("v2-v1", "v2_v1", "v2v1"):
        if k in low:
            return sig[:, low.index(k)], f"channel '{names[low.index(k)]}' from {where}"
    if sig.shape[1] == 1:
        return sig[:, 0], f"single channel '{names[0]}' from {where}, taken as V2-V1"
    raise SystemExit(f"{where} has channels {list(names)}; cannot find V1 and V2.")


def load_v2_minus_v1(path, column=None, raw=None):
    p = Path(path)
    if p.suffix in (".hea", ".dat") and p.with_suffix(".hea").exists():
        import wfdb
        sig, info = wfdb.rdsamp(str(p.with_suffix("")))
        if info["fs"] != 500:
            print(f"  NOTE: header says {info['fs']} Hz; pass --fs {info['fs']:g} to resample.")
        return _bipolar_from_channels(sig, info["sig_name"], "the WFDB record")
    if p.suffix == ".dat":
        if raw["channels"] is None:
            raise SystemExit("No .hea header next to this .dat file, so its layout is unknown. "
                             "Pass --channels N (and --dtype, --gain if not int16 at 1000 counts/mV).")
        a = np.fromfile(p, dtype=np.dtype(raw["dtype"]).newbyteorder("<"))
        n = raw["channels"]
        if a.size % n:
            raise SystemExit(f"{a.size} values do not divide into {n} channels; check --channels and --dtype")
        sig = a.reshape(-1, n).astype(np.float64) / raw["gain"]
        if n == 12:
            names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        elif n == 2:
            names = ["V1", "V2"]
        else:
            names = [f"ch{i}" for i in range(n)]
        return _bipolar_from_channels(sig, names, f"raw .dat ({n} channel(s), {raw['dtype']})")
    if p.suffix == ".npy":
        a = np.load(p)
        if a.ndim == 1:
            return a, "1-D array taken as V2-V1"
        if a.ndim == 2 and a.shape[1] == 2:
            return a[:, 1] - a[:, 0], "column 2 minus column 1 (V2 minus V1)"
        raise SystemExit(f"Expected shape (T,) or (T, 2), got {a.shape}")
    if p.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(p)
        if column:
            return df[column].to_numpy(), f"column '{column}' taken as V2-V1"
        low = {c: c.strip().lower() for c in df.columns}
        v1 = next((c for c, k in low.items() if k.startswith("v1")), None)
        v2 = next((c for c, k in low.items() if k.startswith("v2") and "v1" not in k), None)
        if v1 and v2:
            return df[v2].to_numpy() - df[v1].to_numpy(), f"'{v2}' minus '{v1}'"
        data = [c for c, k in low.items() if not (k == "t" or k.startswith("time"))]
        if len(data) == 1:
            return df[data[0]].to_numpy(), f"column '{data[0]}' taken as V2-V1"
        raise SystemExit(f"Could not tell which columns to use: {list(df.columns)}. "
                         "Pass --column NAME for a V2-V1 signal.")
    raise SystemExit("Input must be .dat, .hea, .csv or .npy")


@torch.no_grad()
def predict(model, x, win):
    """Sliding windows with 50% overlap, sigmoid outputs averaged, as in testing."""
    T = len(x)
    starts = list(range(0, T - win + 1, max(1, win // 2)))
    if starts[-1] != T - win:
        starts.append(T - win)
    xb = torch.from_numpy(x.astype(np.float32)).reshape(1, 1, -1)
    return float(torch.stack([torch.sigmoid(model(xb[:, :, s:s + win])) for s in starts]).mean())


def main(a):
    cfg = json.loads((HERE / "config.json").read_text())
    x, how = load_v2_minus_v1(a.input, a.column,
                              {"channels": a.channels, "dtype": a.dtype, "gain": a.gain})
    x = np.asarray(x, dtype=np.float64) * a.scale
    if np.isnan(x).any():
        raise SystemExit("The signal contains NaN values")
    fs_model = cfg["binary"]["fs"]
    if a.fs != fs_model:
        from scipy.signal import resample_poly
        r = Fraction(fs_model, a.fs).limit_denominator(1000)
        x = resample_poly(x, r.numerator, r.denominator)
        how += f", resampled {a.fs} Hz to {fs_model} Hz"

    ptp = np.ptp(x)
    print(f"Input: {a.input}\n  {how}\n  {len(x)} samples = {len(x) / fs_model:.1f} s, "
          f"peak-to-peak {ptp:.2f} mV")
    if not 0.2 < ptp < 15:
        print("  WARNING: amplitude looks unusual for an ECG in mV. Check units or use --scale.")
    if abs(len(x) / fs_model - 10) > 0.5:
        print("  NOTE: the models were tested on 10 s records.")

    for task, c in cfg.items():
        win = int(round(c["win_sec"] * c["fs"]))
        if len(x) < win:
            print(f"{task}: signal shorter than the {c['win_sec']} s window, skipped")
            continue
        model = build_model(c["model"], 1, 1)
        model.load_state_dict(torch.load(HERE / c["weights"], map_location="cpu"))
        model.eval()
        p = predict(model, (x - c["mu"][0]) / c["sd"][0], win)
        t = c["threshold_balanced"] if c["threshold_balanced"] is not None else 0.5
        verdict = c["positive"] if p >= t else c["negative"]
        print(f"\n{task:7s} probability {p:.3f}  threshold {t:.3f}  ->  {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--fs", type=int, default=500, help="sampling rate of the input (Hz)")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply input to get mV")
    ap.add_argument("--column", help="CSV column holding a V2-V1 signal")
    ap.add_argument("--channels", type=int, help="raw .dat without .hea: number of interleaved channels")
    ap.add_argument("--dtype", default="int16", help="raw .dat without .hea: int16, int32 or float32")
    ap.add_argument("--gain", type=float, default=1000.0, help="raw .dat without .hea: counts per mV")
    main(ap.parse_args())
