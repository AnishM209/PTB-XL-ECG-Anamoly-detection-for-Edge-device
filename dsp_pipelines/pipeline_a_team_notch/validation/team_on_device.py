"""
team_on_device.py: the team's DSP cleaning pipeline (bank_first_phase: 0.6 s median, LPF 150 Hz, SWT, Christov,
QRS removal, thresholding, ISWT) run in the DEVICE setting, compared with the current model path (chain D3).

The device delivers 250 Hz; the team pipeline is designed for 500 Hz, so the signal is upsampled x2 first.
Variants (all causal front-ends at 250 Hz, as on the board):
  D   : notch 50 Hz (Q) + LP 40 Hz -> x2 -> per-window mean removal          (current device model path)
  TA  : x2 -> team pipeline                                                  (no notch: mains goes in)
  TB  : notch 50 Hz (Q) -> x2 -> team pipeline
  TC  : notch 50 Hz (Q) + LP 40 Hz -> x2 -> team pipeline
Team code: ~/test_cleaning_reset (reset patch for the Christov lock-out). Records are mirror-padded and the
flat PTB-XL tail is trimmed, as clean_dataset.py does, so the thresholds match how cleaned data was made.

1) thresholds   PTB-XL fold 9 decimated to 250 Hz -> each variant -> balanced thresholds, written to
                ~/export/config_team_<variant>_q<Q>.json. Fold 10 AUROC / dAUROC vs clean raw as a check.
                SELF-CHECK: variant D thresholds = config_device_q<Q>.json.
      python team_on_device.py thresholds --q 5          (or sbatch team_on_device.sbatch)
2) recording    every variant on a PYNQ recording: p(abnormal), p(AF) per 10 s window (step 5 s), HR from the
                team's Christov R peaks, plot.
      python team_on_device.py recording ~/recordings/move_01.npz --q 5
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi, resample_poly

HERE = Path(__file__).resolve().parent
TEAM_DIR = Path(os.environ.get("TEAM_DIR", Path.home() / "test_cleaning_reset"))
sys.path.insert(0, str(HERE))

FS_DEV, FS = 250, 500
PRE = 500                       # 2 s pre-roll at 250 Hz (as device_chain.chain)
EDGE_PAD = 512                  # as clean_dataset.py
FLAT_MIN = 20
VARIANTS = ["D", "TA", "TB", "TC"]
TEAM = ["TA", "TB", "TC"]


# ----------------------------------------------------------------------------- signal paths
def front(x250, variant, q):
    """Causal front-end at 250 Hz (steady-state start, as device_chain.chain), then x2 to 500 Hz."""
    y = np.r_[x250[PRE:0:-1], x250] if len(x250) > PRE else np.asarray(x250, float)
    filt = []
    if variant in ("TB", "TC", "D"):
        filt.append(iirnotch(50.0, q, FS_DEV))
    if variant in ("TC", "D"):
        filt.append(butter(2, 40.0, btype="lowpass", fs=FS_DEV))
    for b, a in filt:
        y, _ = lfilter(b, a, y, zi=lfilter_zi(b, a) * y[0])
    y = y[PRE:] if len(x250) > PRE else y
    return resample_poly(y, 2, 1)


def team(x500):
    """Team pipeline on a 500 Hz signal, mirror-padded like clean_dataset.py. Returns (clean, r_peaks)."""
    from bank_pipeline import bank_first_phase
    n = len(x500)
    ext = np.r_[x500[EDGE_PAD:0:-1], x500, x500[-2:-EDGE_PAD - 2:-1]]
    out = bank_first_phase(ext, FS, warmup=False)
    clean = np.asarray(out["clean"][EDGE_PAD:EDGE_PAD + n], float)
    if np.isnan(clean).any():
        raise RuntimeError("team pipeline left NaN samples")
    pk = np.asarray(out["r_peaks"], int) - EDGE_PAD
    return clean, pk[(pk >= 0) & (pk < n)]


def ptbxl_variant(x500, variant, q):
    """PTB-XL record (500 Hz, mV) -> 250 Hz as the ADC delivers -> variant -> 500 Hz model input."""
    x = np.asarray(x500, float)
    n = len(x)
    flat = 0
    if variant in TEAM:                         # trim the flat tail as clean_dataset.py does
        k = n - 1
        while k > 0 and abs(x[k - 1] - x[-1]) <= 1e-9:
            k -= 1
        flat = n - k if n - k >= FLAT_MIN else 0
    work = x[: n - flat] if flat else x
    y = front(resample_poly(work, 1, 2), variant, q)
    if variant in TEAM:
        y, _ = team(y)
    y = y[: len(work)]
    if flat:
        y = np.r_[y, np.full(flat, y[-1])]
    out = np.full(n, np.nan, np.float32)
    out[: min(n, len(y))] = y[:n]
    return out


def _init_worker():
    sys.path.insert(0, str(TEAM_DIR))


def _one(args):
    x, variant, q = args
    if np.isnan(x).any():
        return np.full(len(x), np.nan, np.float32)
    try:
        return ptbxl_variant(x, variant, q)
    except Exception:                            # a failed record becomes a NaN row (scorer skips it)
        return np.full(len(x), np.nan, np.float32)


def run_all(X, variant, q, workers):
    with Pool(workers, initializer=_init_worker) as pool:
        return np.stack(pool.map(_one, [(x, variant, q) for x in X], chunksize=16))


# ----------------------------------------------------------------------------- scoring
def scorer(variant):
    """Variant D is scored like device_chain (window mean removed); team outputs like clean06 (score_batch)."""
    import score_batch as sb
    from chain_test import score_wm
    if variant == "D":
        return lambda m, X, c, dev: score_wm(m, X, c, dev)
    return lambda m, X, c, dev: sb.score(m, X, c, dev)


def cfg_path(variant, q):
    import score_batch as sb
    if variant == "D":
        return sb.EXPORT / f"config_device_q{q:g}.json"
    return sb.EXPORT / f"config_team_{variant}_q{q:g}.json"


# ----------------------------------------------------------------------------- 1) thresholds
def thresholds(a):
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    import device_chain as dc
    import score_batch as sb
    import recalibrate as rc

    batch = Path(a.batch); dev = dc.device(); q = a.q
    cfg = json.loads((sb.EXPORT / "config.json").read_text())
    db = pd.read_csv(sb.PTB / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(sb.PTB / "scp_statements.csv", index_col=0)
    lab9, lab10 = rc.labels_for_fold(db, scp, 9), sb.build_labels()
    z9 = np.load(batch / "ns_f9_raw.npz"); ids9, X9 = z9["ids"], z9["X"].astype(np.float32)
    z10 = np.load(batch / "raw.npz"); ids10, X10 = z10["ids"], z10["X"].astype(np.float32)
    dev_cfg = json.loads(cfg_path("D", q).read_text()) if cfg_path("D", q).exists() else None

    S9, S10 = {}, {}
    for v in VARIANTS:
        S9[v] = run_all(X9, v, q, a.workers); S10[v] = run_all(X10, v, q, a.workers)
        print(f"{v}: chains done, failed records fold 9 {int(np.isnan(S9[v]).any(1).sum() - np.isnan(X9).any(1).sum())}, "
              f"fold 10 {int(np.isnan(S10[v]).any(1).sum() - np.isnan(X10).any(1).sum())}", flush=True)

    lines = [f"Team pipeline in the device setting (250 Hz, notch Q{q:g}); thresholds on clean fold 9"]
    new_cfg = {v: json.loads(json.dumps(cfg)) for v in TEAM}
    for task in ("binary", "af"):
        c = cfg[task]; model = dc.load_model(c, dev)
        lines.append(f"\n================ {task} ================")
        iv = lab9[task].index; y9 = lab9[task].to_numpy()
        yt = lab10[task]; ids = yt.index[yt.index.isin(ids10)]; y = yt.loc[ids].to_numpy()
        pr = sb.score(model, pd.DataFrame(X10, index=ids10).loc[ids].to_numpy(np.float32), c, dev)
        lines.append(f"reference clean raw 500 Hz fold 10: AUROC {roc_auc_score(y, pr):.4f}")
        lines.append(f"{'variant':7s} {'thr bal':>7s} {'AUROC':>6s}  {'dAUROC vs raw [95% CI]':28s} {'sens':>5s} {'spec':>5s} {'FP':>4s} {'n':>5s}")
        for v in VARIANTS:
            sc = scorer(v)
            p9 = sc(model, pd.DataFrame(S9[v], index=ids9).loc[iv].to_numpy(np.float32), c, dev)
            k9 = ~np.isnan(p9)
            tb = rc.thr_balanced(y9[k9], p9[k9]); ths = rc.thr_sensitivity(y9[k9], p9[k9], rc.HIGH_SENS_TARGET)
            p = sc(model, pd.DataFrame(S10[v], index=ids10).loc[ids].to_numpy(np.float32), c, dev)
            k = ~np.isnan(p)
            d = sb.paired_delta_auroc(y[k], pr[k], p[k], a.n_boot); m = sb.at_threshold(y[k], p[k], tb)
            star = "*" if d[2] < 0 or d[1] > 0 else " "
            lines.append(f"{v:7s} {tb:7.4f} {roc_auc_score(y[k], p[k]):6.4f}  {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]{star}"
                         f"   {m['sens']:5.3f} {m['spec']:5.3f} {m['fp']:4d} {int(k.sum()):5d}")
            if v == "D" and dev_cfg is not None:
                ok = abs(tb - dev_cfg[task]["threshold_balanced"]) < 1e-3
                lines.append(f"   SELF-CHECK D threshold = {cfg_path('D', q).name}: {'PASS' if ok else 'FAIL'}")
            if v in TEAM:
                new_cfg[v][task].update(threshold_balanced=tb, threshold_high_sens=ths, demean_windows=False,
                                        device_fs=FS_DEV, input_filter=f"ADS1299 250 SPS -> {v} (notch Q{q:g}) -> x2 -> team pipeline",
                                        thresholds_chosen_on="PTB-XL fold 9 decimated to 250 Hz, variant chain")
    for v in TEAM:
        cfg_path(v, q).write_text(json.dumps(new_cfg[v], indent=2))
    txt = "\n".join(lines); print(txt)
    (batch / f"team_on_device_q{q:g}.txt").write_text(txt + "\n")
    print("wrote " + ", ".join(str(cfg_path(v, q)) for v in TEAM))


# ----------------------------------------------------------------------------- 2) recording
def recording(a):
    import pandas as pd
    import device_chain as dc
    _init_worker()
    z = np.load(Path(a.file).expanduser())
    fs = int(z["fs"]); assert fs == FS_DEV, f"recording is {fs} Hz, expected {FS_DEV}"
    x = z["code"] * (float(z["vref"]) / (int(z["gain"]) * (2 ** 23 - 1))) * 1e3
    if a.invert:
        x = -x
    gap_s = np.where(np.diff(z["t"]) > 1.5 / fs)[0] / fs
    dev = dc.device()

    sig, peaks = {}, None
    for v in VARIANTS:
        y = front(x, v, a.q)
        if v in TEAM:
            y, pk = team(y)
            if v == "TC":
                peaks = pk                       # R peaks from the variant with the cleanest front-end
        sig[v] = y.astype(np.float32)
    N = min(len(s) for s in sig.values())
    W, step, start = 10 * FS, 5 * FS, 2 * FS
    starts = list(range(start, N - W + 1, step))
    res = pd.DataFrame({"t_start_s": [s / FS for s in starts]})
    thr = {}
    for v in VARIANTS:
        p_ = cfg_path(v, a.q)
        if not p_.exists():
            sys.exit(f"{p_} missing: run  python team_on_device.py thresholds --q {a.q:g}  first")
        cfg = json.loads(p_.read_text()); thr[v] = cfg
        X = np.stack([sig[v][s:s + W] for s in starts])
        for task in ("binary", "af"):
            c = cfg[task]; model = dc.load_model(c, dev)
            res[f"{task}_{v}"] = scorer(v)(model, X, c, dev)
    hr = []
    for s in starts:
        r = peaks[(peaks >= s) & (peaks < s + W)]
        hr.append(round(60 * FS / np.median(np.diff(r)), 1) if len(r) > 2 else np.nan)
    res["hr_team"] = hr
    res["gap"] = [bool(((gap_s >= s / FS) & (gap_s < (s + W) / FS)).any()) for s in starts]

    print(f"{a.file}: {len(x) / fs:.1f} s, notch Q{a.q:g}{', inverted' if a.invert else ''}")
    print("thresholds (balanced)  " + "  ".join(f"{v}: bin {thr[v]['binary']['threshold_balanced']:.3f} "
                                              f"AF {thr[v]['af']['threshold_balanced']:.3f}" for v in VARIANTS))
    show = res.copy()
    for task in ("binary", "af"):
        for v in VARIANTS:
            t = thr[v][task]["threshold_balanced"]
            show[f"{task}_{v}"] = [f"{p:.2f}{'!' if p >= t else ' '}" for p in res[f"{task}_{v}"]]
    with pd.option_context("display.width", 200):
        print(show.to_string(index=False))
    print("('!' = above the balanced threshold: Abnormal / AF)")
    base = Path(a.file).with_suffix("").as_posix() + f"_team_q{a.q:g}"
    res.to_csv(base + ".csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(VARIANTS) + 2, 1, figsize=(13, 13), sharex=True)
    t = np.arange(N) / FS
    for k, v in enumerate(VARIANTS):
        y = sig[v][:N] - np.median(sig[v][:N]); lo, hi = np.percentile(y, [0.2, 99.8])
        ax[k].plot(t, y, lw=0.5); ax[k].set_ylim(lo - 0.2 * (hi - lo), hi + 0.2 * (hi - lo))
        ax[k].set_title({"D": "D: notch + LP40 (current model path)", "TA": "TA: team pipeline, no notch",
                         "TB": "TB: notch -> team", "TC": "TC: notch + LP40 -> team (R peaks marked)"}[v], fontsize=9)
        for g in gap_s:
            ax[k].axvline(g, color="r", lw=0.5, alpha=0.5)
    pk = peaks[peaks < N]; yc = sig["TC"][:N] - np.median(sig["TC"][:N]); ax[3].plot(pk / FS, yc[pk], "v", ms=3)
    mid = res.t_start_s + 5
    for j, task in enumerate(("binary", "af")):
        for v in VARIANTS:
            ax[4 + j].plot(mid, res[f"{task}_{v}"], "o-", label=v, ms=4)
        ax[4 + j].set_ylim(0, 1); ax[4 + j].set_ylabel(f"p({'abnormal' if task == 'binary' else 'AF'})")
        ax[4 + j].legend(ncol=4, fontsize=8); ax[4 + j].grid(alpha=0.3)
    ax[-1].set_xlabel("time (s); probabilities at window centre (thresholds differ per variant, see text output)")
    fig.tight_layout(); fig.savefig(base + ".png", dpi=110)
    print(f"saved {base}.csv and {base}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("thresholds")
    t.add_argument("--q", type=float, default=5.0)
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--n_boot", type=int, default=500)
    t.add_argument("--batch", default=str(Path.home() / "dsp_test" / "batch"))
    r = sp.add_parser("recording")
    r.add_argument("file"); r.add_argument("--q", type=float, default=5.0)
    r.add_argument("--invert", action="store_true")
    a = ap.parse_args()
    thresholds(a) if a.cmd == "thresholds" else recording(a)
