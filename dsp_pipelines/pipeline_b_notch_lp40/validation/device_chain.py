"""
device_chain.py: the model input chain as it will run on the device, validated on PTB-XL, then applied to
real PYNQ recordings.

Device chain (chain D3, adapted to the board):  ADS1299 at 250 SPS -> causal 50 Hz notch (Q) -> causal
2nd-order Butterworth 40 Hz low-pass -> upsample x2 to 500 Hz (polyphase) -> 10 s buffer -> each model window
minus its own mean -> (x - mu) / sd -> model.

1) validate   PTB-XL folds 9 / 10 are taken down to 250 Hz first (polyphase anti-alias decimation, as the
              ADS1299's own decimation filter does), then run through the chain. Variants: notch Q = 30
              (current PYNQ script) and Q = 5 (wide notch: the real recording has ~17 mV of mains, Q30 leaves
              ~0.07 mV, Q5 ~0.015 mV). Thresholds chosen on clean fold 9; AUROC / dAUROC vs raw on fold 10,
              clean and noisy. Writes ~/export/config_device_q<Q>.json for both.
      python device_chain.py validate
2) recording  runs a PYNQ recording (recordings/<name>.npz) through the chain and both models: one prediction
              per 10 s window (step 5 s), heart rate from R peaks, and a plot.
      python device_chain.py recording ~/recordings/still_01.npz --q 5
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, iirnotch, lfilter, lfilter_zi, resample_poly
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_batch as sb          # noqa: E402
import recalibrate as rc          # noqa: E402
from chain_test import score_wm   # noqa: E402  (window-mean scorer, self-checked against score_batch.score)

FS_DEV, FS_MODEL = 250, 500
PRE = 500                          # 2 s pre-roll at 250 Hz


def chain(x_mv, q, fs_in=FS_DEV):
    """Device model-path chain on one signal in mV at fs_in. Returns the 500 Hz model input."""
    x = np.asarray(x_mv, float)
    if fs_in == FS_MODEL:                                   # PTB-XL simulation: the ADC delivers 250 Hz
        x = resample_poly(x, 1, 2)
    y = np.r_[x[PRE:0:-1], x]                               # steady state, as on a running device
    for b, a in (iirnotch(50.0, q, FS_DEV), butter(2, 40.0, btype="lowpass", fs=FS_DEV)):
        y, _ = lfilter(b, a, y, zi=lfilter_zi(b, a) * y[0])
    return resample_poly(y[PRE:], 2, 1).astype(np.float32)


def chain_all(X, q):
    out = np.full((len(X), X.shape[1]), np.nan, np.float32)
    for k, x in enumerate(X):
        if not np.isnan(x).any():
            y = chain(x, q, FS_MODEL)
            out[k, :min(len(y), X.shape[1])] = y[:X.shape[1]]
    return out


def device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_model(c, dev):
    import torch
    m = sb.build_model(c["model"], 1, 1)
    m.load_state_dict(torch.load(sb.EXPORT / c["weights"], map_location="cpu"))
    return m.to(dev).eval()


# ----------------------------------------------------------------------------- 1) validate
def validate(a):
    batch = Path(a.batch); dev = device()
    cfg = json.loads((sb.EXPORT / "config.json").read_text())
    db = pd.read_csv(sb.PTB / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(sb.PTB / "scp_statements.csv", index_col=0)
    lab9, lab10 = rc.labels_for_fold(db, scp, 9), sb.build_labels()
    z9 = np.load(batch / "ns_f9_raw.npz"); ids9, X9 = z9["ids"], z9["X"].astype(np.float32)
    z10 = np.load(batch / "raw.npz"); ids10 = z10["ids"]
    conds = {"clean": z10["X"].astype(np.float32)}
    for cnd in a.noise:
        n_, s_ = cnd.split(); f = batch / f"ns_{n_}_{s_}_raw.npz"
        if f.exists():
            conds[cnd] = np.load(f)["X"].astype(np.float32)
    S9 = {q: chain_all(X9, q) for q in a.qs}
    S10 = {(c_, q): chain_all(X, q) for c_, X in conds.items() for q in a.qs}
    print("chains done", flush=True)

    lines, rows = [], []
    for task in ("binary", "af"):
        c = cfg[task]; model = load_model(c, dev)
        lines.append(f"\n================ {task} ================")
        iv = lab9[task].index; y9 = lab9[task].to_numpy()
        df9 = pd.DataFrame(X9, index=ids9)
        p = sb.score(model, df9.loc[iv].to_numpy(np.float32), c, dev)
        ok = abs(rc.thr_balanced(y9, p) - c["threshold_balanced"]) < 1e-3
        lines.append(f"SELF-CHECK fold-9 raw balanced threshold = config.json: {'PASS' if ok else 'FAIL'}")
        th = {}
        for q in a.qs:
            p = score_wm(model, pd.DataFrame(S9[q], index=ids9).loc[iv].to_numpy(np.float32), c, dev)
            th[q] = (rc.thr_balanced(y9, p), rc.thr_sensitivity(y9, p, rc.HIGH_SENS_TARGET))
            lines.append(f"Q{q:g}: fold-9 thresholds balanced {th[q][0]:.4f}, high_sens {th[q][1]:.4f}")
            out = sb.EXPORT / f"config_device_q{q:g}.json"
            dc = json.loads(out.read_text()) if out.exists() else json.loads(json.dumps(cfg))
            dc[task].update(threshold_balanced=th[q][0], threshold_high_sens=th[q][1],
                            input_filter=f"ADS1299 250 SPS -> causal notch 50 Hz Q{q:g} -> causal Butterworth LP 40 Hz order 2 "
                                         f"-> resample_poly x2 to 500 Hz -> per-window mean removal",
                            thresholds_chosen_on="PTB-XL fold 9 decimated to 250 Hz and passed through the device chain",
                            demean_windows=True, device_fs=FS_DEV)
            out.write_text(json.dumps(dc, indent=2))
        yt = lab10[task]; ids = yt.index[yt.index.isin(ids10)]; y = yt.loc[ids].to_numpy()
        lines.append(f"{'condition':9s} {'signal':6s} {'AUROC':>6s}  {'dAUROC vs raw same cond. [95% CI]':36s} {'sens':>5s} {'spec':>5s} {'FP':>4s}")
        for c_, X in conds.items():
            pr = sb.score(model, pd.DataFrame(X, index=ids10).loc[ids].to_numpy(np.float32), c, dev)
            m0 = sb.at_threshold(y, pr, c["threshold_balanced"])
            lines.append(f"{c_:9s} {'raw':6s} {roc_auc_score(y, pr):6.4f}  {'':36s} {m0['sens']:5.3f} {m0['spec']:5.3f} {m0['fp']:4d}")
            for q in a.qs:
                pq = score_wm(model, pd.DataFrame(S10[(c_, q)], index=ids10).loc[ids].to_numpy(np.float32), c, dev)
                d = sb.paired_delta_auroc(y, pr, pq, a.n_boot); m = sb.at_threshold(y, pq, th[q][0])
                rows.append(dict(task=task, condition=c_, q=q, auroc=roc_auc_score(y, pq), d=d[0], lo=d[1], hi=d[2],
                                 sens=m["sens"], spec=m["spec"], fp=m["fp"]))
                lines.append(f"{c_:9s} {'Q%g' % q:6s} {roc_auc_score(y, pq):6.4f}  {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]{'':9s} "
                             f"{m['sens']:5.3f} {m['spec']:5.3f} {m['fp']:4d}")
            lines.append("")
    pd.DataFrame(rows).to_csv(batch / "device_chain_summary.csv", index=False)
    txt = "\n".join(lines); print(txt); (batch / "device_chain_summary.txt").write_text(txt + "\n")
    print(f"\nwrote {', '.join(str(sb.EXPORT / f'config_device_q{q:g}.json') for q in a.qs)}")


# ----------------------------------------------------------------------------- 2) recording
def recording(a):
    z = np.load(Path(a.file).expanduser())
    fs = int(z["fs"]); assert fs == FS_DEV, f"recording is {fs} Hz, chain expects {FS_DEV}"
    x = z["code"] * (float(z["vref"]) / (int(z["gain"]) * (2 ** 23 - 1))) * 1e3          # mV
    if a.invert:
        x = -x
    t = z["t"]; gaps = np.where(np.diff(t) > 1.5 / fs)[0]
    y = chain(x, a.q, fs)                                                                  # 500 Hz model input
    cfg = json.loads((sb.EXPORT / f"config_device_q{a.q:g}.json").read_text())
    dev = device()
    # heart rate from the model input (0.5 Hz high-pass only for peak finding)
    yh = filtfilt(*butter(2, 0.5, btype="highpass", fs=FS_MODEL), y)
    sgn = 1 if np.percentile(yh, 99.5) >= -np.percentile(yh, 0.5) else -1
    cand, _ = find_peaks(sgn * yh, distance=int(0.33 * FS_MODEL), prominence=0.1)      # all candidate beats
    amp = (sgn * yh)[cand]
    # 10 s windows, step 5 s, skip the first 2 s (filter settling)
    W, step, start = 10 * FS_MODEL, 5 * FS_MODEL, 2 * FS_MODEL
    starts = list(range(start, len(y) - W + 1, step))
    X = np.stack([y[s:s + W] for s in starts]).astype(np.float32)
    res = pd.DataFrame({"t_start_s": [s / FS_MODEL for s in starts]})
    for task in ("binary", "af"):
        c = cfg[task]; model = load_model(c, dev)
        p = score_wm(model, X, c, dev, demean=True)
        res[f"p_{task}"] = p
        res[f"{task}_call"] = np.where(p >= c["threshold_balanced"], c.get("positive", "positive"), c.get("negative", "negative"))
    hr, keep = [], set()
    for s in starts:                       # per window: beats = candidates above 45% of the median of the 8 largest
        m = (cand >= s) & (cand < s + W)
        if m.sum() < 3:
            hr.append(np.nan); continue
        ref = np.median(np.sort(amp[m])[-8:])
        r = cand[m][amp[m] > 0.45 * ref]; keep.update(r.tolist())
        hr.append(60 * FS_MODEL / np.median(np.diff(r)) if len(r) > 2 else np.nan)
    pk = np.array(sorted(keep), int)
    res["hr_bpm"] = np.round(hr, 1)
    gap_s = gaps / fs
    res["gap_in_window"] = [bool(((gap_s >= s / FS_MODEL) & (gap_s < (s + W) / FS_MODEL)).any()) for s in starts]
    with pd.option_context("display.width", 160, "display.precision", 3):
        print(f"{a.file}: {len(x) / fs:.1f} s at {fs} Hz, chain Q{a.q:g}, dominant QRS polarity {'+' if sgn > 0 else '-'}"
              f"{' (signal inverted with --invert)' if a.invert else ''}, {len(gaps)} timing gaps")
        print(f"thresholds: binary {cfg['binary']['threshold_balanced']:.4f}, AF {cfg['af']['threshold_balanced']:.4f}")
        print(res.to_string(index=False))
    out = Path(a.file).with_suffix("").as_posix() + f"_pred_q{a.q:g}.csv"; res.to_csv(out, index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    tt = np.arange(len(y)) / FS_MODEL
    ax[0].plot(tt, y - np.median(y), lw=0.5); ax[0].plot(pk / FS_MODEL, (y - np.median(y))[pk], "v", ms=3)
    for g in gap_s:
        ax[0].axvline(g, color="r", lw=0.6, alpha=0.5)
    ax[0].set_ylabel("model input (mV)"); ax[0].set_title(f"{Path(a.file).name}: device chain Q{a.q:g} (red = timing gaps)")
    mid = res.t_start_s + 5
    ax[1].plot(mid, res.p_binary, "o-", label="p(abnormal)"); ax[1].plot(mid, res.p_af, "s-", label="p(AF)")
    ax[1].axhline(cfg["binary"]["threshold_balanced"], color="C0", ls=":"); ax[1].axhline(cfg["af"]["threshold_balanced"], color="C1", ls=":")
    ax[1].set_ylim(0, 1); ax[1].set_xlabel("time (s), window centre"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); png = Path(a.file).with_suffix("").as_posix() + f"_pred_q{a.q:g}.png"; fig.savefig(png, dpi=120)
    print(f"saved {out} and {png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    v = sp.add_parser("validate")
    v.add_argument("--qs", nargs="+", type=float, default=[30.0, 5.0])
    v.add_argument("--noise", nargs="+", default=["em 12", "ma 12", "mix 6"])
    v.add_argument("--batch", default=str(Path.home() / "dsp_test" / "batch"))
    v.add_argument("--n_boot", type=int, default=500)
    r = sp.add_parser("recording")
    r.add_argument("file"); r.add_argument("--q", type=float, default=5.0)
    r.add_argument("--invert", action="store_true", help="flip the sign (if V1/V2 are wired the other way round)")
    a = ap.parse_args()
    validate(a) if a.cmd == "validate" else recording(a)
