"""
hw_filter_test.py: Step A. How much does the PYNQ acquisition script's own digital filter chain
(python_file_fpr_uart.py: causal 50 Hz notch Q=30, then causal 2nd-order Butterworth band-pass 0.5-40 Hz)
cost the exported V2-V1 seed-0 models, and which signal should the model be fed live?

Variants (all CAUSAL lfilter, as on the device, started in steady state: 2 s mirrored pre-roll + lfilter_zi,
because on the device the filter has been running long before a 10 s window is taken):
  raw         no filtering (reference = Stage 1)
  notch       50 Hz notch only                         <- candidate model input
  notch_hp05  notch + 0.5 Hz high-pass (2nd order)     <- isolates the high-pass part of the band-pass
  notch_lp40  notch + 40 Hz low-pass (2nd order)       <- isolates the low-pass part
  hw          notch + band-pass 0.5-40 Hz              <- exactly what the script sends over UART now
Conditions: clean fold 10, plus noisy fold 10 from the noise stress test (ns_<noise>_<snr>_raw.npz), to see
whether the 40 Hz low-pass buys robustness against muscle noise.

Thresholds: balanced / high_sens chosen on clean fold 9 (ns_f9_raw.npz) filtered the same way, all records
(as for raw). Also reported: the exported raw thresholds (config.json) applied unchanged, i.e. what happens
if the device feeds filtered signals to the models without re-choosing thresholds.
Same prediction, label and threshold code as score_batch.py / recalibrate.py (imported).
SELF-CHECKS: fold-9 raw reproduces config.json thresholds; clean fold-10 raw reproduces Stage 1.

Writes batch/hw_filter_summary.txt, batch/hw_filter_summary.csv, batch/hw_filter_examples.png
Usage: sbatch hw_filter_test.sbatch   (GPU, ~5-10 min)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_batch as sb          # noqa: E402
import recalibrate as rc          # noqa: E402

FS = 500
PRE = 1000                        # 2 s mirrored pre-roll
NOTCH = iirnotch(50.0, 30.0, FS)                                   # identical to the PYNQ script
BP = butter(2, [0.5, 40.0], btype="bandpass", fs=FS)               # identical to the PYNQ script
HP = butter(2, 0.5, btype="highpass", fs=FS)
LP = butter(2, 40.0, btype="lowpass", fs=FS)
CHAINS = {"raw": [], "notch": [NOTCH], "notch_hp05": [NOTCH, HP], "notch_lp40": [NOTCH, LP], "hw": [NOTCH, BP]}
TASKS = ["binary", "af"]


def causal(x, chain):
    if not chain or np.isnan(x).any():
        return x.copy()
    y = np.r_[x[PRE:0:-1], x].astype(float)
    for b, a in chain:
        y, _ = lfilter(b, a, y, zi=lfilter_zi(b, a) * y[0])
    return y[PRE:].astype(np.float32)


def filt_all(X, chain):
    return np.stack([causal(x, chain) for x in X]) if chain else X


def load(path):
    z = np.load(path)
    return z["ids"], z["X"].astype(np.float32)


def load_model(c, device):
    import torch
    m = sb.build_model(c["model"], 1, 1)
    m.load_state_dict(torch.load(sb.EXPORT / c["weights"], map_location="cpu"))
    return m.to(device).eval()


def qrs_ratio(Xf, Xr):
    """Median per-second peak-to-peak (~QRS size) of filtered / raw, both zero-phase high-passed at 0.5 Hz first."""
    from scipy.signal import filtfilt
    out = []
    for f, r in zip(Xf[:300], Xr[:300]):
        if np.isnan(r).any():
            continue
        r, f = filtfilt(*HP, r), filtfilt(*HP, f)     # remove baseline from both, so only the QRS size is compared
        pf = np.median([np.ptp(f[i:i + FS]) for i in range(FS, len(f) - FS, FS)])
        pr = np.median([np.ptp(r[i:i + FS]) for i in range(FS, len(r) - FS, FS)])
        out.append(pf / pr)
    return float(np.median(out))


def main(a):
    batch = Path(a.batch)
    cfg = json.loads((sb.EXPORT / "config.json").read_text())
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    db = pd.read_csv(sb.PTB / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(sb.PTB / "scp_statements.csv", index_col=0)
    lab9, lab10 = rc.labels_for_fold(db, scp, 9), sb.build_labels()

    # ---- signals
    ids9, X9 = load(batch / "ns_f9_raw.npz")
    conds = [("clean", batch / "raw.npz")] + [(f"{n} {s}", batch / f"ns_{n}_{s}_raw.npz")
                                               for n, s in (c.split() for c in a.noise)]
    conds = [(k, p) for k, p in conds if p.exists()]
    F9 = {v: pd.DataFrame(filt_all(X9, ch), index=ids9) for v, ch in CHAINS.items()}
    F10 = {}
    for k, p in conds:
        ids, X = load(p)
        F10[k] = {v: pd.DataFrame(filt_all(X, ch), index=ids) for v, ch in CHAINS.items()}
        print(f"filtered {k}", flush=True)

    lines = [f"QRS peak-to-peak retention (clean fold 10, median of 300 records, both baseline-removed): " +
             ", ".join(f"{v} {qrs_ratio(F10['clean'][v].to_numpy(), F10['clean']['raw'].to_numpy()):.2f}"
                       for v in CHAINS if v != "raw")]
    rows = []
    for task in TASKS:
        c = cfg[task]
        model = load_model(c, device)
        lines.append(f"\n================ {task} ================")
        # thresholds on clean fold 9
        iv = lab9[task].index; y9 = lab9[task].to_numpy()
        th = {}
        for v in CHAINS:
            p = sb.score(model, F9[v].loc[iv].to_numpy(np.float32), c, device)
            th[v] = (rc.thr_balanced(y9, p), rc.thr_sensitivity(y9, p, rc.HIGH_SENS_TARGET))
        ok9 = abs(th["raw"][0] - c["threshold_balanced"]) < 1e-3 and abs(th["raw"][1] - c["threshold_high_sens"]) < 1e-3
        lines.append(f"SELF-CHECK fold-9 raw thresholds reproduce config.json: {'PASS' if ok9 else 'FAIL'}")
        lines.append("fold-9 thresholds (balanced / high_sens): " + ", ".join(f"{v} {t[0]:.4f} / {t[1]:.4f}" for v, t in th.items()))

        yt = lab10[task]
        ids = yt.index[yt.index.isin(F10["clean"]["raw"].index)]
        y = yt.loc[ids].to_numpy()
        P = {(k, v): sb.score(model, F10[k][v].loc[ids].to_numpy(np.float32), c, device) for k in F10 for v in CHAINS}
        e = sb.EXPECTED[task]; m = sb.at_threshold(y, P[("clean", "raw")], c["threshold_balanced"])
        chk = [len(y) == e["n"], int(y.sum()) == e["pos"], all(m[q] == e[q] for q in ("tp", "fn", "fp", "tn"))]
        if e["auroc"] is not None:
            chk.append(abs(roc_auc_score(y, P[("clean", "raw")]) - e["auroc"]) < 0.001)
        lines.append(f"SELF-CHECK clean fold-10 raw reproduces Stage 1: {'PASS' if all(chk) else 'FAIL ' + str(m)}")

        lines.append(f"{'condition':9s} {'signal':11s} {'AUROC':>6s}  {'dAUROC vs raw, same condition [95% CI]':40s} "
                     f"{'own thr: sens':>13s} {'spec':>5s} {'FP':>4s} | {'exported raw thr: sens':>22s} {'spec':>5s} {'FP':>4s}")
        for k in F10:
            for v in CHAINS:
                p = P[(k, v)]
                r = dict(task=task, condition=k, signal=v, auroc=roc_auc_score(y, p))
                if v != "raw":
                    r["d"], r["d_lo"], r["d_hi"] = sb.paired_delta_auroc(y, P[(k, "raw")], p, a.n_boot)
                own = sb.at_threshold(y, p, th[v][0]); exp = sb.at_threshold(y, p, c["threshold_balanced"])
                hs = sb.at_threshold(y, p, th[v][1])
                r.update(bal_sens=own["sens"], bal_spec=own["spec"], bal_fp=own["fp"], hs_sens=hs["sens"],
                         hs_spec=hs["spec"], exp_sens=exp["sens"], exp_spec=exp["spec"], exp_fp=exp["fp"])
                rows.append(r)
                dtxt = f"{r['d']:+.4f} [{r['d_lo']:+.4f}, {r['d_hi']:+.4f}]" if v != "raw" else ""
                lines.append(f"{k:9s} {v:11s} {r['auroc']:6.4f}  {dtxt:40s} {own['sens']:13.3f} {own['spec']:5.3f} "
                             f"{own['fp']:4d} | {exp['sens']:22.3f} {exp['spec']:5.3f} {exp['fp']:4d}")
            lines.append("")

    pd.DataFrame(rows).to_csv(batch / "hw_filter_summary.csv", index=False)
    txt = "\n".join(lines)
    print(txt)
    (batch / "hw_filter_summary.txt").write_text(txt + "\n")
    examples(F10["clean"], batch)
    print(f"\nsaved hw_filter_summary.txt/.csv and hw_filter_examples.png in {batch}")


def examples(F, batch, n=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ids = F["raw"].index[:n]
    t = np.arange(4 * FS, 6 * FS) / FS
    fig, ax = plt.subplots(n, 1, figsize=(12, 2.8 * n), sharex=True)
    for a_, i in zip(np.atleast_1d(ax), ids):
        for v, col, lw in (("raw", "#7f7f7f", 1.6), ("notch", "#1f77b4", 1.0), ("hw", "#d62728", 1.0)):
            s = F[v].loc[i].to_numpy()[4 * FS:6 * FS]
            a_.plot(t, s - np.median(s),
                    color=col, lw=lw, label={"raw": "raw", "notch": "notch only", "hw": "notch + 0.5-40 Hz (PYNQ script)"}[v])
        a_.set_title(f"ecg_id {i}, 4-6 s (each trace median-centred)"); a_.set_ylabel("mV"); a_.grid(alpha=0.3)
    np.atleast_1d(ax)[0].legend(loc="upper right", fontsize=8); np.atleast_1d(ax)[-1].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(batch / "hw_filter_examples.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=str(Path.home() / "dsp_test" / "batch"))
    ap.add_argument("--noise", nargs="+", default=["em 12", "ma 12", "mix 12", "ma 6", "mix 6"],
                    help="noisy fold-10 conditions from the noise stress test, as 'noise snr'")
    ap.add_argument("--n_boot", type=int, default=500)
    main(ap.parse_args())
