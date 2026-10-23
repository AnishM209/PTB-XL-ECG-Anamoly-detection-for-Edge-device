"""
mains_test.py: how much 50 Hz mains interference can the device model path tolerate, per notch Q?

Same chain, models and scoring as device_chain.py validate (imports its functions):
  PTB-XL V2-V1 (500 Hz) + synthetic mains -> decimate to 250 Hz -> causal notch 50 Hz (Q) -> causal LP 40 Hz
  -> x2 to 500 Hz -> per-window mean removal -> models.

Mains model, per record (same realisation for every Q, so the comparison is paired):
  frequency 50 Hz + fixed offset uniform in +-FDEV (default 0.05 Hz) + slow drift +-FDRIFT (default 0.02 Hz),
  (the recordings still_01 B and C show the mains peak at 49.97 to 50.02 Hz, so +-0.05 Hz is slightly pessimistic;
  the narrow Q30 notch is the one most sensitive to this offset),
  amplitude modulation +-20 %, random phase. Amplitudes are peak values in mV, like the recording summary line
  "50 Hz mains amplitude". Harmonics are not added: the 250 SPS decimation removes 150 Hz and above.

Thresholds per Q are chosen on CLEAN fold 9 (mains 0), as the device would use them; fold 10 is scored at
every mains level. Residual mains = the chain applied to the mains alone (the chain is linear), rms in mV.

SELF-CHECKS: fold-9 raw balanced threshold = config.json, and at mains 0 the Q30 / Q5 thresholds and AUROC
reproduce device_chain_summary.txt (binary Q30 0.5937 / 0.8481, Q5 0.5750 / 0.8440; AF Q30 0.3664 / 0.9632,
Q5 0.3286 / 0.9634).

    python mains_test.py                       # or: sbatch mains_test.sbatch
    python mains_test.py --qs 30 15 10 5 --amps 0 0.5 1 2 5 17
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FS = 500
EXPECTED = {("binary", 30.0): (0.5937, 0.8481), ("binary", 5.0): (0.5750, 0.8440),
            ("af", 30.0): (0.3664, 0.9632), ("af", 5.0): (0.3286, 0.9634)}


FDEV, FDRIFT = 0.05, 0.02


def mains(n, amp_mv, rng, fs=FS):
    """One mains realisation: n samples at fs, peak amplitude amp_mv."""
    if amp_mv == 0:
        return np.zeros(n)
    t = np.arange(n) / fs
    f0 = 50.0 + rng.uniform(-FDEV, FDEV)
    f = f0 + FDRIFT * np.sin(2 * np.pi * t / rng.uniform(5, 20) + rng.uniform(0, 2 * np.pi))
    phase = 2 * np.pi * np.cumsum(f) / fs + rng.uniform(0, 2 * np.pi)
    a = amp_mv * (1 + 0.2 * np.sin(2 * np.pi * t / rng.uniform(3, 15) + rng.uniform(0, 2 * np.pi)))
    return a * np.sin(phase)


def rec_rng(seed, ecg_id):
    return np.random.default_rng([seed, int(ecg_id)])


def chain_with_mains(X, ids, q, amp, seed, chain):
    out = np.full(X.shape, np.nan, np.float32)
    for k, (x, i) in enumerate(zip(X, ids)):
        if np.isnan(x).any():
            continue
        y = chain(x + mains(len(x), amp, rec_rng(seed, i)), q, FS)
        out[k, :min(len(y), X.shape[1])] = y[:X.shape[1]]
    return out


def residual_rms(q, amp, chain, n_real=20, seed=123, n=5000):
    """rms (mV) of the mains left in the model input, mains alone through the chain, first 1 s skipped."""
    if amp == 0:
        return 0.0
    r = [np.sqrt(np.mean(chain(mains(n, amp, rec_rng(seed, k)), q, FS)[FS:] ** 2)) for k in range(n_real)]
    return float(np.median(r))


def main(a):
    global FDEV, FDRIFT
    FDEV, FDRIFT = a.fdev, a.fdrift
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    import device_chain as dc
    import score_batch as sb
    import recalibrate as rc
    from chain_test import score_wm

    batch = Path(a.batch); dev = dc.device()
    cfg = json.loads((sb.EXPORT / "config.json").read_text())
    db = pd.read_csv(sb.PTB / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(sb.PTB / "scp_statements.csv", index_col=0)
    lab9, lab10 = rc.labels_for_fold(db, scp, 9), sb.build_labels()
    z9 = np.load(batch / "ns_f9_raw.npz"); ids9, X9 = z9["ids"], z9["X"].astype(np.float32)
    z10 = np.load(batch / "raw.npz"); ids10, X10 = z10["ids"], z10["X"].astype(np.float32)

    res = {(q, amp): residual_rms(q, amp, dc.chain) for q in a.qs for amp in a.amps}
    lines = [f"mains frequency 50 Hz +- {FDEV} Hz offset, +- {FDRIFT} Hz drift, amplitude +-20 %",
             "Residual mains in the model input (mV rms; ECG band of the recording is ~0.2 mV rms):",
             "  amp (mV peak) " + " ".join(f"{'Q%g' % q:>8s}" for q in a.qs)]
    for amp in a.amps:
        lines.append(f"  {amp:13g} " + " ".join(f"{res[(q, amp)]:8.4f}" for q in a.qs))

    S9 = {q: dc.chain_all(X9, q) for q in a.qs}                       # clean fold 9 for the thresholds
    print("fold-9 chains done", flush=True)
    rows = []
    for task in ("binary", "af"):
        c = cfg[task]; model = dc.load_model(c, dev)
        lines.append(f"\n================ {task} ================")
        iv = lab9[task].index; y9 = lab9[task].to_numpy()
        p = sb.score(model, pd.DataFrame(X9, index=ids9).loc[iv].to_numpy(np.float32), c, dev)
        ok = abs(rc.thr_balanced(y9, p) - c["threshold_balanced"]) < 1e-3
        lines.append(f"SELF-CHECK fold-9 raw balanced threshold = config.json: {'PASS' if ok else 'FAIL'}")
        th = {}
        for q in a.qs:
            pq = score_wm(model, pd.DataFrame(S9[q], index=ids9).loc[iv].to_numpy(np.float32), c, dev)
            th[q] = rc.thr_balanced(y9, pq)
            lines.append(f"Q{q:g}: clean fold-9 balanced threshold {th[q]:.4f}")
        yt = lab10[task]; ids = yt.index[yt.index.isin(ids10)]; y = yt.loc[ids].to_numpy()
        X10t = pd.DataFrame(X10, index=ids10).loc[ids].to_numpy(np.float32)
        pr = sb.score(model, X10t, c, dev)                                 # clean raw 500 Hz reference
        m0 = sb.at_threshold(y, pr, c["threshold_balanced"])
        lines.append(f"reference clean raw 500 Hz: AUROC {roc_auc_score(y, pr):.4f}, sens {m0['sens']:.3f}, "
                     f"spec {m0['spec']:.3f}, FP {m0['fp']}")
        lines.append(f"{'mains mV':>8s} {'chain':>5s} {'resid':>7s} {'AUROC':>6s}  {'dAUROC vs clean raw [95% CI]':30s} "
                     f"{'sens':>5s} {'spec':>5s} {'FP':>4s}")
        for amp in a.amps:
            for q in a.qs:
                S = chain_with_mains(X10t, ids, q, amp, a.seed, dc.chain)
                pq = score_wm(model, S, c, dev)
                au = roc_auc_score(y, pq); d = sb.paired_delta_auroc(y, pr, pq, a.n_boot)
                m = sb.at_threshold(y, pq, th[q])
                rows.append(dict(task=task, mains_mv=amp, q=q, residual_mv_rms=res[(q, amp)], auroc=au, d=d[0],
                                 lo=d[1], hi=d[2], sens=m["sens"], spec=m["spec"], fp=m["fp"], threshold=th[q]))
                lines.append(f"{amp:8g} {'Q%g' % q:>5s} {res[(q, amp)]:7.4f} {au:6.4f}  {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]"
                             f"{'*' if d[2] < 0 or d[1] > 0 else ' '}{'':5s} {m['sens']:5.3f} {m['spec']:5.3f} {m['fp']:4d}")
                if amp == 0 and (task, q) in EXPECTED:
                    et, ea = EXPECTED[(task, q)]
                    okq = abs(th[q] - et) < 1e-3 and abs(au - ea) < 1e-3
                    lines.append(f"   SELF-CHECK Q{q:g} at mains 0 = device_chain validate: {'PASS' if okq else 'FAIL'}")
                print(f"{task} mains {amp} Q{q:g} done", flush=True)
            lines.append("")

    df = pd.DataFrame(rows); df.to_csv(batch / "mains_test_summary.csv", index=False)
    txt = "\n".join(lines); print(txt); (batch / "mains_test_summary.txt").write_text(txt + "\n")
    plot(df, batch / "mains_test.png")
    print(f"saved {batch / 'mains_test_summary.txt'}, .csv and mains_test.png")


def plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    amps = sorted(df.mains_mv.unique()); xpos = {v: k for k, v in enumerate(amps)}
    for j, task in enumerate(("binary", "af")):
        for q, g in df[df.task == task].groupby("q"):
            x = [xpos[v] for v in g.mains_mv]
            ax[0, j].errorbar(x, g.d, yerr=[g.d - g.lo, g.hi - g.d], marker="o", capsize=3, label=f"Q{q:g}")
            ax[1, j].plot(x, g.spec, marker="o", label=f"Q{q:g} spec"); ax[1, j].plot(x, g.sens, ls="--", marker="x")
        ax[0, j].axhline(0, color="k", lw=0.8); ax[0, j].set_title(f"{task}: dAUROC vs clean raw")
        ax[1, j].set_title(f"{task}: specificity (solid) / sensitivity (dashed) at clean thresholds")
        ax[1, j].set_xticks(range(len(amps))); ax[1, j].set_xticklabels([f"{v:g}" for v in amps])
        ax[1, j].set_xlabel("mains amplitude (mV peak)"); ax[0, j].legend(); ax[0, j].grid(alpha=0.3); ax[1, j].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qs", nargs="+", type=float, default=[30.0, 15.0, 10.0, 5.0])
    ap.add_argument("--amps", nargs="+", type=float, default=[0, 0.5, 1, 2, 5, 17])
    ap.add_argument("--fdev", type=float, default=0.05, help="max mains frequency offset from 50 Hz")
    ap.add_argument("--fdrift", type=float, default=0.02, help="slow mains frequency drift amplitude")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--batch", default=str(Path.home() / "dsp_test" / "batch"))
    main(ap.parse_args())
