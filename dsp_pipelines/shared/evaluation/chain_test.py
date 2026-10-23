"""
chain_test.py: which end-to-end chain should the device run in front of the models?

Chains (board filters are CAUSAL, steady state; team pipeline = bank_first_phase, 0.6 s median, A7 thresholded,
run in memory exactly as clean_dataset.py / noise_stress.py: flat tail trimmed, mirror padded, rounded, tail held):
  A   raw            -> team pipeline -> model
  B   notch          -> team pipeline -> model
  C   notch + LP40   -> team pipeline -> model
  D1  notch + LP40                    -> model            (step A winner, no DC handling)
  D2  notch + LP40 + 0.05 Hz high-pass (2nd order, causal) -> model
  D3  notch + LP40, each model window minus its own mean   -> model
  notch = 50 Hz Q30 iirnotch, LP40 = 2nd-order Butterworth 40 Hz: identical to the PYNQ script / hw_filter_test.py.

Conditions (fold 10): clean; dc = clean + 30 mV offset + 3 mV linear drift over the record (synthetic electrode
offset: PTB-XL has none, the device will); em 12, ma 12, mix 12, ma 6, mix 6 (noisy raw from noise_stress.py).
Thresholds: chosen on clean fold 9 processed by the same chain (team chains: flagged records excluded, as
recalibrate.py; D chains: all records, as hw_filter_test.py).
SELF-CHECKS: A clean = ns_orig_none_inf_med06.npz; A fold-9 thresholds = config_clean06.json;
D1 fold-9 thresholds = config_notch_lp40.json; window-mean scorer with demeaning off = score_batch.score;
fold-9 raw thresholds = config.json; clean raw = Stage 1.

Writes batch/chain_summary.txt, batch/chain_summary.csv, ~/export/chain_thresholds.json
Usage: sbatch chain_test.sbatch
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import noise_stress as ns         # noqa: E402  init_worker(), run_pipe(), finish()  (teammate's bank_first_phase)
import score_batch as sb          # noqa: E402
import recalibrate as rc          # noqa: E402

FS = 500
PRE = 1000
NOTCH = iirnotch(50.0, 30.0, FS)
LP = butter(2, 40.0, btype="lowpass", fs=FS)
HP005 = butter(2, 0.05, btype="highpass", fs=FS)
TEAM = {"A": [], "B": [NOTCH], "C": [NOTCH, LP]}
DIRECT = {"D1": [NOTCH, LP], "D2": [NOTCH, LP, HP005]}
CHAINS = ["A", "B", "C", "D1", "D2", "D3"]
DESC = {"A": "raw -> team", "B": "notch -> team", "C": "notch+LP40 -> team", "D1": "notch+LP40",
        "D2": "notch+LP40+HP0.05", "D3": "notch+LP40, window mean removed"}
TASKS = ["binary", "af"]


def causal(x, chain):
    y = np.r_[x[PRE:0:-1], x].astype(float)
    for b, a in chain:
        y, _ = lfilter(b, a, y, zi=lfilter_zi(b, a) * y[0])
    return y[PRE:]


def flat_run_at_end(x, tol=1e-9):          # same as clean_dataset.flat_run_at_end
    k = len(x) - 1
    while k > 0 and abs(x[k - 1] - x[-1]) <= tol:
        k -= 1
    return len(x) - k


def process(item):
    k, x, flat = item
    if np.isnan(x).any():
        return k, {}
    out = {n: causal(x, ch).astype(np.float32) for n, ch in DIRECT.items()}      # whole record, as hw_filter_test
    work = x[:len(x) - flat] if flat else x
    for n, ch in TEAM.items():
        c, _ = ns.run_pipe(causal(work, ch) if ch else work, "med06")
        out[n] = ns.finish(c, flat)
    return k, out


def run_chains(X, flats, code, workers, ptb):
    bufs = {n: np.full(X.shape, np.nan, np.float32) for n in ["A", "B", "C", "D1", "D2"]}
    items = [(k, X[k].astype(float), int(flats[k])) for k in range(len(X))]
    with Pool(workers, initializer=ns.init_worker, initargs=(code, {}, ptb)) as pool:
        for k, out in pool.imap_unordered(process, items, chunksize=4):
            for n, v in out.items():
                bufs[n][k] = v
    bufs["D3"] = bufs["D1"]
    return bufs


def score_wm(model, X, c, device, demean=True, bs=1024):
    """score_batch.score with optional per-window mean removal (before the fixed mu/sd normalisation)."""
    import torch
    win = int(round(c["win_sec"] * c["fs"]))
    T = X.shape[1]
    starts = list(range(0, T - win + 1, max(1, win // 2)))
    if starts[-1] != T - win:
        starts.append(T - win)
    out = np.full(len(X), np.nan)
    ok = ~np.isnan(X).any(axis=1)
    W = np.stack([X[ok][:, s:s + win] for s in starts], axis=1).astype(np.float32)
    if demean:
        W = W - W.mean(axis=-1, keepdims=True)
    W = ((W - c["mu"][0]) / c["sd"][0]).astype(np.float32).reshape(-1, 1, win)
    probs = []
    with torch.no_grad():
        for i in range(0, len(W), bs):
            probs.append(torch.sigmoid(model(torch.from_numpy(W[i:i + bs]).to(device))).float().cpu().numpy().ravel())
    out[ok] = np.concatenate(probs).reshape(-1, len(starts)).mean(axis=1)
    return out


def load_model(c, device):
    import torch
    m = sb.build_model(c["model"], 1, 1)
    m.load_state_dict(torch.load(sb.EXPORT / c["weights"], map_location="cpu"))
    return m.to(device).eval()


def main(a):
    batch = Path(a.batch)
    t0 = time.time()
    # ------------------------------------------------ inputs
    z = np.load(batch / "raw.npz"); ids10, Xc = z["ids"], z["X"].astype(np.float32)
    flats10 = np.array([(lambda f: f if f >= 20 else 0)(flat_run_at_end(x)) if not np.isnan(x).any() else 0 for x in Xc])
    n = Xc.shape[1]
    drift = (30.0 + 3.0 * np.arange(n) / n).astype(np.float32)
    inputs = {"clean": Xc, "dc": Xc + drift}
    for cnd in a.noise:
        nn, s = cnd.split()
        f = batch / f"ns_{nn}_{s}_raw.npz"
        if f.exists():
            zz = np.load(f); assert (zz["ids"] == ids10).all(); inputs[cnd] = zz["X"].astype(np.float32)
    z9 = np.load(batch / "ns_f9_raw.npz"); ids9, X9 = z9["ids"], z9["X"].astype(np.float32)
    flats9 = np.array([(lambda f: f if f >= 20 else 0)(flat_run_at_end(x)) if not np.isnan(x).any() else 0 for x in X9])

    # ------------------------------------------------ DSP (CPU, before any CUDA use)
    code = os.path.expanduser(a.code)
    S10 = {}
    for cnd, X in inputs.items():
        # dc condition: the clean flat-tail length is used (the drift would hide the flat tail)
        S10[cnd] = run_chains(X, flats10, code, a.workers, a.ptb)
        print(f"chains done: {cnd}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    S9 = run_chains(X9, flats9, code, a.workers, a.ptb)
    print(f"chains done: fold 9  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    lines = []
    ref = batch / "ns_orig_none_inf_med06.npz"
    if ref.exists():
        diff = np.abs(S10["clean"]["A"] - np.load(ref)["X"]); d = np.nanmax(diff)
        # float32 (raw.npz) vs float64 (WFDB read) input can flip a 0.001 mV rounding at a few samples
        lines.append(f"SELF-CHECK chain A clean vs ns_orig_none_inf_med06.npz: max |diff| {d:.5f} mV, "
                     f"{100 * np.nanmean(diff > 1e-6):.3f}% of samples differ -> {'PASS' if d <= 0.0011 else 'FAIL'}")

    # ------------------------------------------------ scoring
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    cfg = json.loads((sb.EXPORT / "config.json").read_text())
    ref_cfg = {"A": json.loads((sb.EXPORT / "config_clean06.json").read_text()),
               "D1": json.loads((sb.EXPORT / "config_notch_lp40.json").read_text())}
    db = pd.read_csv(sb.PTB / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(sb.PTB / "scp_statements.csv", index_col=0)
    log = pd.read_csv(Path(a.clean_root).expanduser() / "records500" / "clean_dataset_log.csv")
    flagged = set(db.index[db.filename_hr.isin(set("records500/" + log[log.note.notna()].record))])
    lab9, lab10 = rc.labels_for_fold(db, scp, 9), sb.build_labels()

    def P_of(model, c, df_X, ids_all, ids, chain):
        dfx = pd.DataFrame(df_X, index=ids_all).loc[ids].to_numpy(np.float32)
        if chain == "D3":
            return score_wm(model, dfx, c, device, demean=True)
        return sb.score(model, dfx, c, device)

    rows, new_th = [], {}
    for task in TASKS:
        c = cfg[task]
        model = load_model(c, device)
        lines.append(f"\n================ {task} ================")
        iv = lab9[task].index; y9 = lab9[task].to_numpy(); keep9 = ~iv.isin(flagged)
        p_raw9 = P_of(model, c, X9, ids9, iv, "raw")
        t_raw = (rc.thr_balanced(y9, p_raw9), rc.thr_sensitivity(y9, p_raw9, rc.HIGH_SENS_TARGET))
        lines.append(f"SELF-CHECK fold-9 raw thresholds = config.json: "
                     f"{'PASS' if abs(t_raw[0] - c['threshold_balanced']) < 1e-3 and abs(t_raw[1] - c['threshold_high_sens']) < 1e-3 else 'FAIL'}")
        chk = score_wm(model, X9[:200], c, device, demean=False) - sb.score(model, X9[:200], c, device)
        lines.append(f"SELF-CHECK window scorer (demean off) = score_batch.score: max |diff| {np.nanmax(np.abs(chk)):.2e} "
                     f"-> {'PASS' if np.nanmax(np.abs(chk)) < 1e-5 else 'FAIL'}")
        th = {"raw": t_raw}
        for ch in CHAINS:
            p = P_of(model, c, S9[ch], ids9, iv, ch)
            m = (keep9 if ch in TEAM else np.ones(len(iv), bool)) & ~np.isnan(p)
            th[ch] = (rc.thr_balanced(y9[m], p[m]), rc.thr_sensitivity(y9[m], p[m], rc.HIGH_SENS_TARGET))
        for ch, rcfg in ref_cfg.items():
            r_ = (rcfg[task]["threshold_balanced"], rcfg[task]["threshold_high_sens"])
            ok = abs(th[ch][0] - r_[0]) < 1e-3 and abs(th[ch][1] - r_[1]) < 1e-3
            lines.append(f"SELF-CHECK chain {ch} fold-9 thresholds {th[ch][0]:.4f} / {th[ch][1]:.4f} = "
                         f"{'config_clean06' if ch == 'A' else 'config_notch_lp40'} {r_[0]:.4f} / {r_[1]:.4f}: {'PASS' if ok else 'FAIL'}")
        lines.append("fold-9 thresholds (balanced / high_sens): " + ", ".join(f"{k} {v[0]:.4f} / {v[1]:.4f}" for k, v in th.items()))
        new_th[task] = {k: {"threshold_balanced": v[0], "threshold_high_sens": v[1], "chain": DESC.get(k, "raw")}
                        for k, v in th.items()}

        yt = lab10[task]; ids = yt.index[yt.index.isin(ids10)]; y = yt.loc[ids].to_numpy()
        P = {}
        for cnd in S10:
            P[(cnd, "raw")] = P_of(model, c, inputs[cnd], ids10, ids, "raw")
            for ch in CHAINS:
                P[(cnd, ch)] = P_of(model, c, S10[cnd][ch], ids10, ids, ch)
        e = sb.EXPECTED[task]; m0 = sb.at_threshold(y, P[("clean", "raw")], c["threshold_balanced"])
        ok0 = len(y) == e["n"] and int(y.sum()) == e["pos"] and all(m0[q] == e[q] for q in ("tp", "fn", "fp", "tn"))
        lines.append(f"SELF-CHECK clean raw = Stage 1: {'PASS' if ok0 else 'FAIL ' + str(m0)}")
        p_ref = P[("clean", "raw")]
        lines.append(f"{'condition':9s} {'chain':34s} {'AUROC':>6s}  {'dAUROC vs CLEAN raw [95% CI]':32s} "
                     f"{'dAUROC vs raw same cond.':>24s}  {'sens':>5s} {'spec':>5s} {'FP':>4s}")
        for cnd in S10:
            for ch in ["raw"] + CHAINS:
                p = P[(cnd, ch)]; ok = ~np.isnan(p)
                r = dict(task=task, condition=cnd, chain=ch, n=int(ok.sum()), auroc=roc_auc_score(y[ok], p[ok]))
                if not (cnd == "clean" and ch == "raw"):
                    r["d_clean"], r["d_clean_lo"], r["d_clean_hi"] = sb.paired_delta_auroc(y[ok], p_ref[ok], p[ok], a.n_boot)
                if ch != "raw":
                    r["d_same"] = r["auroc"] - roc_auc_score(y[ok], P[(cnd, "raw")][ok])
                t = th[ch]
                mm = sb.at_threshold(y[ok], p[ok], t[0]); hs = sb.at_threshold(y[ok], p[ok], t[1])
                r.update(sens=mm["sens"], spec=mm["spec"], fp=mm["fp"], hs_sens=hs["sens"], hs_spec=hs["spec"])
                rows.append(r)
                dtxt = f"{r['d_clean']:+.4f} [{r['d_clean_lo']:+.4f}, {r['d_clean_hi']:+.4f}]" if "d_clean" in r else ""
                stxt = f"{r['d_same']:+.4f}" if "d_same" in r else ""
                lines.append(f"{cnd:9s} {(ch + '  ' + DESC.get(ch, 'raw, no processing')):34s} {r['auroc']:6.4f}  "
                             f"{dtxt:32s} {stxt:>24s}  {mm['sens']:5.3f} {mm['spec']:5.3f} {mm['fp']:4d}")
            lines.append("")

    pd.DataFrame(rows).to_csv(batch / "chain_summary.csv", index=False)
    (sb.EXPORT / "chain_thresholds.json").write_text(json.dumps(new_th, indent=2))
    txt = "\n".join(lines)
    print(txt)
    (batch / "chain_summary.txt").write_text(txt + "\n")
    print(f"\nsaved chain_summary.txt/.csv in {batch}, thresholds per chain in {sb.EXPORT / 'chain_thresholds.json'} "
          f"({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="~/test_cleaning", help="teammate's pipeline (original detector, as team06)")
    ap.add_argument("--batch", default=str(Path.home() / "dsp_test" / "batch"))
    ap.add_argument("--clean_root", default=str(Path.home() / "data" / "ptb-xl-clean06"))
    ap.add_argument("--ptb", default=str(Path.home() / "data" / "ptb-xl"))
    ap.add_argument("--noise", nargs="+", default=["em 12", "ma 12", "mix 12", "ma 6", "mix 6"])
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--n_boot", type=int, default=500)
    main(ap.parse_args())
