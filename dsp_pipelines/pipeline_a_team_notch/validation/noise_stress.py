"""
noise_stress.py: MIT-BIH NSTDB noise stress test of the DSP teammate's current pipeline (bank_pipeline.py),
run in memory exactly like clean_dataset.py's clean_record(), with noise injected after the flat-tail trim.

Per record (fold 10 in batch/raw.npz order, or all of fold 9 with --fold 9):
  x = V2 - V1 (read with clean_dataset.read_leads_mv) -> trim flat tail -> work
  noisy = work + a * noise,  noise = random 10 s NSTDB segment (bw, em, ma, or mix = bw+em+ma at equal power)
          resampled 360 -> 500 Hz. Same segment for every SNR and pipeline (paired comparison).
  SNR (NSTDB nst convention): S = p2p^2 / 8, p2p = median per-second peak-to-peak of the 0.5 Hz high-passed
          clean signal (detector-free, so identical for every code version), N = mean square of the segment.
  pipelines (all with A7 thresholded, i.e. A7_included=False):
      med06   0.6 s median (teammate default)        med03   0.3 s median (paper value)
      hp05zp  0.5 Hz Butterworth order 2, filtfilt (zero phase, as dsp_chain.highpass; not realisable on device)
      hp05c   0.5 Hz Butterworth order 2, lfilter (causal, what a streaming device gets)
      raw     no cleaning (noisy raw), only with the default (no --no_raw)
  output = clean_dataset post-processing: hold last value over the flat tail, clip +-32.7 mV, round to 0.001 mV.

Saved to batch/ for score_batch.py (ids, X):
  fold 10: ns_<noise>_<snr>_raw.npz, ns_<tag>_<noise>_<snr>_<pipe>.npz   (noise "none", snr "inf" = no noise)
  fold 9 : ns_f9_raw.npz, ns_f9_<tag>_<pipe>.npz                         (no noise, for re-choosing thresholds)
Metrics (fold 10): batch/ns_<tag>_metrics.csv (per record) and ns_<tag>_summary.csv
  t_ret     T amplitude of the median beat template / same on the CLEAN raw record (beats from a fixed reference)
  qrs_ret   same for QRS
  snr_out   10 log10( |c0|^2 / |cn - c0|^2 ), c0 = pipeline on the clean record, cn = on the noisy one
  r_sens / r_ppv  pipeline's own R peaks vs reference beats, +-75 ms (both refined on the clean signal)
  locked    pipeline found < 2 beats (detector lock-out)

Usage:
  python noise_stress.py --code ~/test_cleaning --tag orig --workers 32
  python noise_stress.py --code ~/test_cleaning_reset --tag reset --no_raw --workers 32
  python noise_stress.py --code ~/test_cleaning --tag orig --fold 9 --pipes med03 hp05zp hp05c --workers 32
  quick test: add --limit 20
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, lfilter, lfilter_zi, resample_poly

HOME = Path.home()
FS = 500
NLEN = 5000            # PTB-XL 500 Hz records are 10 s
FLAT_MIN = 20          # as clean_dataset.py
EDGE_PAD = 512         # as clean_dataset.py
PIPES = ["med06", "med03", "hp05zp", "hp05c"]
NOISES = ["bw", "em", "ma", "mix"]
SNRS = [24, 18, 12, 6, 0]
_G = {}                # per-worker state


# ----------------------------------------------------------------------------- filters / measurement
_HP = butter(2, 0.5 / (0.5 * FS), btype="highpass")
_BP = butter(2, [5 / (0.5 * FS), 25 / (0.5 * FS)], btype="bandpass")


def hp_zero(x, fs=FS, _w=None):
    return filtfilt(*_HP, x)


def hp_causal(x, fs=FS, _w=None):
    b, a = _HP
    y, _ = lfilter(b, a, x, zi=lfilter_zi(b, a) * x[0])
    return y


def s(ms):
    return int(round(ms * 1e-3 * FS))


def refine_r(sig, r, half_ms=40):
    """Copied from dsp_chain.refine_r (dsp_chain is not imported: it puts the OLD repo first on sys.path)."""
    h = s(half_ms)
    out = [max(0, k - h) + int(np.argmax(np.abs(sig[max(0, k - h):min(len(sig), k + h + 1)]))) for k in r]
    return np.unique(np.asarray(out, dtype=int))


def ref_beats(x):
    """Detector-free reference beats on the CLEAN record (same for every code version and pipeline)."""
    y = hp_zero(x)
    e = np.abs(filtfilt(*_BP, y))
    pk, _ = find_peaks(e, height=0.35 * np.quantile(e, 0.99), distance=s(250))
    return y, refine_r(y, pk)


def signal_power(y):
    """nst convention S = p2p^2/8, p2p = median of per-second peak-to-peak of the high-passed clean signal."""
    p2p = [np.ptp(y[i:i + FS]) for i in range(0, len(y) - FS + 1, FS)]
    return float(np.median(p2p)) ** 2 / 8


def template_amp(sig, r):
    """|T| and |QRS| of the median beat (window R-250..R+600 ms), relative to iso = median(R-100..R-60 ms)."""
    r = r[(r - s(250) >= 0) & (r + s(600) <= len(sig))]
    if len(r) < 3 or np.isnan(sig).any():
        return np.nan, np.nan
    tpl = np.median(np.stack([sig[k - s(250):k + s(600)] for k in r]), axis=0)
    c = s(250)
    iso = np.median(tpl[c - s(100):c - s(60)])
    qrs = float(np.max(np.abs(tpl[c - s(50):c + s(50)] - iso)))
    rr = int(np.median(np.diff(r)))
    t_end = min(c + s(450), c + rr - s(100), len(tpl))
    if t_end - (c + s(150)) < s(50):
        return np.nan, qrs
    seg = tpl[c + s(150):t_end] - iso
    return float(np.abs(seg).max()), qrs


def match(ref, det, tol=s(75)):
    if len(ref) == 0:
        return np.nan, np.nan
    if len(det) == 0:
        return 0.0, np.nan
    d = np.abs(ref[:, None] - det[None, :])
    return float((d.min(1) <= tol).mean()), float((d.min(0) <= tol).mean())


def snr_out(c0, cn):
    sl = slice(s(500), len(c0) - s(500))          # ignore 0.5 s at each end
    a = c0[sl] - c0[sl].mean(); e = (cn - c0)[sl]; e = e - e.mean()
    return float(10 * np.log10(np.sum(a ** 2) / max(np.sum(e ** 2), 1e-12)))


# ----------------------------------------------------------------------------- worker
def init_worker(code_dir, noise, ptb):
    sys.path.insert(0, code_dir)
    import bank_pipeline
    import clean_dataset
    _G.update(bp=bank_pipeline, cd=clean_dataset, med=bank_pipeline.median_filter, noise=noise, ptb=ptb)


def run_pipe(work, pipe):
    bp = _G["bp"]
    bp.median_filter = {"med06": _G["med"], "med03": _G["med"], "hp05zp": hp_zero, "hp05c": hp_causal}[pipe]
    mw = 0.3 if pipe == "med03" else 0.6          # median window is ignored by the high-pass variants
    n = len(work)
    ext = np.r_[work[EDGE_PAD:0:-1], work, work[-2:-EDGE_PAD - 2:-1]]
    out = bp.bank_first_phase(ext, FS, warmup=False, median_window=mw, A7_included=False)
    c = out["clean"][EDGE_PAD:EDGE_PAD + n]
    pk = np.asarray(out["r_peaks"], int) - EDGE_PAD
    return c, pk[(pk >= 0) & (pk < n)]


def finish(c, flat, round_clip=True, tail=None):
    """Cleaned: hold the last cleaned value over the flat tail (as clean_dataset.py).
    Raw: append the record's own flat tail (tail), so no-noise raw equals the original record exactly."""
    if flat:
        c = np.r_[c, tail if tail is not None else np.full(flat, c[-1])]
    else:
        c = c.copy()
    if round_clip:
        c = np.round(np.clip(c, -32.7, 32.7), 3)
    out = np.full(NLEN, np.nan, np.float32)
    out[:min(NLEN, len(c))] = c[:NLEN]
    return out


def noise_seg(kind, n, rng):
    if kind == "mix":
        parts = [noise_seg(k, n, rng) for k in ("bw", "em", "ma")]
        return sum(p / np.sqrt(np.mean(p ** 2)) for p in parts)
    arr = _G["noise"][kind]
    ch, st = rng.integers(arr.shape[1]), rng.integers(0, arr.shape[0] - n)
    seg = arr[st:st + n, ch].astype(float)
    return seg - seg.mean()


def work_record(item):
    k, ecg_id, rel, conds, pipes, seed, metrics = item
    outs, rows = {}, []
    try:
        leads, _ = _G["cd"].read_leads_mv(str(Path(_G["ptb"]) / (rel + ".hea")))
        x = leads["V2"] - leads["V1"]
        flat = _G["cd"].flat_run_at_end(x)
        flat = flat if flat >= FLAT_MIN else 0
        work = x[:len(x) - flat] if flat else x
        if len(work) < 3 * FS:
            return k, outs, [dict(ecg_id=ecg_id, status="short")]
        n = len(work)
        if metrics:
            y, r_ref = ref_beats(work)
            S = signal_power(y)
            t0, q0 = template_amp(work, r_ref)
        noisy = {}
        for kind, snr in conds:
            if kind == "none":
                noisy[(kind, snr)] = work
                continue
            rng = np.random.default_rng([seed, int(ecg_id), NOISES.index(kind)])
            seg = noise_seg(kind, n, rng)
            S_ = S if metrics else signal_power(hp_zero(work))
            noisy[(kind, snr)] = work + np.sqrt(S_ / (np.mean(seg ** 2) * 10 ** (int(snr) / 10))) * seg
        for p in pipes:
            c0 = None
            for kind, snr in conds:
                xin = noisy[(kind, snr)]
                if p == "raw":
                    c, pk = xin, None
                else:
                    c, pk = run_pipe(xin, p)
                    if np.isnan(c).any():
                        raise RuntimeError(f"NaN output ({p}, {kind}, {snr})")
                outs[(kind, snr, p)] = finish(c, flat, round_clip=(p != "raw"),
                                              tail=x[len(x) - flat:] if p == "raw" else None)
                if kind == "none":
                    c0 = c
                if metrics:
                    t, q = template_amp(c, r_ref)
                    rs, rp = match(r_ref, refine_r(y, pk)) if pk is not None else (np.nan, np.nan)
                    rows.append(dict(ecg_id=ecg_id, noise=kind, snr=snr, pipe=p, status="ok",
                                     t_ret=t / t0 if t0 and t0 >= 0.05 else np.nan,
                                     qrs_ret=q / q0 if q0 else np.nan,
                                     snr_out=snr_out(c0, c) if kind != "none" else np.nan,
                                     r_sens=rs, r_ppv=rp, n_ref=len(r_ref),
                                     n_det=len(pk) if pk is not None else np.nan,
                                     locked=(len(pk) < 2) if pk is not None else np.nan))
    except Exception as e:
        return k, {}, [dict(ecg_id=ecg_id, status=f"error {type(e).__name__}: {e}"[:200])]
    return k, outs, rows


# ----------------------------------------------------------------------------- main
def load_noise(nst):
    import wfdb
    out = {}
    for kind in ("bw", "em", "ma"):
        rec = wfdb.rdrecord(str(Path(nst) / kind))
        assert int(rec.fs) == 360, f"{kind}: fs {rec.fs}"
        out[kind] = resample_poly(rec.p_signal, 25, 18, axis=0).astype(np.float32)   # 360 -> 500 Hz
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, help="folder with the teammate's pipeline (bank_pipeline.py etc.)")
    ap.add_argument("--tag", required=True, help="name for this code version, e.g. orig / reset")
    ap.add_argument("--fold", type=int, default=10, choices=[9, 10])
    ap.add_argument("--pipes", nargs="+", default=PIPES, choices=PIPES)
    ap.add_argument("--noises", nargs="+", default=NOISES, choices=NOISES)
    ap.add_argument("--snrs", nargs="+", type=int, default=SNRS)
    ap.add_argument("--no_raw", action="store_true", help="do not save noisy raw (already saved by another run)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--ptb", default=str(HOME / "data" / "ptb-xl"))
    ap.add_argument("--nstdb", default=str(HOME / "data" / "nstdb"))
    ap.add_argument("--batch", default=str(HOME / "dsp_test" / "batch"))
    a = ap.parse_args()

    batch = Path(a.batch)
    db = pd.read_csv(Path(a.ptb) / "ptbxl_database.csv", index_col="ecg_id")
    if a.fold == 10:
        ids = np.load(batch / "raw.npz")["ids"]
        conds = [("none", "inf")] + [(n, str(v)) for n in a.noises for v in a.snrs]
    else:
        ids = db.index[db["strat_fold"] == 9].values
        conds = [("none", "inf")]
    ids = np.asarray(ids)[:a.limit]
    pipes = list(a.pipes) + ([] if a.no_raw else ["raw"])
    metrics = a.fold == 10
    noise = load_noise(a.nstdb) if a.fold == 10 else {}

    def fname(kind, snr, p):
        if a.fold == 9:
            return "ns_f9_raw" if p == "raw" else f"ns_f9_{a.tag}_{p}"
        return f"ns_{kind}_{snr}_raw" if p == "raw" else f"ns_{a.tag}_{kind}_{snr}_{p}"

    bufs = {(kd, sn, p): np.full((len(ids), NLEN), np.nan, np.float32) for kd, sn in conds for p in pipes}
    print(f"{len(ids)} records, {len(conds)} conditions x {len(pipes)} pipelines = {len(bufs)} outputs, "
          f"code {a.code} (tag {a.tag}), {a.workers} workers", flush=True)
    items = [(k, int(i), db.loc[int(i), "filename_hr"], conds, pipes, a.seed, metrics) for k, i in enumerate(ids)]
    rows, t0 = [], time.time()
    with Pool(a.workers, initializer=init_worker, initargs=(os.path.expanduser(a.code), noise, a.ptb)) as pool:
        for j, (k, outs, r) in enumerate(pool.imap_unordered(work_record, items, chunksize=2), 1):
            for key, v in outs.items():
                bufs[key][k] = v
            rows += r
            if j % 100 == 0 or j == len(items):
                el = time.time() - t0
                print(f"  {j}/{len(items)}  {el / 60:.1f} min, ~{el / j * (len(items) - j) / 60:.1f} min left", flush=True)

    for key, X in bufs.items():
        np.savez(batch / f"{fname(*key)}.npz", ids=ids, X=X)
    bad = [r for r in rows if r.get("status") != "ok"]
    print(f"saved {len(bufs)} npz files to {batch} | {len(bad)} records skipped/failed")
    for r in bad[:10]:
        print("  ", r)

    if not metrics:
        return
    df = pd.DataFrame([r for r in rows if r.get("status") == "ok"])
    df.to_csv(batch / f"ns_{a.tag}_metrics.csv", index=False)
    g = df.groupby(["noise", "snr", "pipe"])
    summ = pd.DataFrame({
        "n": g["ecg_id"].size(),
        "t_ret_med": g["t_ret"].median(), "t_ret_p10": g["t_ret"].quantile(0.10),
        "t_below70_pct": g["t_ret"].apply(lambda v: 100 * (v.dropna() < 0.7).mean()),
        "qrs_ret_med": g["qrs_ret"].median(), "snr_out_med": g["snr_out"].median(),
        "r_sens": g["r_sens"].mean(), "r_ppv": g["r_ppv"].mean(), "locked": g["locked"].sum(),
    }).reset_index()
    summ.to_csv(batch / f"ns_{a.tag}_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500, "display.precision", 3):
        print(summ)

    # self-check: no-noise med06 must reproduce team06.npz (clean_dataset.py output), no-noise raw = raw.npz
    for key, ref in [(("none", "inf", "med06"), "team06.npz"), (("none", "inf", "raw"), "raw.npz")]:
        if key in bufs and (batch / ref).exists():
            R = np.load(batch / ref)["X"][:len(ids)]
            diff = np.abs(bufs[key] - R)
            d = np.nanmax(diff)
            frac = 100 * np.nanmean(diff > 0.0011)
            # team06 went through a WFDB write/read (1000 units/mV): allow 2 units of rounding
            tol = 0.0021 if ref == "team06.npz" else 1e-6
            print(f"self-check {fname(*key)} vs {ref}: max |diff| {d:.5f} mV, {frac:.3f}% of samples > 1 unit "
                  f"-> {'PASS' if d <= tol else 'FAIL'}")


if __name__ == "__main__":
    main()
