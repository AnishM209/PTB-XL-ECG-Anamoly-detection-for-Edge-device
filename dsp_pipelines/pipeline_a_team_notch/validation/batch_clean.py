"""
batch_clean.py: Part A of the batch DSP test.
Cleans every PTB-XL fold-10 record (V2-V1) with each DSP variant, saves the signals for scoring,
and measures per-record T-wave retention against raw.

Outputs in --out (default ~/dsp_test/batch):
  raw.npz, teammate.npz, hp05_noA7.npz   arrays: ids (n,), X (n, 5000) float32 in mV (NaN rows = failed)
  beat_stats.csv                          one row per record and signal: T, QRS, T/QRS, beats, error
  retention_summary.txt                   summary printed at the end

Smoke test on the login node:  python batch_clean.py --limit 20 --workers 4
Full run:                      sbatch batch_clean.sbatch
"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from dsp_chain import beat_amplitudes, clean, refine_r

PTB = Path.home() / "data" / "ptb-xl"
FS = 500
N = 5000
VARIANTS = {
    "teammate":  dict(median_mode="single", median_win=0.3, a7_included=True),
    "hp05_noA7": dict(median_mode="hp05", a7_included=False),
}
SIGNALS = ["raw"] + list(VARIANTS)


def process(item):
    ecg_id, rel = item
    sig, f = wfdb.rdsamp(str(PTB / rel))
    names = [n.upper() for n in f["sig_name"]]
    raw = sig[:, names.index("V2")] - sig[:, names.index("V1")]

    sigs = {"raw": raw.astype(np.float32)}
    errors = {"raw": ""}
    r_ref = None
    for name, kw in VARIANTS.items():
        try:
            cleaned, info = clean(raw, FS, **kw)
            sigs[name] = cleaned.astype(np.float32)
            errors[name] = ""
            if r_ref is None:
                r_ref = refine_r(raw, info["r_peaks"], FS)
        except Exception as e:  # keep going, record the failure
            sigs[name] = np.full(len(raw), np.nan, np.float32)
            errors[name] = repr(e)[:200]

    rows = []
    for name in SIGNALS:
        if r_ref is None or errors[name]:
            t = q = rq = np.nan
            nb = 0
        else:
            t, q, rq, nb = beat_amplitudes(sigs[name].astype(float), r_ref, FS)
        rows.append(dict(ecg_id=ecg_id, signal=name, t_mv=t, qrs_mv=q, t_qrs=rq, beats=nb,
                         n_rpeaks=0 if r_ref is None else len(r_ref), error=errors[name]))
    return ecg_id, sigs, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "dsp_test" / "batch"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N test records (smoke test)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    db = pd.read_csv(PTB / "ptbxl_database.csv", index_col="ecg_id")
    test = db[db.strat_fold == 10].sort_index()
    if args.limit:
        test = test.iloc[: args.limit]
    items = list(zip(test.index.astype(int), test.filename_hr))
    print(f"{len(items)} test-fold records, {args.workers} workers", flush=True)

    ids, X, all_rows = [], {s: [] for s in SIGNALS}, []
    with Pool(args.workers) as pool:
        for k, (ecg_id, sigs, rows) in enumerate(pool.imap(process, items, chunksize=8), 1):
            ids.append(ecg_id)
            for s in SIGNALS:
                x = sigs[s]
                if len(x) != N:
                    x = np.pad(x, (0, max(0, N - len(x))), constant_values=np.nan)[:N]
                X[s].append(x)
            all_rows.extend(rows)
            if k % 200 == 0:
                print(f"  {k}/{len(items)}", flush=True)

    ids = np.asarray(ids)
    for s in SIGNALS:
        np.savez(out / f"{s}.npz", ids=ids, X=np.stack(X[s]))
    stats = pd.DataFrame(all_rows)
    stats.to_csv(out / "beat_stats.csv", index=False)

    # Retention summary: T(variant) / T(raw), records with a measurable raw T wave (>= 0.05 mV)
    piv = stats.pivot(index="ecg_id", columns="signal", values="t_mv")
    lines = [f"records: {len(ids)}"]
    for s in VARIANTS:
        n_fail = int((stats[(stats.signal == s)].error != "").sum())
        ok = piv[(piv["raw"] >= 0.05) & piv[s].notna()]
        ret = ok[s] / ok["raw"]
        lines.append(
            f"{s:10s} failures {n_fail:4d} | T retention over {len(ret)} records: "
            f"median {ret.median():.2f}, 10th pct {ret.quantile(0.10):.2f}, "
            f"below 0.7 in {100 * (ret < 0.7).mean():.1f}%"
        )
    summary = "\n".join(lines)
    print(summary)
    (out / "retention_summary.txt").write_text(summary + "\n")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
