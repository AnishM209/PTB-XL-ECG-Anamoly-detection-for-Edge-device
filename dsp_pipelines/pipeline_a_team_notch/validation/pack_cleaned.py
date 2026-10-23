"""
pack_cleaned.py: takes the records written by the DSP teammate's clean_dataset.py (single lead "V2-V1",
same folder structure and names as PTB-XL), keeps the fold-10 test records in the same order as
batch/raw.npz, saves them as batch/<name>.npz for score_batch.py, and measures T-wave retention
against raw with the same method as batch_clean.py.

Usage:
  python pack_cleaned.py --clean_root ~/data/ptb-xl-clean06 --name team06 --workers 16
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


def work(item):
    ecg_id, rel, raw, clean_root = item
    p = Path(clean_root) / rel
    try:
        sig, f = wfdb.rdsamp(str(p))
        x = sig[:, 0].astype(np.float32)
        err = ""
    except Exception as e:
        return ecg_id, np.full(len(raw), np.nan, np.float32), np.nan, np.nan, repr(e)[:200]
    if len(x) != len(raw):
        x = np.pad(x, (0, max(0, len(raw) - len(x))), constant_values=np.nan)[: len(raw)]
        err = "length differs"
    # beat positions from raw, exactly as batch_clean.py
    _, info = clean(raw.astype(float), FS, median_mode="single", median_win=0.3, a7_included=True)
    r_ref = refine_r(raw, info["r_peaks"], FS)
    t_raw = beat_amplitudes(raw.astype(float), r_ref, FS)[0]
    t_cln = beat_amplitudes(x.astype(float), r_ref, FS)[0] if not np.isnan(x).any() else np.nan
    return ecg_id, x, t_raw, t_cln, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_root", required=True, help="output folder given to clean_dataset.py (contains records500/)")
    ap.add_argument("--name", default="team06")
    ap.add_argument("--batch", default=str(Path.home() / "dsp_test" / "batch"))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    a = ap.parse_args()

    batch = Path(a.batch)
    z = np.load(batch / "raw.npz")
    ids, Xraw = z["ids"], z["X"]
    db = pd.read_csv(PTB / "ptbxl_database.csv", index_col="ecg_id")
    # clean_dataset.py mirrors records500/<folder>/<name>; filename_hr is "records500/<folder>/<name>"
    root = Path(a.clean_root).expanduser()
    rels = [db.loc[int(i), "filename_hr"] for i in ids]
    if not (root / (rels[0] + ".hea")).exists() and (root / Path(rels[0]).relative_to("records500")).with_suffix(".hea").exists():
        rels = [str(Path(r).relative_to("records500")) for r in rels]   # clean_root points at records500 itself
    items = [(int(i), r, Xraw[k], str(root)) for k, (i, r) in enumerate(zip(ids, rels))]

    with Pool(a.workers) as pool:
        res = pool.map(work, items, chunksize=8)
    X = np.stack([r[1] for r in res])
    np.savez(batch / f"{a.name}.npz", ids=ids, X=X)

    t_raw = np.array([r[2] for r in res]); t_cln = np.array([r[3] for r in res])
    errs = [r for r in res if r[4]]
    ok = (t_raw >= 0.05) & ~np.isnan(t_cln)
    ret = t_cln[ok] / t_raw[ok]
    msg = (f"{a.name}: {len(ids)} records, {len(errs)} read problems | T retention over {ok.sum()} records: "
           f"median {np.median(ret):.2f}, 10th pct {np.quantile(ret, 0.10):.2f}, below 0.7 in {100 * (ret < 0.7).mean():.1f}%")
    print(msg)
    for r in errs[:10]:
        print("  ", r[0], r[4])
    with open(batch / "retention_summary.txt", "a") as f:
        f.write(msg + "\n")
    print(f"saved {batch / (a.name + '.npz')}")


if __name__ == "__main__":
    main()
