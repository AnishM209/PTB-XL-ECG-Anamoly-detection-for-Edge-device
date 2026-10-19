"""
Step 1b: build the signal cache and inspect label distributions for all three
candidate tasks. The CSVs it writes are the evidence for choosing the task.

Usage:
    python inspect_dataset.py --root /path/to/ptb-xl --out ./stage1_out [--skip-cache]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import ptbxl_data as P


def prevalence_by_split(Y, splits):
    """Positive count and percentage per column, per split."""
    rows = {}
    for name, ids in splits.items():
        sub = Y.loc[Y.index.intersection(ids)].astype(np.float64)
        rows[(name, "n")] = sub.sum().astype(int)
        rows[(name, "%")] = (100 * sub.mean()).round(1)
    return pd.DataFrame(rows)


def counts_by_split(y, splits):
    """Category counts per split for a single-label Series (NaN shown as 'excluded')."""
    out = {name: y.loc[y.index.intersection(ids)].fillna("excluded").value_counts().rename_axis(None)
           for name, ids in splits.items()}
    return pd.DataFrame(out).fillna(0).astype(int)


def main(root, out, skip_cache, workers):
    out.mkdir(parents=True, exist_ok=True)
    db, scp = P.load_metadata(root)
    splits = P.split_ids(db)

    # --- split integrity: no patient may appear in two splits
    pats = {k: set(db.loc[v, "patient_id"]) for k, v in splits.items()}
    leaks = {f"{a}-{b}": len(pats[a] & pats[b])
             for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    print("Records per split:", {k: len(v) for k, v in splits.items()})
    print("Patient overlap between splits (must be 0):", leaks)

    # --- task A: five superclasses (multi-label)
    Y_sup = P.superclass_labels(db, scp)
    sup_tab = prevalence_by_split(Y_sup, splits)
    n_labels = Y_sup.sum(axis=1).value_counts().sort_index()
    print("\nSuperclass prevalence:\n", sup_tab)
    print("Labels per record (0 = no diagnostic statement):\n", n_labels.to_string())
    sup_tab.to_csv(out / "task_superclass.csv")

    # --- task B: binary normal vs abnormal
    y_bin = P.binary_labels(Y_sup).map({0.0: "normal", 1.0: "abnormal"})
    bin_tab = counts_by_split(y_bin, splits)
    print("\nBinary task:\n", bin_tab)
    bin_tab.to_csv(out / "task_binary.csv")

    # --- task C: rhythm (raw codes and proposed grouping)
    R_raw, r_grp = P.rhythm_labels(db, scp)
    raw_tab = prevalence_by_split(R_raw, splits).sort_values(("train", "n"), ascending=False)
    grp_tab = counts_by_split(r_grp, splits)
    multi = int(((R_raw.sum(axis=1) > 1)).sum())
    print("\nRaw rhythm codes:\n", raw_tab)
    print("\nGrouped rhythm task:\n", grp_tab)
    print(f"Records with more than one rhythm code: {multi}")
    raw_tab.to_csv(out / "task_rhythm_raw.csv")
    grp_tab.to_csv(out / "task_rhythm_grouped.csv")

    # --- recording-quality flags (relevant later for noise robustness)
    q = pd.Series({c: int(db[c].notna().sum()) for c in P.QUALITY_COLS if c in db})
    print("\nRecords with a quality annotation:\n", q.to_string())
    q.to_csv(out / "quality_flags.csv", header=["n_records"])

    # --- signal cache and sanity statistics
    if not skip_cache:
        print("\nBuilding signal cache (one-time)...")
        P.build_signal_cache(db, root, out / "cache", workers=workers)
        X, ids = P.load_signal_cache(out / "cache")
        assert (ids == db.index.to_numpy()).all(), "cache order does not match metadata"
        rng = np.random.default_rng(0)
        sample = np.asarray(X[rng.choice(len(X), size=min(500, len(X)), replace=False)])
        stats = pd.DataFrame({
            "nan_frac": np.isnan(sample).mean(axis=(0, 2)),
            "p1_mV": np.nanpercentile(sample, 1, axis=(0, 2)),
            "p99_mV": np.nanpercentile(sample, 99, axis=(0, 2)),
            "flat_records": (np.nanstd(sample, axis=2) < 1e-3).sum(axis=0),
        }, index=P.LEADS).round(3)
        print("\nPer-lead signal sanity (500-record sample):\n", stats)
        stats.to_csv(out / "signal_sanity.csv")

    print(f"\nSummaries written to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", default=Path("stage1_out"), type=Path)
    ap.add_argument("--skip-cache", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    main(a.root, a.out, a.skip_cache, a.workers)
