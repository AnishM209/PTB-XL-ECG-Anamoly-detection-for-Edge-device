"""
PTB-XL data preparation for Stage 1 benchmarking.

Three things live here:
  1. Labels for all three candidate tasks (superclass, binary, rhythm), so the
     task decision does not block the data work.
  2. A one-time signal cache: every 500 Hz record is read once and written to a
     single .npy memmap of shape (N, 12, 5000). Lead selection later is a slice,
     so the lead sweep never re-reads WFDB files.
  3. The official patient-wise split (strat_fold 1-8 train, 9 val, 10 test).
"""
import ast
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

FS = 500
N_SAMPLES = 5000  # 10 s at 500 Hz
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
FOLDS = {"train": list(range(1, 9)), "val": [9], "test": [10]}
QUALITY_COLS = ["baseline_drift", "static_noise", "burst_noise",
                "electrodes_problems", "extra_beats", "pacemaker"]

# Proposed grouping of PTB-XL rhythm codes into a small single-label task.
# Rare rhythms are pooled into OTHER; edit this after inspecting the counts.
RHYTHM_GROUPS = {
    "SR": "SR",
    "AFIB": "AF", "AFLT": "AF",
    "STACH": "STACH",
    "SBRAD": "SBRAD",
    "SARRH": "OTHER", "SVARR": "OTHER", "PACE": "OTHER", "BIGU": "OTHER",
    "SVTAC": "OTHER", "PSVT": "OTHER", "TRIGU": "OTHER",
}


# ----------------------------------------------------------------- metadata
def load_metadata(root):
    root = Path(root)
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    return db, scp


def split_ids(db):
    """ecg_ids per split, using the official patient-wise stratified folds."""
    return {name: db.index[db["strat_fold"].isin(f)].to_numpy() for name, f in FOLDS.items()}


# ------------------------------------------------------------------- labels
def superclass_labels(db, scp, min_likelihood=0.0):
    """Multi-hot (N, 5) over NORM, MI, STTC, CD, HYP.

    min_likelihood=0 keeps every diagnostic statement, which matches the
    published benchmark setup. Raise it to study label-confidence effects.
    """
    code_to_class = scp.loc[scp["diagnostic"] == 1, "diagnostic_class"].to_dict()
    Y = pd.DataFrame(0, index=db.index, columns=SUPERCLASSES, dtype=np.float32)
    for ecg_id, codes in db["scp_codes"].items():
        for code, lik in codes.items():
            if code in code_to_class and lik >= min_likelihood:
                Y.at[ecg_id, code_to_class[code]] = 1.0
    return Y


def binary_labels(Y_super):
    """0 = normal, 1 = abnormal, NaN = no diagnostic statement (exclude).

    Normal means NORM is the only superclass present. Any of MI, STTC, CD or
    HYP makes the record abnormal, even if NORM is also tagged.
    """
    abnormal = Y_super[["MI", "STTC", "CD", "HYP"]].sum(axis=1) > 0
    normal = (Y_super["NORM"] == 1) & ~abnormal
    y = pd.Series(np.nan, index=Y_super.index, name="abnormal")
    y[normal] = 0.0
    y[abnormal] = 1.0
    return y


def rhythm_labels(db, scp, groups=RHYTHM_GROUPS):
    """Returns (multi-hot over every raw rhythm code, grouped single label).

    The grouped label is NaN when a record has no rhythm statement or maps to
    more than one group; the inspection script reports how many that affects.
    """
    rhythm_codes = scp.index[scp["rhythm"] == 1].tolist()
    raw = pd.DataFrame(0, index=db.index, columns=rhythm_codes, dtype=np.float32)
    grouped = pd.Series(np.nan, index=db.index, dtype=object, name="rhythm_group")
    for ecg_id, codes in db["scp_codes"].items():
        present = [c for c in codes if c in rhythm_codes]
        for c in present:
            raw.at[ecg_id, c] = 1.0
        g = {groups.get(c, "OTHER") for c in present}
        if len(g) == 1:
            grouped[ecg_id] = g.pop()
    return raw, grouped


# ------------------------------------------------------------------ signals
def _read_one(args):
    root, rel_path = args
    import wfdb
    sig, fields = wfdb.rdsamp(str(Path(root) / rel_path))
    names = [n.upper() for n in fields["sig_name"]]
    order = [names.index(l) for l in LEADS]  # enforce a fixed lead order
    return sig[:, order].T.astype(np.float32)  # (12, 5000), units: mV


def build_signal_cache(db, root, out_dir, workers=8, reader=_read_one):
    """Reads every 500 Hz record once. Writes signals_500hz.npy and ecg_ids.npy."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = db.index.to_numpy()
    X = np.lib.format.open_memmap(out_dir / "signals_500hz.npy", mode="w+",
                                  dtype=np.float32, shape=(len(ids), 12, N_SAMPLES))
    jobs = [(str(root), p) for p in db["filename_hr"]]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, sig in enumerate(ex.map(reader, jobs, chunksize=64)):
            if sig.shape != (12, N_SAMPLES):
                raise ValueError(f"ecg_id {ids[i]}: unexpected shape {sig.shape}")
            X[i] = sig
            if (i + 1) % 2000 == 0:
                print(f"  cached {i + 1}/{len(ids)}")
    X.flush()
    np.save(out_dir / "ecg_ids.npy", ids)
    return out_dir


def load_signal_cache(cache_dir, leads=None):
    """Memory-mapped (N, n_leads, 5000) array plus aligned ecg_ids.

    leads: list such as ["II"] or ["I", "V2"]; None returns all 12.
    """
    cache_dir = Path(cache_dir)
    X = np.load(cache_dir / "signals_500hz.npy", mmap_mode="r")
    ids = np.load(cache_dir / "ecg_ids.npy")
    if leads is not None:
        X = X[:, [LEADS.index(l.upper()) for l in leads], :]
    return X, ids
