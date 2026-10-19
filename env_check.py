"""
Step 1a: verify the Python environment and the PTB-XL download before any training.

Usage:
    python env_check.py --root /path/to/ptb-xl
"""
import argparse
import importlib
import sys
from pathlib import Path

REQUIRED = ["numpy", "pandas", "scipy", "sklearn", "wfdb", "torch", "matplotlib"]
EXPECTED_FILES = ["ptbxl_database.csv", "scp_statements.csv", "records500"]


def check_libraries():
    ok = True
    print(f"Python {sys.version.split()[0]}")
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            print(f"  [ok]      {name:<11} {getattr(mod, '__version__', '?')}")
        except ImportError:
            print(f"  [MISSING] {name}")
            ok = False
    try:
        import torch
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        pass
    return ok


def check_dataset(root: Path):
    ok = True
    print(f"\nDataset root: {root}")
    for f in EXPECTED_FILES:
        present = (root / f).exists()
        ok &= present
        print(f"  [{'ok' if present else 'MISSING':<7}] {f}")
    if not ok:
        return False

    import pandas as pd
    db = pd.read_csv(root / "ptbxl_database.csv")
    n_hea = sum(1 for _ in (root / "records500").rglob("*.hea"))
    print(f"  records in metadata : {len(db)}")
    print(f"  patients            : {db['patient_id'].nunique()}")
    print(f"  500 Hz header files : {n_hea}")
    if n_hea != len(db):
        print("  WARNING: header count does not match metadata; download may be incomplete")
        ok = False
    print("  Record the PTB-XL version (from the download folder name) in your report.")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    args = ap.parse_args()
    libs_ok = check_libraries()
    data_ok = check_dataset(args.root)
    print("\nREADY" if (libs_ok and data_ok) else "\nNOT READY: fix the items above")
