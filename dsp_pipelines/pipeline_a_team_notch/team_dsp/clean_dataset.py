"""
Clean a whole PTB-XL style dataset for CNN training.

For every WFDB record (<name>.hea + <name>.dat) found in the input folder (sub-folders included):
  1. read V1 and V2, form the 2-electrode signal  V2 - V1  (mV)
  2. cut off the constant (flat) stretch that PTB-XL records have at the end (usually the last ~0.1 s)
  3. run the First Phase pipeline exactly as planned for the hardware: median filter (0.6 s, A7 kept --
     settings in pipeline_config.py, or --median-window / --remove-A7) + LPF + continuous
     SWT (FPGA part), then the processor part bank by bank in 2048-sample (4.1 s) ping-pong banks
     (bank_pipeline.py)
  4. pad the cleaned signal back to the original length (holding the last value over the flat stretch)
     (each record is mirror-extended by ~1 s at both ends while it is processed, so its first and last
      samples are cleaned as if they were in the middle of a long recording -- no start/end transients)
  5. write it as a single-lead WFDB record (<name>.dat + <name>.hea, format 16, 1000 units/mV, same fs)

The output folder mirrors the input folder structure and keeps the SAME record names, so the dataset's
label file (e.g. ptbxl_database.csv, column filename_hr) still points to the right records -- only the
signal changes from 12 leads to one lead "V2-V1". A summary (clean_dataset_log.csv) lists every record.

Usage (from the folder with the pipeline .py files):
    python clean_dataset.py  <input_folder>  <output_folder>
    python clean_dataset.py  C:/data/ptb-xl/records500  C:/data/ptb-xl-clean/records500  --workers 4

Options:
    --workers N     process N records in parallel (default 1; use about the number of CPU cores)
    --limit N       only the first N records (quick test)
    --overwrite     re-process records that already exist in the output folder (default: skip them,
                    so an interrupted run can simply be started again)

Notes:
  * Only 500 Hz records are processed (the 100 Hz PTB-XL copies are skipped and logged): the pipeline's
    150 Hz low-pass filter and median filter are designed for 500 Hz.
  * The R-peak detector starts from the first 5 s of each record (as in the full-signal pipeline). This is
    the detector's steady-state behaviour on the device; its 5 s warm-up after power-on is a one-time
    event on the hardware and is not applied to each 10 s training record.
"""
import argparse
import csv
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FS_REQUIRED = 500
FLAT_MIN = 20          # a constant run of at least this many samples at the end is treated as padding
EDGE_PAD = 512         # samples mirrored onto each end of a record before processing (removed afterwards)


# ----------------------------------------------------------------------------- WFDB reading
def read_header(hea_path):
    """Parse a WFDB header (single segment). Returns dict with fs, n_samples and per-signal info."""
    lines = [l.strip() for l in open(hea_path) if l.strip() and not l.lstrip().startswith("#")]
    rec = lines[0].split()
    n_sig = int(rec[1])
    fs = float(rec[2].split("/")[0]) if len(rec) > 2 else 250.0
    n_samples = int(rec[3]) if len(rec) > 3 else None
    sigs = []
    for l in lines[1:1 + n_sig]:
        p = l.split()
        fmt = p[1].split("x")[0].split(":")[0].split("+")[0]
        g = p[2] if len(p) > 2 else "200"
        gain = float(g.split("(")[0].split("/")[0]) or 200.0
        base = int(g.split("(")[1].split(")")[0]) if "(" in g else (int(p[4]) if len(p) > 4 else 0)
        units = g.split("/")[1] if "/" in g else "mV"
        sigs.append(dict(file=p[0], fmt=fmt, gain=gain, baseline=base, units=units,
                         name=p[8] if len(p) > 8 else f"sig{len(sigs)}"))
    return dict(name=rec[0], n_sig=n_sig, fs=fs, n_samples=n_samples, signals=sigs)


def read_leads_mv(hea_path, leads=("V1", "V2")):
    """Return {lead: signal in mV} for the requested leads (format 16 only, as in PTB-XL)."""
    h = read_header(hea_path)
    folder = os.path.dirname(hea_path)
    names = [s["name"].upper() for s in h["signals"]]
    missing = [l for l in leads if l.upper() not in names]
    if missing:
        raise ValueError(f"lead(s) {missing} not in record (has {names})")
    files = {s["file"] for s in h["signals"]}
    if len(files) != 1 or any(s["fmt"] != "16" for s in h["signals"]):
        raise ValueError("only single-file format-16 records are supported (PTB-XL format)")
    data = np.fromfile(os.path.join(folder, h["signals"][0]["file"]), dtype="<i2")
    data = data[: (len(data) // h["n_sig"]) * h["n_sig"]].reshape(-1, h["n_sig"])
    out = {}
    for l in leads:
        i = names.index(l.upper()); s = h["signals"][i]
        scale = 1000.0 if s["units"].lower() in ("uv", "µv") else 1.0     # convert to mV
        out[l] = (data[:, i].astype(float) - s["baseline"]) / s["gain"] / scale
    return out, h


# ----------------------------------------------------------------------------- one record
def flat_run_at_end(x, tol=1e-9):
    k = len(x) - 1
    while k > 0 and abs(x[k - 1] - x[-1]) <= tol:
        k -= 1
    return len(x) - k


def clean_record(hea_path, in_root, out_root, overwrite=False, median_window=None, A7_included=None):
    """Process one record. Returns a dict for the log."""
    from bank_pipeline import bank_first_phase
    from txt_to_dat import write_wfdb_single_lead
    from pipeline_config import MEDIAN_WINDOW_S, A7_INCLUDED
    median_window = MEDIAN_WINDOW_S if median_window is None else median_window
    A7_included = A7_INCLUDED if A7_included is None else A7_included

    rel = os.path.relpath(hea_path, in_root)
    rec_name = os.path.splitext(os.path.basename(hea_path))[0]
    out_dir = os.path.join(out_root, os.path.dirname(rel))
    row = dict(record=os.path.splitext(rel)[0].replace("\\", "/"), status="", fs="", samples="",
               flat_tail=0, beats="", heart_rate_bpm="", min_mV="", max_mV="", note="")
    try:
        if not overwrite and os.path.exists(os.path.join(out_dir, rec_name + ".hea")):
            row["status"] = "exists (skipped)"; return row
        h = read_header(hea_path)
        row["fs"] = h["fs"]
        if int(round(h["fs"])) != FS_REQUIRED:
            row["status"] = "skipped"; row["note"] = f"fs {h['fs']} Hz (pipeline needs {FS_REQUIRED} Hz)"; return row
        leads, h = read_leads_mv(hea_path)
        x = leads["V2"] - leads["V1"]
        N = len(x); row["samples"] = N

        flat = flat_run_at_end(x)
        flat = flat if flat >= FLAT_MIN else 0
        row["flat_tail"] = flat
        work = x[: N - flat] if flat else x
        if len(work) < 3 * FS_REQUIRED:
            row["status"] = "skipped"; row["note"] = "less than 3 s of signal"; return row

        # Mirror EDGE_PAD samples onto both ends so the record start/end look like the middle of a long
        # stream (on the device the SWT/ISWT always have past and future data; a 10 s file does not).
        n_w = len(work)
        ext = np.r_[work[EDGE_PAD:0:-1], work, work[-2:-EDGE_PAD - 2:-1]]
        out = bank_first_phase(ext, FS_REQUIRED, warmup=False, median_window=median_window, A7_included=A7_included)
        clean = out["clean"][EDGE_PAD:EDGE_PAD + n_w]
        if np.any(np.isnan(clean)):
            raise RuntimeError("pipeline left NaN samples")
        peaks = out["r_peaks"] - EDGE_PAD
        peaks = peaks[(peaks >= 0) & (peaks < n_w)]
        if flat:
            clean = np.r_[clean, np.full(flat, clean[-1])]

        row["beats"] = len(peaks)
        if len(peaks) > 1:
            row["heart_rate_bpm"] = round(60.0 * FS_REQUIRED / np.mean(np.diff(peaks)), 1)
        if len(peaks) < 2:
            row["note"] = "few/no R peaks detected - check this record (flat or very noisy lead?)"
        row["min_mV"], row["max_mV"] = round(float(clean.min()), 3), round(float(clean.max()), 3)
        if np.max(np.abs(clean)) > 32.7:
            clean = np.clip(clean, -32.7, 32.7); row["note"] += " clipped to +-32.7 mV"

        write_wfdb_single_lead(clean, rec_name, out_dir, fs=FS_REQUIRED, sig_name="V2-V1")
        row["status"] = "ok"
    except Exception as e:                       # keep going with the other records
        row["status"] = "error"; row["note"] = f"{type(e).__name__}: {e}"
    return row


# ----------------------------------------------------------------------------- whole folder
def find_records(in_root, skip_root=None):
    found = []
    for dirpath, _, files in os.walk(in_root):
        if skip_root and os.path.abspath(dirpath).startswith(skip_root):
            continue                                  # never re-read our own output
        for f in files:
            if f.lower().endswith(".hea") and os.path.exists(os.path.join(dirpath, f[:-4] + ".dat")):
                found.append(os.path.join(dirpath, f))
    return sorted(found)


def main():
    ap = argparse.ArgumentParser(description="Clean a folder of WFDB (.dat/.hea) ECG records -> V2-V1 clean records")
    ap.add_argument("input_folder")
    ap.add_argument("output_folder")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--median-window", type=float, default=None,
                    help="median filter window in s (default: MEDIAN_WINDOW_S in pipeline_config.py = 0.6)")
    ap.add_argument("--remove-A7", action="store_true",
                    help="remove all of A7 (old behaviour); default: threshold A7 (pipeline_config.py)")
    args = ap.parse_args()
    from pipeline_config import MEDIAN_WINDOW_S, A7_INCLUDED
    median_window = MEDIAN_WINDOW_S if args.median_window is None else args.median_window
    A7_included = True if args.remove_A7 else A7_INCLUDED

    in_root, out_root = os.path.abspath(args.input_folder), os.path.abspath(args.output_folder)
    if in_root == out_root:
        sys.exit("output folder must be different from the input folder")
    records = find_records(in_root, out_root)[: args.limit]
    if not records:
        sys.exit(f"no .hea/.dat record pairs found under {in_root}")
    os.makedirs(out_root, exist_ok=True)
    print(f"{len(records)} records found under {in_root}\nwriting cleaned records to {out_root}")
    print(f"settings: median window {median_window} s, A7 {'removed' if A7_included else 'thresholded (kept)'}")
    with open(os.path.join(out_root, "clean_dataset_settings.txt"), "w") as f:
        f.write(f"median_window_s = {median_window}\nA7_removed = {A7_included}\nfs = {FS_REQUIRED}\n"
                f"output = single lead V2-V1, WFDB format 16, 1000 units/mV\n")

    rows, t0 = [], time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(clean_record, r, in_root, out_root, args.overwrite, median_window, A7_included) for r in records]
            for k, f in enumerate(as_completed(futs), 1):
                rows.append(f.result()); _progress(k, len(records), t0)
    else:
        for k, r in enumerate(records, 1):
            rows.append(clean_record(r, in_root, out_root, args.overwrite, median_window, A7_included)); _progress(k, len(records), t0)

    rows.sort(key=lambda r: r["record"])
    log_path = os.path.join(out_root, "clean_dataset_log.csv")
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\ndone in %.1f min: %s" % ((time.time() - t0) / 60, ", ".join(f"{v} {k}" for k, v in counts.items())))
    flagged = [r for r in rows if r["status"] == "error" or r["note"]]
    if flagged:
        print(f"{len(flagged)} records have a note or error -- see the 'note' column in {log_path}")
    print("log:", log_path)


def _progress(k, n, t0):
    if k == n or k % 50 == 0:
        el = time.time() - t0
        print(f"  {k}/{n} records  ({el / 60:.1f} min elapsed, ~{el / k * (n - k) / 60:.1f} min left)", flush=True)


if __name__ == "__main__":
    main()
