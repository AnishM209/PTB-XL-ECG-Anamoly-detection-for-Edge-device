"""
Convert a cleaned single-lead ECG txt file (one value per line, in mV) into a WFDB record
(<name>.dat + <name>.hea) in the same format as the PTB-XL files:

    format 16 (16-bit little-endian integers), gain 1000 ADC units per mV (1 unit = 1 uV),
    baseline 0, 500 Hz, one signal named "V2-V1".

    python txt_to_dat.py 00057_hr_V2_minus_V1_clean.txt                # -> 00057_hr_V2_minus_V1_clean.dat/.hea
    python txt_to_dat.py 00057_hr_V2_minus_V1_clean.txt --name 00057_hr_clean --fs 500

Read it back with the wfdb package (what the CNN team will likely use):
    import wfdb
    rec = wfdb.rdrecord("00057_hr_V2_minus_V1_clean")     # rec.p_signal[:, 0] is the ECG in mV
"""
import argparse
import os

import numpy as np

from ecg_io import load_ecg_txt


def write_wfdb_single_lead(signal_mv, record_name, out_dir=".", fs=500, gain=1000.0, sig_name="V2-V1",
                           units="mV"):
    """Write signal_mv as a WFDB format-16 record. Returns (dat_path, hea_path, digital_samples)."""
    x = np.asarray(signal_mv, dtype=float)
    digital = np.round(x * gain)
    if digital.min() < -32767 or digital.max() > 32767:          # -32768 is reserved as 'invalid sample'
        raise ValueError(f"Signal out of 16-bit range at gain {gain}: {x.min():.3f}..{x.max():.3f} {units}")
    digital = digital.astype("<i2")

    os.makedirs(out_dir, exist_ok=True)
    dat_path = os.path.join(out_dir, record_name + ".dat")
    hea_path = os.path.join(out_dir, record_name + ".hea")
    digital.tofile(dat_path)

    checksum = int(digital.astype(np.int64).sum()) & 0xFFFF       # 16-bit checksum, as in the PTB-XL headers
    with open(hea_path, "w", newline="\n") as f:
        f.write(f"{record_name} 1 {fs} {len(digital)}\n")
        # file  format  gain(baseline)/units  adc_resolution  adc_zero  initial_value  checksum  block_size  name
        f.write(f"{record_name}.dat 16 {gain:.1f}(0)/{units} 16 0 {int(digital[0])} {checksum} 0 {sig_name}\n")
    return dat_path, hea_path, digital


def read_wfdb_single_lead(record_path):
    """Minimal reader (no wfdb package needed) -- returns (signal in physical units, fs)."""
    lines = [l.split() for l in open(record_path + ".hea") if l.strip() and not l.startswith("#")]
    fs = float(lines[0][2])
    gain = float(lines[1][2].split("(")[0].split("/")[0])
    base = int(lines[1][2].split("(")[1].split(")")[0]) if "(" in lines[1][2] else 0
    d = np.fromfile(record_path + ".dat", dtype="<i2")
    return (d - base) / gain, fs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Clean ECG txt -> WFDB .dat/.hea")
    ap.add_argument("files", nargs="+", help="one or more clean txt files")
    ap.add_argument("--fs", type=int, default=500)
    ap.add_argument("--name", default=None, help="record name (only when converting a single file)")
    ap.add_argument("--out", default=".", help="output folder (default: this folder)")
    args = ap.parse_args()

    for path in args.files:
        name = args.name if (args.name and len(args.files) == 1) else os.path.splitext(os.path.basename(path))[0]
        x = load_ecg_txt(path)
        dat, hea, digital = write_wfdb_single_lead(x, name, args.out, fs=args.fs)
        back, _ = read_wfdb_single_lead(os.path.join(args.out, name))
        err = np.max(np.abs(back - x))
        print(f"{os.path.basename(path)} -> {name}.dat / {name}.hea  ({len(digital)} samples, "
              f"{x.min():+.3f}..{x.max():+.3f} mV, max rounding error {1000 * err:.2f} uV)")
