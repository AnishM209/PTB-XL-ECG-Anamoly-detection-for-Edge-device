"""
check_dsp.py: (1) reproduction check of our chain against the DSP teammate's cleaned files,
              (2) stage-by-stage T-wave measurement for three variants, saving .npy files for infer.py.

Usage (in ~/dsp_test, env ecg):
  python check_dsp.py --ids 57 15526
  python check_dsp.py --ids 57 15526 --pair 57 ~/samples/cleaned/00057_clean --pair 15526 ~/samples/cleaned/15526_clean

--pair takes the WFDB record path of the teammate's cleaned file WITHOUT the extension.
"""
import argparse
from pathlib import Path

import numpy as np
import wfdb

from dsp_chain import beat_amplitudes, clean, refine_r

PTB = Path.home() / "data" / "ptb-xl" / "records500"
FS = 500

VARIANTS = {
    "teammate":       dict(median_mode="single", median_win=0.3, a7_included=True),
    "two_stage":      dict(median_mode="two_stage", a7_included=True),
    "two_stage_noA7": dict(median_mode="two_stage", a7_included=False),
    "hp05_noA7":      dict(median_mode="hp05", a7_included=False),
}


def load_v2v1(ecg_id):
    p = PTB / f"{(ecg_id // 1000) * 1000:05d}" / f"{ecg_id:05d}_hr"
    sig, f = wfdb.rdsamp(str(p))
    names = [n.upper() for n in f["sig_name"]]
    assert f["fs"] == FS, f"unexpected fs {f['fs']}"
    return sig[:, names.index("V2")] - sig[:, names.index("V1")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", default=[57, 15526])
    ap.add_argument("--pair", nargs=2, action="append", default=[], metavar=("ID", "CLEANED_PATH"))
    ap.add_argument("--out", default="npy_out")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    pairs = {int(i): str(Path(p).expanduser()) for i, p in args.pair}

    for ecg_id in args.ids:
        raw = load_v2v1(ecg_id)
        np.save(out / f"{ecg_id:05d}_raw.npy", raw.astype(np.float32))
        print(f"\n=== ecg_id {ecg_id}  ({len(raw)} samples) ===")

        results = {}
        for name, kw in VARIANTS.items():
            cleaned, info = clean(raw, FS, **kw)
            results[name] = (cleaned, info)
            np.save(out / f"{ecg_id:05d}_{name}.npy", cleaned.astype(np.float32))

        # Same beat positions for every signal: Christov peaks from the teammate variant, refined on raw.
        r_idx = refine_r(raw, results["teammate"][1]["r_peaks"], FS)
        print(f"beats detected: {len(r_idx)}")
        print(f"{'variant':16s} {'stage':8s} {'|T| mV':>8s} {'|QRS| mV':>9s} {'T/QRS':>7s} {'beats':>6s}")
        t, q, rq, n = beat_amplitudes(raw, r_idx, FS)
        print(f"{'-':16s} {'raw':8s} {t:8.3f} {q:9.3f} {rq:7.3f} {n:6d}")
        for name, (cleaned, info) in results.items():
            for stage, sig in (("pre", info["pre"]), ("cleaned", cleaned)):
                t, q, rq, n = beat_amplitudes(sig, r_idx, FS)
                print(f"{name:16s} {stage:8s} {t:8.3f} {q:9.3f} {rq:7.3f} {n:6d}")

        if ecg_id in pairs:
            theirs = wfdb.rdsamp(pairs[ecg_id])[0][:, 0]
            ours = results["teammate"][0]
            L = min(len(ours), len(theirs))
            if len(ours) != len(theirs):
                print(f"WARNING: length differs (ours {len(ours)}, theirs {len(theirs)}); comparing first {L}")
            d = ours[:L] - theirs[:L]
            corr = np.corrcoef(ours[:L], theirs[:L])[0, 1]
            print(f"REPRO vs teammate file: max|diff| {np.max(np.abs(d)):.5f} mV, "
                  f"RMS diff {np.sqrt(np.mean(d ** 2)):.5f} mV, corr {corr:.6f}")
            for name, (cleaned, _) in results.items():
                if name != "teammate":
                    dd = cleaned[:L] - theirs[:L]
                    print(f"  (for reference, {name} vs teammate file: RMS diff {np.sqrt(np.mean(dd ** 2)):.5f} mV)")

    print(f"\nSaved .npy files to {out.resolve()}")


if __name__ == "__main__":
    main()
