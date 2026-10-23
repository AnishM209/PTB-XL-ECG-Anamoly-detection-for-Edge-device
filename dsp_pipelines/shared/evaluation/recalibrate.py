"""
recalibrate.py: re-choose the decision thresholds of the exported models for signals cleaned by the team's
DSP pipeline (clean_dataset.py output), the same way Stage 1 did, and check the effect on the test fold.

  1. Fold 9 (validation), raw V2-V1:      thresholds must reproduce config.json  (SELF-CHECK)
  2. Fold 9, cleaned (team pipeline):     new thresholds (balanced = max(TPR - FPR), high_sens = first TPR >= 0.90)
  3. Fold 10 (test), cleaned:             metrics at the old and at the new thresholds, raw for reference
  4. Writes ~/export/config_clean06.json  (same models, new thresholds) for infer.py on cleaned signals

Records flagged in clean_dataset_log.csv (detector found < 2 beats) are excluded from fold 9 and reported
separately on fold 10.

Usage: sbatch recalibrate.sbatch   (GPU, a few minutes)
"""
import argparse
import ast
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.metrics import roc_auc_score, roc_curve

PTB = Path.home() / "data" / "ptb-xl"
EXPORT = Path.home() / "export"
sys.path.insert(0, str(EXPORT))
from models import build_model  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_batch import at_threshold, score  # noqa: E402  (same prediction code as the test scoring)

HIGH_SENS_TARGET = 0.90


def thr_balanced(y, p):                      # identical to stage1/evaluate.py
    fpr, tpr, th = roc_curve(y, p)
    return float(th[np.argmax(tpr - fpr)])


def thr_sensitivity(y, p, target):           # identical to stage1/evaluate.py
    fpr, tpr, th = roc_curve(y, p)
    return float(th[np.argmax(tpr >= target)])


def labels_for_fold(db, scp, fold):
    diag = scp[scp.diagnostic == 1].diagnostic_class.dropna().to_dict()
    rhythm = set(scp[scp.rhythm == 1].index)
    binary, af = {}, {}
    for ecg_id, c in db[db.strat_fold == fold].scp_codes.apply(ast.literal_eval).items():
        classes = {diag[k] for k in c if k in diag}
        if classes:
            binary[ecg_id] = int(bool(classes - {"NORM"}))
        if any(k in rhythm for k in c):
            af[ecg_id] = int(("AFIB" in c) or ("AFLT" in c))
    return {"binary": pd.Series(binary), "af": pd.Series(af)}


def read_pair(item):
    ecg_id, rel, clean_root = item
    sig, f = wfdb.rdsamp(str(PTB / rel))
    n = [s.upper() for s in f["sig_name"]]
    raw = (sig[:, n.index("V2")] - sig[:, n.index("V1")]).astype(np.float32)
    cln = wfdb.rdsamp(str(Path(clean_root) / rel))[0][:, 0].astype(np.float32)
    return ecg_id, raw, cln


def main(a):
    cfg = json.loads((EXPORT / "config.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    db = pd.read_csv(PTB / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(PTB / "scp_statements.csv", index_col=0)
    clean_root = Path(a.clean_root).expanduser()
    log = pd.read_csv(clean_root / "records500" / "clean_dataset_log.csv")
    flagged_rel = set("records500/" + log[log.note.notna()].record)
    flagged = set(db.index[db.filename_hr.isin(flagged_rel)])

    # ---- fold 9 signals (raw and cleaned)
    val_ids = sorted(db.index[db.strat_fold == 9])
    items = [(int(i), db.loc[i, "filename_hr"], str(clean_root)) for i in val_ids]
    with Pool(a.workers) as pool:
        res = pool.map(read_pair, items, chunksize=16)
    V = {"raw": pd.DataFrame(np.stack([r[1] for r in res]), index=[r[0] for r in res]),
         "clean": pd.DataFrame(np.stack([r[2] for r in res]), index=[r[0] for r in res])}
    val_lab = labels_for_fold(db, scp, 9)

    # ---- fold 10 signals from the batch test
    batch = Path(a.batch)
    T = {}
    for name, f in (("raw", "raw.npz"), ("clean", f"{a.test_name}.npz")):
        z = np.load(batch / f)
        T[name] = pd.DataFrame(z["X"], index=z["ids"])
    test_lab = labels_for_fold(db, scp, 10)

    new_cfg = json.loads(json.dumps(cfg))
    lines = []
    for task, c in cfg.items():
        model = build_model(c["model"], 1, 1)
        model.load_state_dict(torch.load(EXPORT / c["weights"], map_location="cpu"))
        model.to(device).eval()

        # validation
        yv = val_lab[task]
        ids = yv.index
        kv = ~ids.isin(flagged)
        pv = {s: score(model, V[s].loc[ids].to_numpy(np.float32), c, device) for s in ("raw", "clean")}
        y_all = yv.values
        # raw: all records, exactly as Stage 1 (self-check); cleaned: flagged records excluded
        th = {"raw": (thr_balanced(y_all, pv["raw"]), thr_sensitivity(y_all, pv["raw"], HIGH_SENS_TARGET)),
              "clean": (thr_balanced(y_all[kv], pv["clean"][kv]),
                        thr_sensitivity(y_all[kv], pv["clean"][kv], HIGH_SENS_TARGET))}
        ok = abs(th["raw"][0] - c["threshold_balanced"]) < 1e-3 and abs(th["raw"][1] - c["threshold_high_sens"]) < 1e-3
        lines.append(f"\n===== {task} =====")
        lines.append(f"fold 9: {len(ids)} records ({int(yv.sum())} positive); {int((~kv).sum())} flagged, excluded for the cleaned thresholds")
        lines.append(f"  raw     thresholds: balanced {th['raw'][0]:.4f}, high_sens {th['raw'][1]:.4f}   "
                     f"(config.json: {c['threshold_balanced']:.4f}, {c['threshold_high_sens']:.4f})")
        lines.append("  SELF-CHECK raw fold-9 thresholds reproduce config.json: " + ("PASS" if ok else
                     "FAIL (validation predictions differ from Stage 1: stop and check)"))
        lines.append(f"  cleaned thresholds: balanced {th['clean'][0]:.4f}, high_sens {th['clean'][1]:.4f}")
        new_cfg[task]["threshold_balanced"] = th["clean"][0]
        new_cfg[task]["threshold_high_sens"] = th["clean"][1]

        # test
        yt = test_lab[task]
        ids_t = yt.index[yt.index.isin(T["raw"].index)]
        y = yt.loc[ids_t].to_numpy()
        keep = ~ids_t.isin(flagged)
        pt = {s: score(model, T[s].loc[ids_t].to_numpy(np.float32), c, device) for s in ("raw", "clean")}
        lines.append(f"fold 10: {len(y)} records ({int(y.sum())} positive), {int((~keep).sum())} of them flagged")
        lines.append(f"  {'signal, threshold':34s} {'AUROC':>6s} {'sens':>6s} {'spec':>6s} {'PPV':>6s} {'TP':>5s} {'FN':>5s} {'FP':>5s} {'TN':>5s}")
        rows = [("raw, exported balanced", "raw", c["threshold_balanced"]),
                ("cleaned, exported balanced", "clean", c["threshold_balanced"]),
                ("cleaned, NEW balanced", "clean", th["clean"][0]),
                ("raw, exported high_sens", "raw", c["threshold_high_sens"]),
                ("cleaned, NEW high_sens", "clean", th["clean"][1])]
        for lab, s, t in rows:
            for subset, mask in (("", np.ones(len(y), bool)), (" (excl. flagged)", keep)):
                if subset and (~keep).sum() == 0:
                    continue
                m = at_threshold(y[mask], pt[s][mask], t)
                lines.append(f"  {(lab + subset):34s} {roc_auc_score(y[mask], pt[s][mask]):6.4f} {m['sens']:6.3f} "
                             f"{m['spec']:6.3f} {m['ppv']:6.3f} {m['tp']:5d} {m['fn']:5d} {m['fp']:5d} {m['tn']:5d}")

    for task in new_cfg:
        new_cfg[task]["thresholds_chosen_on"] = (f"PTB-XL fold 9 cleaned with clean_dataset.py "
                                                 f"(0.6 s median, A7 thresholded), flagged records excluded")
    out = EXPORT / "config_clean06.json"
    out.write_text(json.dumps(new_cfg, indent=2))
    summary = "\n".join(lines)
    print(summary)
    (batch / "recalibration_summary.txt").write_text(summary + "\n")
    print(f"\nwrote {out}\nsaved {batch / 'recalibration_summary.txt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_root", default=str(Path.home() / "data" / "ptb-xl-clean06"))
    ap.add_argument("--batch", default=str(Path.home() / "dsp_test" / "batch"))
    ap.add_argument("--test_name", default="team06")
    ap.add_argument("--workers", type=int, default=8)
    main(ap.parse_args())
