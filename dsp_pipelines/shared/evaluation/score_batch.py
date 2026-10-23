"""
score_batch.py: Part B of the batch DSP test.
Runs the exported V2-V1 seed-0 binary and AF models (~/export) on the signals saved by batch_clean.py
and compares each DSP variant with raw.

Labels are rebuilt from the PTB-XL csv files with the Stage 1 definitions:
  binary: records with at least one diagnostic statement; normal = NORM is the only diagnostic class,
          abnormal = any of MI, STTC, CD, HYP.
  AF:     records with at least one rhythm statement; positive = AFIB or AFLT.
SELF-CHECK: the raw rows must reproduce the Stage 1 test fold (binary n 2158 / 1246 pos, AUROC 0.8519,
TP 868 FN 378 FP 126 TN 786; AF n 2098 / 157 pos, TP 136 FN 21 FP 45 TN 1896). If they do not,
stop: the comparison is not valid.

Prediction exactly as infer.py: (x - mu) / sd, windows with 50% overlap, sigmoid averaged.
Thresholds: the exported balanced thresholds from config.json.

Usage: sbatch score_batch.sbatch     (or: python score_batch.py on a GPU node)
"""
import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

PTB = Path.home() / "data" / "ptb-xl"
EXPORT = Path.home() / "export"
sys.path.insert(0, str(EXPORT))
from models import build_model  # noqa: E402

EXPECTED = {"binary": dict(n=2158, pos=1246, tp=868, fn=378, fp=126, tn=786, auroc=0.8519),
            "af": dict(n=2098, pos=157, tp=136, fn=21, fp=45, tn=1896, auroc=None)}


def build_labels():
    db = pd.read_csv(PTB / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(PTB / "scp_statements.csv", index_col=0)
    diag = scp[scp.diagnostic == 1].diagnostic_class.dropna().to_dict()
    rhythm = set(scp[scp.rhythm == 1].index)
    test = db[db.strat_fold == 10]
    codes = test.scp_codes.apply(ast.literal_eval)

    binary, af = {}, {}
    for ecg_id, c in codes.items():
        classes = {diag[k] for k in c if k in diag}
        if classes:
            binary[ecg_id] = int(bool(classes - {"NORM"}))
        if any(k in rhythm for k in c):
            af[ecg_id] = int(("AFIB" in c) or ("AFLT" in c))
    return {"binary": pd.Series(binary), "af": pd.Series(af)}


@torch.no_grad()
def score(model, X, c, device, bs=1024):
    win = int(round(c["win_sec"] * c["fs"]))
    T = X.shape[1]
    starts = list(range(0, T - win + 1, max(1, win // 2)))
    if starts[-1] != T - win:
        starts.append(T - win)
    out = np.full(len(X), np.nan)
    ok = ~np.isnan(X).any(axis=1)
    Xn = ((X[ok] - c["mu"][0]) / c["sd"][0]).astype(np.float32)
    flat = np.stack([Xn[:, s:s + win] for s in starts], axis=1).reshape(-1, 1, win)
    probs = []
    for i in range(0, len(flat), bs):
        xb = torch.from_numpy(flat[i:i + bs]).to(device)
        probs.append(torch.sigmoid(model(xb)).float().cpu().numpy().ravel())
    out[ok] = np.concatenate(probs).reshape(-1, len(starts)).mean(axis=1)
    return out


def at_threshold(y, p, t):
    pred = p >= t
    tp = int(((y == 1) & pred).sum()); fn = int(((y == 1) & ~pred).sum())
    fp = int(((y == 0) & pred).sum()); tn = int(((y == 0) & ~pred).sum())
    return dict(tp=tp, fn=fn, fp=fp, tn=tn,
                sens=tp / max(1, tp + fn), spec=tn / max(1, tn + fp), ppv=tp / max(1, tp + fp))


def paired_delta_auroc(y, p_ref, p_var, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    n = len(y)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], p_var[i]) - roc_auc_score(y[i], p_ref[i]))
    d = np.asarray(d)
    return float(np.mean(d)), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def main(a):
    global VARIANTS, SIGNALS
    VARIANTS = a.variants
    SIGNALS = ["raw"] + VARIANTS
    batch = Path(a.batch)
    cfg = json.loads((EXPORT / "config.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    data = {}
    for s in SIGNALS:
        z = np.load(batch / f"{s}.npz")
        data[s] = pd.DataFrame(z["X"], index=z["ids"])
    labels = build_labels()

    lines, score_rows = [], []
    for task, c in cfg.items():
        y_ser = labels[task]
        ids = y_ser.index[y_ser.index.isin(data["raw"].index)]
        y = y_ser.loc[ids].to_numpy()
        model = build_model(c["model"], 1, 1)
        model.load_state_dict(torch.load(EXPORT / c["weights"], map_location="cpu"))
        model.to(device).eval()
        t = c["threshold_balanced"]

        P = {s: score(model, data[s].loc[ids].to_numpy(dtype=np.float32), c, device) for s in SIGNALS}
        score_rows.append(pd.DataFrame({"task": task, "ecg_id": ids, "y": y,
                                        **{f"p_{s}": P[s] for s in SIGNALS}}))

        lines.append(f"\n===== {task}: n {len(y)}, positives {int(y.sum())}, balanced threshold {t:.3f} =====")
        lines.append(f"{'signal':10s} {'n':>5s} {'AUROC':>7s} {'AUPRC':>7s} {'sens':>6s} {'spec':>6s} "
                     f"{'PPV':>6s} {'TP':>5s} {'FN':>5s} {'FP':>5s} {'TN':>5s}")
        for s in SIGNALS:
            ok = ~np.isnan(P[s])
            m = at_threshold(y[ok], P[s][ok], t)
            lines.append(f"{s:10s} {ok.sum():5d} {roc_auc_score(y[ok], P[s][ok]):7.4f} "
                         f"{average_precision_score(y[ok], P[s][ok]):7.4f} {m['sens']:6.3f} {m['spec']:6.3f} "
                         f"{m['ppv']:6.3f} {m['tp']:5d} {m['fn']:5d} {m['fp']:5d} {m['tn']:5d}")

        # Self-check of the raw row against Stage 1
        e = EXPECTED[task]
        m = at_threshold(y, P["raw"], t)
        checks = [len(y) == e["n"], int(y.sum()) == e["pos"],
                  all(m[k] == e[k] for k in ("tp", "fn", "fp", "tn"))]
        if e["auroc"] is not None:
            checks.append(abs(roc_auc_score(y, P["raw"]) - e["auroc"]) < 0.001)
        lines.append("SELF-CHECK raw vs Stage 1: " + ("PASS" if all(checks) else
                     f"FAIL (expected {e}, got n {len(y)}, pos {int(y.sum())}, {m})"))

        # Changes relative to raw, at the balanced threshold
        pr = P["raw"] >= t
        for s in VARIANTS:
            ok = ~np.isnan(P[s])
            pv = P[s] >= t
            neg, pos = (y == 0) & ok, (y == 1) & ok
            dm, lo, hi = paired_delta_auroc(y[ok], P["raw"][ok], P[s][ok])
            lines.append(
                f"{s:10s} vs raw: dAUROC {dm:+.4f} [95% CI {lo:+.4f}, {hi:+.4f}] | "
                f"negatives: new false alarms {int((neg & ~pr & pv).sum())}, cleared {int((neg & pr & ~pv).sum())} | "
                f"positives: lost {int((pos & pr & ~pv).sum())}, gained {int((pos & ~pr & pv).sum())} | "
                f"failed records {int((~ok).sum())}")

    summary = "\n".join(lines)
    print(summary)
    (batch / "score_summary.txt").write_text(summary + "\n")
    pd.concat(score_rows).to_csv(batch / "scores.csv", index=False)
    print(f"\nsaved score_summary.txt and scores.csv to {batch}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=str(Path.home() / "dsp_test" / "batch"))
    ap.add_argument("--variants", nargs="+", default=["teammate", "hp05_noA7"],
                    help="npz names in --batch to compare with raw, e.g. teammate hp05_noA7 team06")
    main(ap.parse_args())
