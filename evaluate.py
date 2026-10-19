"""
Detailed test-set evaluation of trained Stage 1 models, for presenting results.

For one task and model, and one or more lead settings, this loads every seed's
best.pt, recomputes predictions on the validation and test folds, and reports:
  - AUROC and AUPRC on the test fold (mean and spread across seeds)
  - sensitivity, specificity, PPV, NPV and F1 at two operating points, both
    chosen on the VALIDATION fold only and then applied unchanged to test:
        balanced   threshold maximising sensitivity + specificity (Youden)
        high_sens  highest threshold that reaches --sens_target sensitivity
  - 95% bootstrap intervals on the test fold (seed 0), which show how much the
    numbers could move with a different set of test patients
  - figures: ROC and PR curves, confusion matrices, example ECG strips

Usage:
    python evaluate.py --task af     --leads V2-V1 V3
    python evaluate.py --task binary --leads V2-V1 V3
Outputs: ~/stage1/eval/<task>_<model>/
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

import ptbxl_data as P
import train as T
from models import build_model

POS_NAME = {"af": "AF", "binary": "Abnormal"}
NEG_NAME = {"af": "No AF", "binary": "Normal"}


# ------------------------------------------------------------------ runs
def lead_tag(leads_arg):
    return "12lead" if leads_arg == "all" else "+".join(leads_arg.split(","))


def find_runs(runs_dir, task, model, leads_arg):
    """Finished seed runs for one configuration, sorted by seed."""
    found = []
    for d in Path(runs_dir).glob(f"{task}_{model}_{lead_tag(leads_arg)}_*"):
        if d.name.endswith("_smoke"):
            continue
        if not ((d / "results.json").exists() and (d / "best.pt").exists()):
            continue
        cfg = json.loads((d / "results.json").read_text())["config"]
        if cfg["task"] == task and cfg["model"] == model and cfg["leads"] == leads_arg:
            found.append((cfg["seed"], d, cfg))
    return sorted(found, key=lambda t: t[0])


# ------------------------------------------------------------------ data
def prepare(cfg):
    """Validation and test data, normalised exactly as train.py does it."""
    db, scp = P.load_metadata(cfg["root"])
    if cfg.get("exclude_paced"):
        db = db[db["pacemaker"].isna()]
    Y_df = T.build_task(db, scp, cfg["task"])
    leads = None if cfg["leads"] == "all" else cfg["leads"].split(",")
    X_all, ids = T.load_signals(cfg["cache"], cfg["fs"], leads)
    row = {e: i for i, e in enumerate(ids)}
    data = {}
    for split, folds in P.FOLDS.items():
        sel = Y_df.index[db.loc[Y_df.index, "strat_fold"].isin(folds)].to_numpy()
        data[split] = (X_all[[row[e] for e in sel]], Y_df.loc[sel].to_numpy(np.float32)[:, 0], sel)
    del X_all
    mu = data["train"][0].mean(axis=(0, 2), keepdims=True)
    sd = data["train"][0].std(axis=(0, 2), keepdims=True) + 1e-6
    raw_test = data["test"][0]
    out = {k: (((x - mu) / sd).astype(np.float32), y, sel)
           for k, (x, y, sel) in data.items() if k != "train"}
    return out, raw_test, db


def predict_run(run_dir, cfg, data, device):
    win = int(round(cfg["win_sec"] * cfg["fs"]))
    n_leads = data["val"][0].shape[1]
    model = build_model(cfg["model"], n_leads, 1).to(device)
    model.load_state_dict(torch.load(Path(run_dir) / "best.pt", map_location=device))
    return {k: T.predict(model, data[k][0], win, device)[:, 0] for k in ("val", "test")}


# --------------------------------------------------------------- metrics
def thr_balanced(y, p):
    fpr, tpr, th = roc_curve(y, p)
    return float(th[np.argmax(tpr - fpr)])


def thr_sensitivity(y, p, target):
    fpr, tpr, th = roc_curve(y, p)
    return float(th[np.argmax(tpr >= target)])


def at_threshold(y, p, t):
    pred, pos = p >= t, y == 1
    tp, fn = int(np.sum(pred & pos)), int(np.sum(~pred & pos))
    fp, tn = int(np.sum(pred & ~pos)), int(np.sum(~pred & ~pos))
    div = lambda a, b: a / b if b else float("nan")
    return {"threshold": t, "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn), "f1": div(2 * tp, 2 * tp + fp + fn),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def bootstrap(y, p, thresholds, n_boot, seed=0):
    """95% intervals on the test fold for AUROC, AUPRC and the operating points."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        r = {"auroc": roc_auc_score(y[i], p[i]), "auprc": average_precision_score(y[i], p[i])}
        for name, t in thresholds.items():
            m = at_threshold(y[i], p[i], t)
            r[f"{name}_sensitivity"], r[f"{name}_specificity"] = m["sensitivity"], m["specificity"]
        rows.append(r)
    B = pd.DataFrame(rows)
    return {c: (float(B[c].quantile(0.025)), float(B[c].quantile(0.975))) for c in B.columns}


# ---------------------------------------------------------------- figures
def plot_curves(curves, task, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for i, (tag, c) in enumerate(curves.items()):
        col = f"C{i}"
        for j, (y, p) in enumerate(c["seeds"]):
            fpr, tpr, _ = roc_curve(y, p)
            prec, rec, _ = precision_recall_curve(y, p)
            lab = (f"{tag}: AUROC {c['auroc'][0]:.3f} ± {c['auroc'][1]:.3f}" if j == 0 else None)
            labp = (f"{tag}: AUPRC {c['auprc'][0]:.3f} ± {c['auprc'][1]:.3f}" if j == 0 else None)
            ax[0].plot(fpr, tpr, color=col, lw=1.4 if j == 0 else 0.6, alpha=1 if j == 0 else 0.5, label=lab)
            ax[1].plot(rec, prec, color=col, lw=1.4 if j == 0 else 0.6, alpha=1 if j == 0 else 0.5, label=labp)
        for name, mk in (("balanced", "o"), ("high_sens", "s")):
            m = c["op"][name]
            ax[0].plot(1 - m["specificity"], m["sensitivity"], mk, color=col, ms=7, mec="k")
    prev = curves[next(iter(curves))]["seeds"][0][0].mean()
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[1].axhline(prev, color="k", ls=":", lw=0.8, label=f"chance (prevalence {prev:.2f})")
    ax[0].set(xlabel="1 - specificity (false positive rate)", ylabel="Sensitivity",
              title=f"ROC, {POS_NAME[task]} (circle: balanced, square: high sensitivity)")
    ax[1].set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)", title=f"Precision-recall, {POS_NAME[task]}")
    for a in ax:
        a.legend(fontsize=8, loc="lower right" if a is ax[0] else "lower left")
        a.grid(alpha=0.3)
        a.set_xlim(0, 1)
        a.set_ylim(0, 1.01)
    fig.suptitle("PTB-XL test fold (fold 10), seed 0 bold, other seeds thin", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "roc_pr.png", dpi=150)
    plt.close(fig)


def plot_confusion(curves, task, out):
    tags = list(curves)
    fig, ax = plt.subplots(2, len(tags), figsize=(3.6 * len(tags), 6.6), squeeze=False)
    for j, tag in enumerate(tags):
        for i, name in enumerate(("balanced", "high_sens")):
            m = curves[tag]["op"][name]
            M = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
            a = ax[i, j]
            a.imshow(M / M.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
            for r in range(2):
                for c in range(2):
                    frac = M[r, c] / M[r].sum()
                    a.text(c, r, f"{M[r, c]}\n({frac:.0%})", ha="center", va="center",
                           color="white" if frac > 0.5 else "black", fontsize=10)
            a.set_xticks([0, 1], [NEG_NAME[task], POS_NAME[task]])
            a.set_yticks([0, 1], [NEG_NAME[task], POS_NAME[task]])
            a.set_xlabel("Predicted")
            a.set_ylabel("True")
            a.set_title(f"{tag}, {name}\nsens {m['sensitivity']:.2f}  spec {m['specificity']:.2f}", fontsize=9)
    fig.suptitle("Test fold, seed 0, thresholds chosen on validation", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=150)
    plt.close(fig)


def plot_examples(tag, raw, y, p, ids, t, db, task, fs, out):
    """Most confident correct detection, most confident miss, most confident false alarm."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    picks = [("Correct detection", pos[np.argmax(p[pos])]),
             ("Missed (false negative)", pos[np.argmin(p[pos])]),
             ("False alarm (false positive)", neg[np.argmax(p[neg])])]
    tt = np.arange(raw.shape[-1]) / fs
    fig, ax = plt.subplots(len(picks), 1, figsize=(12, 2.4 * len(picks)), sharex=True)
    for a, (title, i) in zip(ax, picks):
        codes = ", ".join(sorted(db.loc[ids[i], "scp_codes"], key=lambda c: -db.loc[ids[i], "scp_codes"][c])[:4])
        a.plot(tt, raw[i, 0], lw=0.7, color="k")
        verdict = POS_NAME[task] if p[i] >= t else NEG_NAME[task]
        a.set_title(f"{title}: ecg_id {ids[i]}, label {POS_NAME[task] if y[i] else NEG_NAME[task]}, "
                    f"p={p[i]:.2f} -> {verdict}   [PTB-XL codes: {codes}]", fontsize=9, loc="left")
        a.set_ylabel(f"{tag} (mV)")
        a.grid(alpha=0.3)
    ax[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out / f"examples_{tag}.png", dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------- main
def main(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out) / f"{a.task}_{a.model}"
    out.mkdir(parents=True, exist_ok=True)
    rows, curves, cis = [], {}, {}
    for leads_arg in a.leads:
        runs = find_runs(a.runs, a.task, a.model, leads_arg)
        if not runs:
            print(f"No finished runs for {a.task} {a.model} {leads_arg}, skipping")
            continue
        tag = lead_tag(leads_arg)
        print(f"{tag}: seeds {[s for s, _, _ in runs]}  device={device}")
        data, raw_test, db = prepare(runs[0][2])
        y_val, y_test, test_ids = data["val"][1], data["test"][1], data["test"][2]
        seed_curves = []
        for k, (seed, d, cfg) in enumerate(runs):
            pr = predict_run(d, cfg, data, device)
            th = {"balanced": thr_balanced(y_val, pr["val"]),
                  "high_sens": thr_sensitivity(y_val, pr["val"], a.sens_target)}
            r = {"leads": tag, "seed": seed, "n_test": len(y_test), "n_pos_test": int(y_test.sum()),
                 "auroc": roc_auc_score(y_test, pr["test"]),
                 "auprc": average_precision_score(y_test, pr["test"])}
            ops = {name: at_threshold(y_test, pr["test"], t) for name, t in th.items()}
            for name, m in ops.items():
                r.update({f"{name}_{key}": v for key, v in m.items()})
            rows.append(r)
            seed_curves.append((y_test, pr["test"]))
            if k == 0:
                first = (pr, th, ops)
                cis[tag] = bootstrap(y_test, pr["test"], th, a.n_boot)
                plot_examples(tag, raw_test, y_test, pr["test"], test_ids, th["high_sens"],
                              db, a.task, cfg["fs"], out)
        R = pd.DataFrame([r for r in rows if r["leads"] == tag])
        curves[tag] = {"seeds": seed_curves, "op": first[2],
                       "auroc": (R["auroc"].mean(), R["auroc"].std()),
                       "auprc": (R["auprc"].mean(), R["auprc"].std())}
    if not rows:
        raise SystemExit("Nothing to evaluate.")

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_seed.csv", index=False)
    cols = ["auroc", "auprc"] + [f"{o}_{m}" for o in ("balanced", "high_sens")
                                 for m in ("sensitivity", "specificity", "ppv", "npv", "f1")]
    summ = df.groupby("leads")[cols].agg(["mean", "std"])
    summ.to_csv(out / "summary.csv")

    # Markdown table for slides or the report.
    lines = [f"# {POS_NAME[a.task]} detection, {a.model}, PTB-XL test fold",
             "", f"Thresholds chosen on the validation fold. high_sens targets "
             f"{a.sens_target:.0%} sensitivity. Mean ± spread over seeds; "
             "[95% bootstrap interval, seed 0].", ""]
    lines += ["| Lead | n test (pos) | AUROC | AUPRC | Balanced sens / spec | "
              "High-sens sens / spec | High-sens PPV |", "|---|---|---|---|---|---|---|"]
    fmt = lambda c, g: f"{g[(c, 'mean')]:.3f} ± {g[(c, 'std')]:.3f}"
    for tag, g in summ.iterrows():
        ci = cis[tag]
        n = df[df["leads"] == tag].iloc[0]
        lines.append(
            f"| {tag} | {n['n_test']} ({n['n_pos_test']}) "
            f"| {fmt('auroc', g)} [{ci['auroc'][0]:.3f}, {ci['auroc'][1]:.3f}] "
            f"| {fmt('auprc', g)} [{ci['auprc'][0]:.3f}, {ci['auprc'][1]:.3f}] "
            f"| {g[('balanced_sensitivity', 'mean')]:.2f} / {g[('balanced_specificity', 'mean')]:.2f} "
            f"| {g[('high_sens_sensitivity', 'mean')]:.2f} / {g[('high_sens_specificity', 'mean')]:.2f} "
            f"[spec {ci['high_sens_specificity'][0]:.2f} to {ci['high_sens_specificity'][1]:.2f}] "
            f"| {g[('high_sens_ppv', 'mean')]:.2f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")

    plot_curves(curves, a.task, out)
    plot_confusion(curves, a.task, out)
    print("\n".join(lines))
    print(f"\nWritten to {out}: summary.md, summary.csv, per_seed.csv, roc_pr.png, "
          f"confusion.png, examples_<lead>.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["binary", "af"], required=True)
    ap.add_argument("--model", default="resnet1d_wang_s5")
    ap.add_argument("--leads", nargs="+", default=["V2-V1", "V3"])
    ap.add_argument("--runs", type=Path, default=Path.home() / "stage1/runs")
    ap.add_argument("--out", type=Path, default=Path.home() / "stage1/eval")
    ap.add_argument("--sens_target", type=float, default=0.90)
    ap.add_argument("--n_boot", type=int, default=1000)
    main(ap.parse_args())
