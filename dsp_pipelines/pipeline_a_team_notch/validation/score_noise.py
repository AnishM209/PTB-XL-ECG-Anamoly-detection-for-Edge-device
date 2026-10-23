"""
score_noise.py: scores the noise stress test outputs (noise_stress.py) with the exported V2-V1 seed-0 models,
using the SAME prediction, label and threshold code as score_batch.py / recalibrate.py (imported, not copied).

Signals (fold 10, batch/):  ns_<noise>_<snr>_raw.npz            noisy raw (no cleaning)
                            ns_<tag>_<noise>_<snr>_<pipe>.npz   cleaned, tag orig / reset
Thresholds (balanced = max(TPR-FPR), high_sens = first TPR >= 0.90), chosen on CLEAN fold 9 per pipeline,
then applied unchanged at every noise level (calibrate once on clean data, meet noise in the field):
  raw            config.json                      (SELF-CHECK: ns_f9_raw.npz reproduces it)
  med06          config_clean06.json               (recalibrate.py, fold 9 via clean_dataset.py)
  med03, hp05zp, hp05c   chosen here on ns_f9_orig_<pipe>.npz, flagged records excluded as in recalibrate.py
SELF-CHECK 2: ns_none_inf_raw reproduces the Stage 1 test fold (same check as score_batch.py).

Per (task, tag, noise, snr, pipe): AUROC, AUPRC, dAUROC vs CLEAN raw and vs NOISY raw of the same condition
(paired bootstrap, 95% CI), sens / spec / PPV / FP at both thresholds.
Writes batch/noise_scores_summary.csv, batch/noise_scores_probs.csv, batch/noise_scores_summary.txt,
       batch/noise_auroc.png, batch/noise_operating.png

Usage: sbatch score_noise.sbatch     (GPU, ~15-30 min, mostly bootstrap)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_batch as sb          # noqa: E402  score(), at_threshold(), paired_delta_auroc(), build_labels(), EXPECTED
import recalibrate as rc          # noqa: E402  thr_balanced(), thr_sensitivity(), labels_for_fold()

PIPES = ["med06", "med03", "hp05zp", "hp05c"]
NOISES = ["bw", "em", "ma", "mix"]
SNRS = ["0", "6", "12", "18", "24"]
CONDS = [("none", "inf")] + [(n, s) for n in NOISES for s in SNRS]
TASKS = ["binary", "af"]


def load_model(c, device):
    import torch
    m = sb.build_model(c["model"], 1, 1)
    m.load_state_dict(torch.load(sb.EXPORT / c["weights"], map_location="cpu"))
    return m.to(device).eval()


def load_npz(path):
    z = np.load(path)
    return pd.DataFrame(z["X"], index=z["ids"])


def predict(model, df, ids, c, device):
    return sb.score(model, df.loc[ids].to_numpy(dtype=np.float32), c, device)


def main(a):
    batch = Path(a.batch)
    cfg = json.loads((sb.EXPORT / "config.json").read_text())
    cfg06 = json.loads((sb.EXPORT / "config_clean06.json").read_text())
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    print(f"device: {device}", flush=True)

    db = pd.read_csv(sb.PTB / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(sb.PTB / "scp_statements.csv", index_col=0)
    log = pd.read_csv(Path(a.clean_root).expanduser() / "records500" / "clean_dataset_log.csv")
    flagged = set(db.index[db.filename_hr.isin(set("records500/" + log[log.note.notna()].record))])
    lab9, lab10 = rc.labels_for_fold(db, scp, 9), sb.build_labels()

    tags = [t for t in a.tags if (batch / f"ns_{t}_none_inf_med06.npz").exists()]
    lines, rows, probs = [], [], []
    for task in TASKS:
        c = cfg[task]
        model = load_model(c, device)
        lines.append(f"\n================ {task} ================")

        # ---------------- thresholds on clean fold 9
        yv = lab9[task]; iv = yv.index; y9 = yv.to_numpy(); keep9 = ~iv.isin(flagged)
        f9raw = load_npz(batch / "ns_f9_raw.npz")
        p = predict(model, f9raw, iv, c, device)
        th = {"raw": (rc.thr_balanced(y9, p), rc.thr_sensitivity(y9, p, rc.HIGH_SENS_TARGET))}
        ok9 = abs(th["raw"][0] - c["threshold_balanced"]) < 1e-3 and abs(th["raw"][1] - c["threshold_high_sens"]) < 1e-3
        lines.append(f"SELF-CHECK fold-9 raw thresholds {th['raw'][0]:.4f} / {th['raw'][1]:.4f} vs config.json "
                     f"{c['threshold_balanced']:.4f} / {c['threshold_high_sens']:.4f}: {'PASS' if ok9 else 'FAIL'}")
        th["med06"] = (cfg06[task]["threshold_balanced"], cfg06[task]["threshold_high_sens"])
        for pipe in PIPES[1:]:
            f = batch / f"ns_f9_orig_{pipe}.npz"
            if not f.exists():
                continue
            p = predict(model, load_npz(f), iv, c, device)
            m = keep9 & ~np.isnan(p)
            th[pipe] = (rc.thr_balanced(y9[m], p[m]), rc.thr_sensitivity(y9[m], p[m], rc.HIGH_SENS_TARGET))
        lines.append("thresholds (balanced / high_sens): " +
                     ", ".join(f"{k} {v[0]:.4f} / {v[1]:.4f}" for k, v in th.items()))

        # ---------------- fold 10
        yt = lab10[task]
        ref = load_npz(batch / "ns_none_inf_raw.npz")
        ids = yt.index[yt.index.isin(ref.index)]
        y = yt.loc[ids].to_numpy()
        P = {}
        for kind, snr in CONDS:
            P[("raw", kind, snr, "raw")] = predict(model, load_npz(batch / f"ns_{kind}_{snr}_raw.npz"), ids, c, device)
            for tag in tags:
                for pipe in PIPES:
                    f = batch / f"ns_{tag}_{kind}_{snr}_{pipe}.npz"
                    if f.exists() and pipe in th:
                        P[(tag, kind, snr, pipe)] = predict(model, load_npz(f), ids, c, device)
            print(f"  {task}: scored {kind} {snr}", flush=True)

        # self-check 2: clean raw = Stage 1
        e = sb.EXPECTED[task]; p0 = P[("raw", "none", "inf", "raw")]
        m = sb.at_threshold(y, p0, c["threshold_balanced"])
        chk = [len(y) == e["n"], int(y.sum()) == e["pos"], all(m[k] == e[k] for k in ("tp", "fn", "fp", "tn"))]
        if e["auroc"] is not None:
            chk.append(abs(roc_auc_score(y, p0) - e["auroc"]) < 0.001)
        lines.append(f"SELF-CHECK clean raw vs Stage 1 (n {len(y)}, pos {int(y.sum())}): {'PASS' if all(chk) else 'FAIL ' + str(m)}")

        for (tag, kind, snr, pipe), p in P.items():
            noisy_raw = P[("raw", kind, snr, "raw")]
            ok = ~np.isnan(p) & ~np.isnan(p0) & ~np.isnan(noisy_raw)
            r = dict(task=task, tag=tag, noise=kind, snr=snr, pipe=pipe, n=int(ok.sum()),
                     auroc=roc_auc_score(y[ok], p[ok]), auprc=average_precision_score(y[ok], p[ok]))
            if (tag, kind, snr, pipe) != ("raw", "none", "inf", "raw"):
                r["d_clean"], r["d_clean_lo"], r["d_clean_hi"] = sb.paired_delta_auroc(y[ok], p0[ok], p[ok], a.n_boot)
            if pipe != "raw":
                r["d_noisyraw"], r["d_noisyraw_lo"], r["d_noisyraw_hi"] = \
                    sb.paired_delta_auroc(y[ok], noisy_raw[ok], p[ok], a.n_boot)
            for name, t in zip(("bal", "hs"), th[pipe]):
                mm = sb.at_threshold(y[ok], p[ok], t)
                r.update({f"{name}_sens": mm["sens"], f"{name}_spec": mm["spec"], f"{name}_ppv": mm["ppv"],
                          f"{name}_fp": mm["fp"], f"{name}_tp": mm["tp"]})
            rows.append(r)
        probs.append(pd.DataFrame({"task": task, "ecg_id": ids, "y": y,
                                   **{"p_" + "_".join(k): v for k, v in P.items()}}))

    S = pd.DataFrame(rows)
    S.to_csv(batch / "noise_scores_summary.csv", index=False)
    pd.concat(probs).to_csv(batch / "noise_scores_probs.csv", index=False)

    # ---------------- readable tables
    order = ["raw"] + PIPES
    cols = [f"{n} {s}" for n in NOISES for s in SNRS[::-1]]
    for task in TASKS:
        for metric, title in (("auroc", "AUROC"), ("bal_sens", "balanced sensitivity"), ("bal_spec", "balanced specificity"),
                              ("hs_spec", "specificity at the high-sens threshold")):
            T = S[(S.task == task) & S.tag.isin(["raw", "orig"])].copy()
            T["cond"] = T.noise + " " + T.snr
            clean = T[T.noise == "none"].set_index("pipe")[metric]
            tab = T[T.noise != "none"].pivot_table(index="pipe", columns="cond", values=metric).reindex(index=order, columns=cols)
            tab.insert(0, "clean", clean.reindex(order))
            lines.append(f"\n{task}, {title} (orig detector; columns noise + SNR dB, nst convention)")
            lines.append(tab.round(3).to_string())
        R = S[(S.task == task) & (S["pipe"] == "med06") & (S.tag.isin(["orig", "reset"]))]
        if R.tag.nunique() == 2:
            dd = R.pivot_table(index=["noise", "snr"], columns="tag", values="auroc")
            lines.append(f"\n{task}, med06 AUROC: reset patch minus original, max |diff| {np.abs(dd['reset'] - dd['orig']).max():.4f}")
    txt = "\n".join(lines)
    print(txt)
    (batch / "noise_scores_summary.txt").write_text(txt + "\n")
    plots(S, batch)
    print(f"\nsaved noise_scores_summary.csv/.txt, noise_scores_probs.csv, noise_auroc.png, noise_operating.png in {batch}")


# ----------------------------------------------------------------------------- plots
STYLE = {"raw": ("#7f7f7f", "-", "noisy raw (no cleaning)"), "med06": ("#1f77b4", "-", "0.6 s median (team)"),
         "med03": ("#d62728", "-", "0.3 s median"), "hp05zp": ("#2ca02c", "--", "0.5 Hz HP zero-phase"),
         "hp05c": ("#2ca02c", "-", "0.5 Hz HP causal")}


def plots(S, batch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [0, 6, 12, 18, 24, 30]
    xl = ["0", "6", "12", "18", "24", "clean"]

    def line(ax, task, noise, metric):
        for pipe in ["raw"] + PIPES:
            tag = "raw" if pipe == "raw" else "orig"
            D = S[(S.task == task) & (S["pipe"] == pipe) & (S.tag == tag)]
            y = [D[(D.noise == noise) & (D.snr == s)][metric].mean() for s in SNRS] + \
                [D[D.noise == "none"][metric].mean()]
            col, ls, lab = STYLE[pipe]
            ax.plot(xs, y, ls, color=col, marker="o", ms=3, lw=1.4, label=lab)
        ax.set_xticks(xs); ax.set_xticklabels(xl); ax.grid(alpha=0.3)

    fig, ax = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    for i, task in enumerate(TASKS):
        for j, noise in enumerate(NOISES):
            line(ax[i, j], task, noise, "auroc")
            ax[i, j].set_title(f"{task}, {noise}"); ax[i, 0].set_ylabel("AUROC")
    for j in range(4):
        ax[1, j].set_xlabel("SNR (dB, nst convention)")
    ax[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Noise stress test, PTB-XL fold 10, V2-V1 seed-0 models (thresholds not involved)")
    fig.tight_layout(); fig.savefig(batch / "noise_auroc.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(4, 4, figsize=(16, 13), sharex=True)
    for i, (task, metric, lab) in enumerate([("binary", "bal_sens", "binary sens (balanced)"),
                                             ("binary", "bal_spec", "binary spec (balanced)"),
                                             ("af", "bal_sens", "AF sens (balanced)"),
                                             ("af", "bal_fp", "AF false positives (balanced)")]):
        for j, noise in enumerate(NOISES):
            line(ax[i, j], task, noise, metric)
            ax[i, 0].set_ylabel(lab)
            if i == 0:
                ax[i, j].set_title(noise)
    for j in range(4):
        ax[3, j].set_xlabel("SNR (dB, nst convention)")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Operating points: thresholds chosen once on clean fold 9 per pipeline")
    fig.tight_layout(); fig.savefig(batch / "noise_operating.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=str(Path.home() / "dsp_test" / "batch"))
    ap.add_argument("--clean_root", default=str(Path.home() / "data" / "ptb-xl-clean06"))
    ap.add_argument("--tags", nargs="+", default=["orig", "reset"])
    ap.add_argument("--n_boot", type=int, default=500)
    main(ap.parse_args())
