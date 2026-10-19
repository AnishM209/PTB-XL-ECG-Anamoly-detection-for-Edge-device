"""
Package trained Stage 1 models for inference on a laptop.

For the V2-V1 lead and seed 0, this collects the binary (normal vs abnormal) and
AF models into ~/export/ together with everything needed to use them:
  models.py                     the architecture code
  binary_V2-V1_seed0.pt         trained weights
  af_V2-V1_seed0.pt             trained weights
  config.json                   window length, sampling rate, the training-set
                                mean and std used to standardise the input, and
                                the decision thresholds chosen on validation
Run on the cluster from ~/stage1:   python export_model.py
"""
import json
import shutil
from pathlib import Path

import pandas as pd

import ptbxl_data as P
import train as T

HOME = Path.home()
RUNS, EVAL, OUT = HOME / "stage1/runs", HOME / "stage1/eval", HOME / "export"
MODEL, LEADS, SEED = "resnet1d_wang_s5", "V2-V1", 0
NAMES = {"binary": ("Normal", "Abnormal"), "af": ("No AF", "AF")}

OUT.mkdir(exist_ok=True)
shutil.copy(HOME / "stage1/models.py", OUT)
config = {}
for task, (neg, pos) in NAMES.items():
    runs = [d for d in RUNS.glob(f"{task}_{MODEL}_{LEADS}_*_seed{SEED}") if (d / "results.json").exists()]
    if len(runs) != 1:
        raise SystemExit(f"Expected one finished run for {task}, found {runs}")
    run = runs[0]
    cfg = json.loads((run / "results.json").read_text())["config"]

    # Training-set statistics, computed exactly as train.py does.
    db, scp = P.load_metadata(cfg["root"])
    if cfg.get("exclude_paced"):
        db = db[db["pacemaker"].isna()]
    Y = T.build_task(db, scp, task)
    X, ids = T.load_signals(cfg["cache"], cfg["fs"], cfg["leads"].split(","))
    row = {e: i for i, e in enumerate(ids)}
    sel = Y.index[db.loc[Y.index, "strat_fold"].isin(P.FOLDS["train"])].to_numpy()
    Xtr = X[[row[e] for e in sel]]
    mu = Xtr.mean(axis=(0, 2))
    sd = Xtr.std(axis=(0, 2)) + 1e-6
    del X, Xtr

    # Thresholds chosen on the validation fold by evaluate.py.
    th = {"balanced": None, "high_sens": None}
    f = EVAL / f"{task}_{MODEL}" / "per_seed.csv"
    if f.exists():
        r = pd.read_csv(f).query("leads == @LEADS and seed == @SEED")
        if len(r):
            th = {k: float(r.iloc[0][f"{k}_threshold"]) for k in th}

    weights = f"{task}_{LEADS}_seed{SEED}.pt"
    shutil.copy(run / "best.pt", OUT / weights)
    config[task] = {"weights": weights, "model": cfg["model"], "leads": cfg["leads"],
                    "fs": cfg["fs"], "win_sec": cfg["win_sec"],
                    "mu": [float(v) for v in mu], "sd": [float(v) for v in sd],
                    "threshold_balanced": th["balanced"], "threshold_high_sens": th["high_sens"],
                    "negative": neg, "positive": pos, "trained_run": run.name}
    print(f"{task}: mu={mu[0]:.5f} mV  sd={sd[0]:.5f} mV  thresholds={th}")

(OUT / "config.json").write_text(json.dumps(config, indent=2))
print(f"\nExported to {OUT}:", sorted(p.name for p in OUT.iterdir()))
