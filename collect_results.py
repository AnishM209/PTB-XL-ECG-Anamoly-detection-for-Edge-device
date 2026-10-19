"""
Summarise every finished run in ~/stage1/runs into one table: mean and standard
deviation of test metrics across seeds, per configuration.

Usage:  python collect_results.py [--task superclass]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=Path, default=Path.home() / "stage1/runs")
ap.add_argument("--task", default=None)
a = ap.parse_args()

rows = []
for f in sorted(a.runs.glob("*/results.json")):
    if f.parent.name.endswith("_smoke"):
        continue
    r = json.loads(f.read_text())
    c = r["config"]
    if a.task and c["task"] != a.task:
        continue
    row = {"task": c["task"], "model": c["model"], "leads": c["leads"], "fs": c["fs"],
           "win_sec": c["win_sec"], "seed": c["seed"], "params": r["params"],
           "minutes": r["train_minutes"], "test_macro_auroc": r["test"]["macro_auroc"],
           "test_macro_auprc": r["test"]["macro_auprc"]}
    row.update({f"auroc_{k}": v for k, v in r["test"]["auroc"].items()})
    rows.append(row)

if not rows:
    raise SystemExit("no finished runs found")
df = pd.DataFrame(rows)
df.to_csv(a.runs / "all_runs.csv", index=False)

keys = ["task", "model", "leads", "fs", "win_sec"]
metrics = [c for c in df.columns if c.startswith(("test_", "auroc_"))]
g = df.groupby(keys)
summary = g[metrics].agg(["mean", "std"]).round(4)
summary.insert(0, ("n_seeds", ""), g.size())
summary.insert(1, ("params", ""), g["params"].first())
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
print(summary)
summary.to_csv(a.runs / "summary.csv")
print(f"\nWritten: {a.runs / 'all_runs.csv'} and {a.runs / 'summary.csv'}")
