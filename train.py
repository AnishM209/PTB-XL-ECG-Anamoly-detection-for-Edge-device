"""
Train and evaluate one configuration on PTB-XL. Used for steps 2 to 4 of Stage 1.

Protocol (follows the PTB-XL benchmark of Strodthoff et al., which used 100 Hz;
this project uses 500 Hz to match the hardware sampling rate):
  - official folds: 1-8 train, 9 val (model selection), 10 test (reported once)
  - training on random crops of length --win_sec
  - evaluation: sliding windows (50% overlap) over the full 10 s record,
    sigmoid outputs averaged per record
  - per-lead standardisation using training-set statistics only

Step 2, 12-lead sanity reproduction:
    python train.py --task superclass --leads all --fs 500 --win_sec 2.5
Smoke test (CPU, a few minutes, checks the code runs end to end):
    python train.py --task superclass --leads all --fs 500 --smoke

Leads: "all", one lead ("V2"), a comma list ("V1,V2,V3"), or a derived bipolar
lead written A-B ("V2-V1" = V2 minus V1). Subtracting two chest leads cancels the
Wilson central terminal exactly, so V2-V1 is the voltage between the V1 and V2
electrode sites, the same kind of signal a two-electrode chest device records.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import decimate
from sklearn.metrics import average_precision_score, roc_auc_score

import ptbxl_data as P
from models import build_model, count_params


# ------------------------------------------------------------------- tasks
def build_task(db, scp, task):
    """Label DataFrame indexed by ecg_id; records without a valid label are dropped."""
    if task == "superclass":
        Y = P.superclass_labels(db, scp)
        return Y[Y.sum(axis=1) > 0]
    if task == "binary":
        return P.binary_labels(P.superclass_labels(db, scp)).dropna().to_frame()
    if task == "af":
        R, _ = P.rhythm_labels(db, scp)
        af = ((R.get("AFIB", 0) + R.get("AFLT", 0)) > 0).astype(np.float32).rename("AF")
        return af[R.sum(axis=1) > 0].to_frame()  # needs at least one rhythm statement
    raise ValueError(task)


# ----------------------------------------------------------------- signals
def load_signals(cache_dir, fs, leads):
    """(N, n_leads, T) float32 in RAM. 100 Hz is derived once from the 500 Hz cache."""
    cache_dir = Path(cache_dir)
    ids = np.load(cache_dir / "ecg_ids.npy")
    if fs == 500:
        X = np.load(cache_dir / "signals_500hz.npy", mmap_mode="r")
    else:
        f = cache_dir / "signals_100hz.npy"
        if not f.exists():
            print("Creating 100 Hz cache (one-time, anti-aliased decimation by 5)...")
            X500 = np.load(cache_dir / "signals_500hz.npy", mmap_mode="r")
            out = np.lib.format.open_memmap(f, mode="w+", dtype=np.float32,
                                            shape=(len(X500), 12, P.N_SAMPLES // 5))
            for s in range(0, len(X500), 1000):
                out[s:s + 1000] = decimate(np.asarray(X500[s:s + 1000]), 5,
                                           axis=-1, zero_phase=True)
            out.flush()
            del out
        X = np.load(f, mmap_mode="r")
    return select_leads(X, leads), ids


def select_leads(X, leads):
    """(N, 12, T) -> (N, n_leads, T). A token "A-B" gives lead A minus lead B."""
    if leads is None:
        return np.ascontiguousarray(X, dtype=np.float32)
    chans = []
    for tok in leads:
        if "-" in tok:
            a, b = (P.LEADS.index(t.strip().upper()) for t in tok.split("-"))
            chans.append(np.asarray(X[:, a, :], dtype=np.float32)
                         - np.asarray(X[:, b, :], dtype=np.float32))
        else:
            chans.append(np.asarray(X[:, P.LEADS.index(tok.strip().upper()), :], dtype=np.float32))
    return np.ascontiguousarray(np.stack(chans, axis=1))


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, win):
        self.X, self.Y, self.win = X, Y, win

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        T = self.X.shape[-1]
        s = int(torch.randint(0, T - self.win + 1, (1,)))
        return torch.from_numpy(self.X[i, :, s:s + self.win]), torch.from_numpy(self.Y[i])


# --------------------------------------------------------------- evaluation
@torch.no_grad()
def predict(model, X, win, device, bs=256):
    model.eval()
    T = X.shape[-1]
    starts = list(range(0, T - win + 1, max(1, win // 2)))
    if starts[-1] != T - win:
        starts.append(T - win)
    out = []
    for b in range(0, len(X), bs):
        xb = torch.from_numpy(X[b:b + bs]).to(device)
        p = torch.stack([torch.sigmoid(model(xb[:, :, s:s + win])) for s in starts]).mean(0)
        out.append(p.float().cpu().numpy())
    return np.concatenate(out)


def compute_metrics(Y, Pr, names):
    def safe(fn, y, p):
        return float(fn(y, p)) if len(np.unique(y)) == 2 else float("nan")
    auroc = {n: safe(roc_auc_score, Y[:, j], Pr[:, j]) for j, n in enumerate(names)}
    auprc = {n: safe(average_precision_score, Y[:, j], Pr[:, j]) for j, n in enumerate(names)}
    return {"macro_auroc": float(np.nanmean(list(auroc.values()))),
            "macro_auprc": float(np.nanmean(list(auprc.values()))),
            "auroc": auroc, "auprc": auprc}


# ------------------------------------------------------------------- main
def main(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not (a.smoke or a.allow_cpu):
        raise SystemExit("No usable GPU found; refusing a full run on CPU. "
                         "Check the CUDA warning above, or pass --allow_cpu.")
    leads = None if a.leads == "all" else a.leads.split(",")
    lead_tag = "12lead" if leads is None else "+".join(leads)
    run = Path(a.runs) / f"{a.task}_{a.model}_{lead_tag}_{a.fs}hz_{a.win_sec}s_seed{a.seed}"
    if a.smoke:
        run = run.with_name(run.name + "_smoke")
    if a.skip_done and not a.smoke and (run / "results.json").exists():
        print(f"Already finished, skipping: {run}")
        return
    run.mkdir(parents=True, exist_ok=True)

    db, scp = P.load_metadata(a.root)
    if a.exclude_paced:
        db = db[db["pacemaker"].isna()]
    Y_df = build_task(db, scp, a.task)
    names = list(Y_df.columns)

    X_all, ids = load_signals(a.cache, a.fs, leads)
    row = {e: i for i, e in enumerate(ids)}

    data = {}
    for split, folds in P.FOLDS.items():
        sel = Y_df.index[db.loc[Y_df.index, "strat_fold"].isin(folds)].to_numpy()
        if a.smoke:
            sel = sel[: 512 if split == "train" else 256]
        data[split] = (X_all[[row[e] for e in sel]], np.array(Y_df.loc[sel].to_numpy(np.float32)))
    del X_all

    mu = data["train"][0].mean(axis=(0, 2), keepdims=True)
    sd = data["train"][0].std(axis=(0, 2), keepdims=True) + 1e-6
    data = {k: (((x - mu) / sd).astype(np.float32), y) for k, (x, y) in data.items()}
    print(f"device={device}  task={a.task}  classes={names}  leads={lead_tag}  fs={a.fs}")
    print("records:", {k: len(v[1]) for k, v in data.items()})

    win = int(round(a.win_sec * a.fs))
    model = build_model(a.model, data["train"][0].shape[1], len(names)).to(device)
    n_params = count_params(model)
    print(f"model={a.model}  params={n_params:,}  window={win} samples")

    loader = torch.utils.data.DataLoader(
        CropDataset(*data["train"], win), batch_size=a.bs, shuffle=True,
        num_workers=a.workers, drop_last=True, pin_memory=device.type == "cuda")
    epochs = 1 if a.smoke else a.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, epochs=epochs,
                                                steps_per_epoch=len(loader))
    loss_fn = nn.BCEWithLogitsLoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, best_epoch, log = -1.0, -1, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(xb)
        val = compute_metrics(data["val"][1], predict(model, data["val"][0], win, device), names)
        log.append({"epoch": ep, "train_loss": total / len(loader.dataset),
                    "val_macro_auroc": val["macro_auroc"]})
        print(f"epoch {ep:3d}  loss {log[-1]['train_loss']:.4f}  "
              f"val macro AUROC {val['macro_auroc']:.4f}")
        if val["macro_auroc"] > best:
            best, best_epoch = val["macro_auroc"], ep
            torch.save(model.state_dict(), run / "best.pt")

    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    val = compute_metrics(data["val"][1], predict(model, data["val"][0], win, device), names)
    test = compute_metrics(data["test"][1], predict(model, data["test"][0], win, device), names)
    results = {"config": vars(a) | {"root": str(a.root), "cache": str(a.cache), "runs": str(a.runs)},
               "params": n_params, "best_epoch": best_epoch,
               "train_minutes": round((time.time() - t0) / 60, 1), "val": val, "test": test}
    (run / "results.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(log).to_csv(run / "train_log.csv", index=False)

    print(f"\nBest epoch {best_epoch}.  TEST macro AUROC {test['macro_auroc']:.4f}  "
          f"macro AUPRC {test['macro_auprc']:.4f}")
    for n in names:
        print(f"  {n:<6} AUROC {test['auroc'][n]:.4f}  AUPRC {test['auprc'][n]:.4f}")
    print(f"Saved to {run}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.home() / "data/ptb-xl")
    ap.add_argument("--cache", type=Path, default=Path.home() / "stage1/stage1_out/cache")
    ap.add_argument("--runs", type=Path, default=Path.home() / "stage1/runs")
    ap.add_argument("--task", choices=["superclass", "binary", "af"], default="superclass")
    ap.add_argument("--leads", default="all", help='"all", "V2", "V1,V2,V3", or bipolar "V2-V1" (V2 minus V1)')
    ap.add_argument("--fs", type=int, choices=[100, 500], default=500)
    ap.add_argument("--win_sec", type=float, default=2.5)
    ap.add_argument("--model", default="resnet1d_wang")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude_paced", action="store_true")
    ap.add_argument("--allow_cpu", action="store_true", help="permit a full run without a GPU")
    ap.add_argument("--skip_done", action="store_true", help="exit if this run already has results")
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 1 epoch")
    main(ap.parse_args())
