"""Shared harness for model wrappers.

Provides: generic CSV windowed dataset, train/eval loop, metrics, and a uniform
CLI `run(build_fn, model_name, defaults)`. Models stay tiny — they only supply a
`build(cfg)` factory and their default hyper-parameters.

Forecasting setup (matches the LTSF/TSLib convention we verified):
  * sliding window of `seq_len` past steps -> predict next `pred_len` steps
  * StandardScaler fit on the TRAIN split only, applied to all splits
  * MSE/MAE reported in scaled space (standard for these benchmarks)
  * features:  S  = univariate target only
               MS = all feature columns in, predict the target column only
               M  = all columns in and out
The target column is moved to the last position (f_dim = -1), as in the originals.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------------------------------------------------------- data
def _numeric_feature_columns(df):
    drop = {c for c in df.columns if c.lower() in ("date", "datetime", "time", "timestamp")}
    cols = [c for c in df.columns if c not in drop]
    num = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    return num


class WindowDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, target_idx, features):
        self.data = data                      # [T, C] float32, already scaled
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_idx = target_idx
        self.features = features

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, i):
        s = i
        e = i + self.seq_len
        x = self.data[s:e]                            # [seq_len, C]
        y_full = self.data[e:e + self.pred_len]       # [pred_len, C]
        return torch.from_numpy(x), torch.from_numpy(y_full)


def load_csv_splits(cfg, df=None):
    """Returns (train_ds, val_ds, test_ds, n_features, target_idx, scaler_stats).

    If `df` is given it is used directly (in-process, no disk read); otherwise the
    CSV at cfg.data_path is read.
    """
    if df is None:
        df = pd.read_csv(cfg.data_path)
    else:
        df = df.copy()
    num_cols = _numeric_feature_columns(df)
    if not num_cols:
        raise ValueError(f"No numeric columns in {cfg.data_path}")

    target = cfg.target if cfg.target in num_cols else num_cols[-1]

    if cfg.features == "S":
        cols = [target]
    else:  # M or MS use all numeric columns, target last
        cols = [c for c in num_cols if c != target] + [target]
    df = df[cols].astype("float32")
    target_idx = len(cols) - 1

    arr = df.values  # [T, C]
    n = len(arr)
    n_train = int(n * cfg.train_ratio)
    n_test = int(n * cfg.test_ratio)
    n_val = n - n_train - n_test

    # StandardScaler on train only
    train_slice = arr[:n_train]
    mean = train_slice.mean(0, keepdims=True)
    std = train_slice.std(0, keepdims=True)
    std[std == 0] = 1.0
    if not cfg.no_scale:
        arr = (arr - mean) / std

    # borders mirror the LTSF loader: val/test windows can look back into prior split
    b1 = [0, n_train - cfg.seq_len, n_train + n_val - cfg.seq_len]
    b2 = [n_train, n_train + n_val, n]
    train = WindowDataset(arr[b1[0]:b2[0]], cfg.seq_len, cfg.pred_len, target_idx, cfg.features)
    val = WindowDataset(arr[b1[1]:b2[1]], cfg.seq_len, cfg.pred_len, target_idx, cfg.features)
    test = WindowDataset(arr[b1[2]:b2[2]], cfg.seq_len, cfg.pred_len, target_idx, cfg.features)
    stats = dict(columns=cols, target=target, n=n, n_train=n_train, n_val=n_val,
                 n_test=n_test, mean=mean.ravel().tolist(), std=std.ravel().tolist())
    return train, val, test, len(cols), target_idx, stats


# ----------------------------------------------------------------------------- loop
def _slice_target(out, y, features, target_idx):
    if features == "MS":
        yt = y[:, :, target_idx:target_idx + 1]
        # model may already output a single (mixed) target channel, or all channels
        ot = out if out.shape[-1] == 1 else out[:, :, target_idx:target_idx + 1]
        return ot, yt
    return out, y  # M / S use all channels


def _run_epoch(model, loader, criterion, optim, device, cfg, train):
    model.train(train)
    losses = []
    for x, y in loader:
        x = x.float().to(device)
        y = y.float().to(device)
        if train:
            optim.zero_grad()
        out = model(x)
        out, yt = _slice_target(out, y, cfg.features, cfg.target_idx)
        loss = criterion(out, yt)
        if train:
            loss.backward()
            clip = getattr(cfg, "clip_grad", 0.0)
            if clip and clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optim.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x = x.float().to(device)
        out = model(x)
        out, yt = _slice_target(out, y.float().to(device), cfg.features, cfg.target_idx)
        preds.append(out.cpu().numpy())
        trues.append(yt.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    mse = float(np.mean((preds - trues) ** 2))
    mae = float(np.mean(np.abs(preds - trues)))
    # R^2 (coefficient of determination) on the target test predictions
    ss_res = float(np.sum((trues - preds) ** 2))
    ss_tot = float(np.sum((trues - trues.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    # directional accuracy: do predicted step-to-step moves match the true ones?
    if preds.shape[1] >= 2:
        dp, dt = np.diff(preds, axis=1), np.diff(trues, axis=1)
        da = float(np.mean(np.sign(dp) == np.sign(dt)))
    else:
        da = float("nan")
    return dict(mse=mse, mae=mae, r2=r2, da=da, preds=preds)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------------- CLI
def base_parser(model_name, defaults):
    p = argparse.ArgumentParser(description=f"{model_name} forecaster")
    p.add_argument("--data_path", required=True, help="path to input CSV")
    p.add_argument("--target", default=defaults.get("target", "Close"))
    p.add_argument("--features", default=defaults.get("features", "MS"), choices=["S", "MS", "M"])
    p.add_argument("--seq_len", type=int, default=defaults.get("seq_len", 96))
    p.add_argument("--pred_len", type=int, default=defaults.get("pred_len", 24))
    p.add_argument("--epochs", type=int, default=defaults.get("epochs", 20))
    p.add_argument("--batch_size", type=int, default=defaults.get("batch_size", 32))
    p.add_argument("--lr", type=float, default=defaults.get("lr", 0.005))
    p.add_argument("--patience", type=int, default=defaults.get("patience", 5))
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--no_scale", action="store_true")
    p.add_argument("--individual", action="store_true")
    p.add_argument("--mix_channels", action="store_true",
                   help="combine per-channel forecasts into target (lets features matter)")
    p.add_argument("--kernel_size", type=int, default=defaults.get("kernel_size", 25))
    p.add_argument("--clip_grad", type=float, default=defaults.get("clip_grad", 0.0),
                   help="max grad-norm for clipping (0 = off); improves stability")
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default=None, help="if set, save metrics.json + pred.npy here")
    p.add_argument("--quiet", action="store_true")
    return p


def run(build_fn, model_name, defaults=None):
    """Parse CLI, train, evaluate, optionally save. Returns metrics dict."""
    defaults = defaults or {}
    cfg = base_parser(model_name, defaults).parse_args()
    return run_cfg(build_fn, model_name, cfg)


def run_cfg(build_fn, model_name, cfg, df=None):
    """Same as run() but takes a ready cfg (and optional in-memory df).

    Used by the study to train on a preprocessed DataFrame without disk I/O.
    """
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    train_ds, val_ds, test_ds, n_feat, target_idx, stats = load_csv_splits(cfg, df=df)
    cfg.enc_in = n_feat
    cfg.target_idx = target_idx

    model = build_fn(cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    best_val = float("inf")
    best_state = None
    bad = 0
    t0 = time.time()
    for ep in range(cfg.epochs):
        tr = _run_epoch(model, train_loader, criterion, optim, device, cfg, train=True)
        vl = evaluate(model, val_loader, device, cfg)["mse"]
        if not cfg.quiet:
            print(f"  epoch {ep+1:>3d}/{cfg.epochs}  train {tr:.4f}  val {vl:.4f}")
        if vl < best_val - 1e-6:
            best_val, best_state, bad = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    ev = evaluate(model, test_loader, device, cfg)
    mse, mae, r2, da, preds = ev["mse"], ev["mae"], ev["r2"], ev["da"], ev["preds"]
    elapsed = round(time.time() - t0, 1)
    metrics = dict(model=model_name, data=str(getattr(cfg, "data_path", "<df>")), features=cfg.features,
                   seq_len=cfg.seq_len, pred_len=cfg.pred_len, n_features=n_feat,
                   target=stats["target"], mse=mse, mae=mae, r2=r2, da=da, val_mse=best_val,
                   elapsed_s=elapsed, epochs_ran=ep + 1, seed=cfg.seed)
    if not cfg.quiet:
        print(f"{model_name}  MSE {mse:.4f}  MAE {mae:.4f}  R2 {r2:.4f}  DA {da:.3f}  ({elapsed}s, {ep+1} epochs)")

    if cfg.out_dir:
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps({**metrics, **stats}, indent=2))
        np.save(out / "pred.npy", preds)
    return metrics
