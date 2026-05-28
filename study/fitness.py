"""Fitness for the PH evolutionary study.

fitness(individual) = mean test-MSE of DLinear (channel-mixing) trained on each of
the 12 stock datasets after applying the genome's PH preprocessing. Lower is better.

For every individual we also record MAE, DA (directional accuracy), R2 and wall-clock
time (per dataset + averaged), appended to a results table. Deterministic seed ->
identical genomes are cached (not retrained).
"""
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT))

import _common                                   # noqa: E402  (models/_common.py)
from preprocessing import apply as apply_pipeline  # noqa: E402
from study import genome as G                    # noqa: E402

PENALTY = 10.0
METRICS = ["mse", "mae", "da", "r2"]

# model registry: per-model build factory + sensible defaults (lr / epochs / channel-mix).
# `mix_channels` only applies to DLinear; the TSLib models mix channels by construction.
MODELS = {
    "dlinear":   dict(build="dlinear.model:build",   lr=0.005,  epochs=10, mix=True),
    "patchtst":  dict(build="patchtst.model:build",  lr=0.0001, epochs=10, mix=False),
    "timemixer": dict(build="timemixer.model:build", lr=0.01,   epochs=10, mix=False),
}


def _load_build(spec):
    import importlib
    m, a = spec.split(":")
    return getattr(importlib.import_module(m), a)


class StudyConfig:
    def __init__(self, datasets, model="dlinear", seq_len=64, pred_len=5, target="Close",
                 epochs=None, batch_size=32, lr=None, patience=4, seed=2021, device=None,
                 log_path=None, results_path=None):
        if model not in MODELS:
            raise ValueError(f"model {model!r}; choose from {list(MODELS)}")
        reg = MODELS[model]
        self.model = model
        self.build_fn = _load_build(reg["build"])
        self.mix_channels = bool(reg["mix"])
        self.datasets = [Path(d) for d in datasets]
        self.seq_len, self.pred_len, self.target = seq_len, pred_len, target
        self.epochs = int(epochs if epochs is not None else reg["epochs"])
        self.batch_size, self.patience = batch_size, patience
        self.lr = float(lr if lr is not None else reg["lr"])
        self.seed = seed
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.log_path = Path(log_path) if log_path else None        # full per-eval JSONL
        self.results_path = Path(results_path) if results_path else None  # tidy CSV table
        self._raw = {d: pd.read_csv(d) for d in self.datasets}
        self._feat_cache, self._fit_cache = {}, {}
        self.n_evals = 0
        self.current_gen = 0          # set by the EA so each results row knows its generation


def _model_cfg(sc):
    return types.SimpleNamespace(
        seq_len=sc.seq_len, pred_len=sc.pred_len, features="MS", target=sc.target,
        epochs=sc.epochs, batch_size=sc.batch_size, lr=sc.lr, patience=sc.patience,
        train_ratio=0.7, test_ratio=0.2, no_scale=False, individual=False,
        mix_channels=sc.mix_channels, kernel_size=25, clip_grad=4.0, seed=sc.seed,
        device=sc.device, out_dir=None, quiet=True)


def _steps_hash(steps):
    return hashlib.md5(json.dumps(steps, sort_keys=True, default=str).encode()).hexdigest()


def _processed(sc, dataset, steps):
    key = (str(dataset), _steps_hash(steps))
    if key not in sc._feat_cache:
        if len(sc._feat_cache) > 64:
            sc._feat_cache.clear()
        sc._feat_cache[key] = apply_pipeline(sc._raw[dataset], steps, target=sc.target)
    return sc._feat_cache[key]


def _write_results_row(sc, rec):
    """Append one tidy row (key genes + averaged metrics + time) to results.csv."""
    if not sc.results_path:
        return
    g = rec["genome"]
    row = dict(eval=rec["eval"], gen=getattr(sc, "current_gen", 0),
               point_cloud=g["point_cloud"], filtration=g["filtration"],
               vectorizer=g["vectorizer"], homology_dim=g["homology_dim"],
               ph_source=g["ph_source"], window=g["window"], dimension=g["dimension"],
               delay=g["delay"], maxdim=g["maxdim"], metric=g["metric"], resolution=g["resolution"],
               mse=rec["mse"], mae=rec["mae"], da=rec["da"], r2=rec["r2"], seconds=rec["seconds"])
    header = list(row)
    new = not sc.results_path.exists()
    with open(sc.results_path, "a", newline="") as f:
        if new:
            f.write(",".join(header) + "\n")
        f.write(",".join(f"{row[k]:.6f}" if isinstance(row[k], float) else str(row[k]) for k in header) + "\n")


def _evaluate_steps(sc, steps):
    """Train DLinear on each dataset with `steps`; return averaged metrics + per-dataset."""
    cfg = _model_cfg(sc)
    per = {}
    t0 = time.time()
    for d in sc.datasets:
        df = _processed(sc, d, steps)
        # degenerate guard: if the PH columns carry no information (all ~constant/zero,
        # e.g. an empty diagram), do NOT reward the model's free extra-channel capacity.
        ph_cols = [c for c in df.columns if c.startswith("ph_")]
        if ph_cols and float(np.nanstd(df[ph_cols].to_numpy(dtype=float))) < 1e-9:
            per[d.stem] = dict(mse=PENALTY, mae=PENALTY, da=float("nan"), r2=float("nan"), seconds=0.0)
            continue
        m = _common.run_cfg(sc.build_fn, sc.model, cfg, df=df)
        mse = m["mse"]
        if not np.isfinite(mse) or mse > PENALTY:    # bound degenerate/diverged configs
            m = dict(mse=PENALTY, mae=PENALTY, r2=float("nan"), da=float("nan"), elapsed_s=m.get("elapsed_s", 0))
        per[d.stem] = dict(mse=float(m["mse"]), mae=float(m["mae"]),
                           da=float(m["da"]), r2=float(m["r2"]), seconds=float(m["elapsed_s"]))
    agg = {k: float(np.nanmean([per[ds][k] for ds in per])) for k in METRICS}
    agg["seconds"] = round(time.time() - t0, 2)
    return agg, per


def evaluate(individual, sc: StudyConfig):
    sig = G.signature(individual)
    if sig in sc._fit_cache:
        return (sc._fit_cache[sig]["mse"],)
    steps = G.decode(individual)
    try:
        agg, per = _evaluate_steps(sc, steps)
    except Exception as e:
        agg = dict(mse=PENALTY, mae=PENALTY, da=float("nan"), r2=float("nan"), seconds=0.0)
        per = {"error": str(e)[:200]}
    sc.n_evals += 1
    rec = dict(eval=sc.n_evals, genome=G.to_dict(individual), signature=sig, steps=steps,
               per_dataset=per, **agg)
    sc._fit_cache[sig] = rec
    if sc.log_path:
        with open(sc.log_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    _write_results_row(sc, rec)
    return (rec["mse"],)
