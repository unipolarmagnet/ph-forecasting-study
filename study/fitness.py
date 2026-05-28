"""Fitness for the PH evolutionary study.

fitness(individual) = mean test-MSE across (dataset x seed) trainings of the chosen
model after applying the genome's PH preprocessing. Lower is better.

By default each individual is trained with MULTIPLE seeds per dataset (configurable
via `StudyConfig.seeds`) and the MSEs are averaged — this kills the seed-luck
overfit where a single-seed best score doesn't reproduce on different seeds.
Per-eval between-seed std (`mse_std`) is logged so you can see how robust each
fitness actually is.

PH features are seed-independent, so they are cached per (dataset, genome) — only
the model training multiplies with seed count.
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

PENALTY_Z = 10.0       # divergence threshold in z-space (MSE in scaled units)
PENALTY = 1.0e6        # fallback marker for degenerate cases (well above any realistic
                       # $^2 MSE for any of the stock datasets we use)
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
                 epochs=None, batch_size=32, lr=None, patience=4, seed=2021, seeds=None,
                 device=None, log_path=None, results_path=None):
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
        # `seeds` is the list trained PER individual; `seed` stays for back-compat (== seeds[0]).
        # Default to a single seed so existing call sites keep their cost; the EA CLI sets 3.
        self.seeds = [int(s) for s in (seeds if seeds else [seed])]
        self.seed = self.seeds[0]
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.log_path = Path(log_path) if log_path else None        # full per-eval JSONL
        self.results_path = Path(results_path) if results_path else None  # tidy CSV table
        self._raw = {d: pd.read_csv(d) for d in self.datasets}
        self._feat_cache, self._fit_cache = {}, {}
        self.n_evals = 0
        self.current_gen = 0          # set by the EA so each results row knows its generation


def _model_cfg(sc, seed):
    return types.SimpleNamespace(
        seq_len=sc.seq_len, pred_len=sc.pred_len, features="MS", target=sc.target,
        epochs=sc.epochs, batch_size=sc.batch_size, lr=sc.lr, patience=sc.patience,
        train_ratio=0.7, test_ratio=0.2, no_scale=False, individual=False,
        mix_channels=sc.mix_channels, kernel_size=25, clip_grad=4.0, seed=int(seed),
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
               mse=rec["mse"], mse_std=rec.get("mse_std", 0.0),
               mae=rec["mae"], da=rec["da"], r2=rec["r2"], seconds=rec["seconds"])
    header = list(row)
    new = not sc.results_path.exists()
    with open(sc.results_path, "a", newline="") as f:
        if new:
            f.write(",".join(header) + "\n")
        f.write(",".join(f"{row[k]:.6f}" if isinstance(row[k], float) else str(row[k]) for k in header) + "\n")


def _safe_mean(xs):                                           # no RuntimeWarning on all-NaN
    a = np.asarray(xs, dtype=float)
    return float("nan") if a.size == 0 or np.all(np.isnan(a)) else float(np.nanmean(a))


def _safe_std(xs):
    a = np.asarray(xs, dtype=float)
    return float("nan") if a.size == 0 or np.all(np.isnan(a)) else float(np.nanstd(a))


def _evaluate_steps(sc, steps):
    """Train the model on every (dataset x seed) pair; aggregate by mean.

    Returns (agg, per) where `agg` has mean MSE/MAE/DA/R2 (over all trainings)
    plus `mse_std` (std of MSE across seeds, averaged across datasets — i.e. how
    seed-sensitive this genome's MSE is). `per` is per-dataset with per-seed lists.
    """
    per = {}
    t0 = time.time()
    for d in sc.datasets:
        df = _processed(sc, d, steps)
        # degenerate guard: if the PH columns carry no information (all ~constant/zero,
        # e.g. an empty diagram), do NOT reward the model's free extra-channel capacity.
        ph_cols = [c for c in df.columns if c.startswith("ph_")]
        if ph_cols and float(np.nanstd(df[ph_cols].to_numpy(dtype=float))) < 1e-9:
            # penalty in original (price) units — use the target column's std as the scale
            tstd = float(df[sc.target].std()) if sc.target in df.columns else 1.0
            pen_orig = PENALTY_Z * tstd * tstd
            per[d.stem] = dict(mse=pen_orig, mae=PENALTY_Z * tstd,
                               da=float("nan"), r2=float("nan"), seconds=0.0,
                               mse_per_seed=[pen_orig] * len(sc.seeds), mse_std=0.0)
            continue
        seed_mse, seed_mae, seed_da, seed_r2, seed_s = [], [], [], [], []
        for seed in sc.seeds:
            cfg = _model_cfg(sc, seed)
            m = _common.run_cfg(sc.build_fn, sc.model, cfg, df=df)
            # divergence check on scaled-space MSE (where PENALTY_Z=10 is meaningfully
            # bad regardless of dataset scale). Penalty value is reported back in
            # original (price) units via target_std so it stays comparable.
            mse_z = m.get("mse_z", m["mse"]); tstd = m.get("target_std", 1.0)
            if not np.isfinite(m["mse"]) or mse_z > PENALTY_Z:
                pen_orig = PENALTY_Z * tstd * tstd
                m = dict(mse=pen_orig, mae=PENALTY_Z * tstd,
                         r2=float("nan"), da=float("nan"),
                         elapsed_s=m.get("elapsed_s", 0))
            seed_mse.append(float(m["mse"])); seed_mae.append(float(m["mae"]))
            seed_da.append(float(m["da"]));   seed_r2.append(float(m["r2"]))
            seed_s.append(float(m["elapsed_s"]))
        per[d.stem] = dict(
            mse=_safe_mean(seed_mse), mae=_safe_mean(seed_mae),
            da=_safe_mean(seed_da),   r2=_safe_mean(seed_r2),
            seconds=float(sum(seed_s)),
            mse_per_seed=seed_mse, mse_std=_safe_std(seed_mse))

    agg = {k: _safe_mean([per[ds][k] for ds in per]) for k in METRICS}
    # how seed-sensitive is this genome — mean of per-dataset between-seed std
    agg["mse_std"] = _safe_mean([per[ds]["mse_std"] for ds in per])
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
        agg = dict(mse=PENALTY, mae=PENALTY, da=float("nan"), r2=float("nan"),
                   mse_std=0.0, seconds=0.0)
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
