"""Cross-model validation of the best PH genome found by the DLinear EA.

For each model in {DLinear, TimeMixer, PatchTST}:
  For each seed in [2021, 2022, 2023]:
    - train + eval WITH the best PH features (decoded from dlinear_best_genomes.json)
    - train + eval WITHOUT PH (raw OHLCV)
  -> compare mean +/- std across seeds.  Does PH transfer to the other models?
"""
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))

import _common                                       # noqa: E402
from preprocessing import apply as apply_pipeline    # noqa: E402

MODELS = {
    "dlinear":    dict(build="dlinear.model:build",    lr=0.005,  mix=True),
    "patchtst":   dict(build="patchtst.model:build",   lr=0.0001, mix=False),
    "timemixer":  dict(build="timemixer.model:build",  lr=0.01,   mix=False),
}
DATASETS = ["SPY", "AAPL", "JPM"]
SEEDS = [2021, 2022, 2023]
EPOCHS = 10
SEQ_LEN, PRED_LEN = 64, 5
DATA = ROOT / "data"


def _load_build(spec):
    import importlib
    m, a = spec.split(":")
    return getattr(importlib.import_module(m), a)


def _cfg(seed, lr, mix):
    import torch
    return types.SimpleNamespace(
        seq_len=SEQ_LEN, pred_len=PRED_LEN, features="MS", target="Close",
        epochs=EPOCHS, batch_size=32, lr=lr, patience=4,
        train_ratio=0.7, test_ratio=0.2, no_scale=False, individual=False,
        mix_channels=mix, kernel_size=25, clip_grad=4.0, seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
        out_dir=None, quiet=True)


def _eval(model_name, seed, steps):
    """Train+eval the model on each dataset with given preprocessing steps; aggregate."""
    reg = MODELS[model_name]
    build = _load_build(reg["build"])
    cfg = _cfg(seed, reg["lr"], reg["mix"])
    per = []
    for ticker in DATASETS:
        df = pd.read_csv(DATA / f"{ticker}.csv")
        if steps:
            df = apply_pipeline(df, steps, target="Close")
        m = _common.run_cfg(build, model_name, cfg, df=df)
        per.append(dict(ticker=ticker, mse=m["mse"], mae=m["mae"],
                        da=m["da"], r2=m["r2"], elapsed=m["elapsed_s"]))
    return {k: float(np.nanmean([p[k] for p in per])) for k in ("mse", "mae", "da", "r2")}, per


def main():
    g = json.load(open("dlinear_best_genomes.json"))
    best = g["best_individuals"][0]
    ph_steps = best["steps"]
    print(f"Best PH genome (from DLinear EA): mse {best['fitness']:.4f}")
    print(f"  {best['genome']['point_cloud']}/{best['genome']['filtration']}/"
          f"{best['genome']['vectorizer']} (H{best['genome']['homology_dim']})  "
          f"window={best['genome']['window']} stride={best['genome']['stride']}\n")
    t0 = time.time()
    results = {}
    for name in MODELS:
        print(f"=== {name.upper()} ===")
        raw_runs, ph_runs = [], []
        for seed in SEEDS:
            ts = time.time()
            raw = _eval(name, seed, [])[0]
            ph = _eval(name, seed, ph_steps)[0]
            raw_runs.append(raw); ph_runs.append(ph)
            print(f"  seed {seed}: raw mse {raw['mse']:.4f}  ph mse {ph['mse']:.4f}  "
                  f"(d {(raw['mse']-ph['mse'])/raw['mse']*100:+.1f}%)  [{time.time()-ts:.0f}s]")
        def stat(rs, k): return float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))
        results[name] = dict(
            baseline={k: stat(raw_runs, k) for k in ("mse", "mae", "da", "r2")},
            ph={k: stat(ph_runs, k) for k in ("mse", "mae", "da", "r2")},
            raw_runs=raw_runs, ph_runs=ph_runs)

    print("\n=== SUMMARY (mean +/- std over 3 seeds x 3 datasets) ===")
    print(f"{'model':10s}  {'baseline mse':>20s}  {'+PH mse':>18s}  {'delta':>8s}   verdict")
    for name, r in results.items():
        bm, bs = r["baseline"]["mse"]; pm, ps = r["ph"]["mse"]
        delta = (bm - pm) / bm * 100
        # PH genuinely better if mean<baseline AND gap > baseline noise std
        verdict = ("HELPS (above noise)" if pm < bm and (bm - pm) > bs else
                   "marginal (within noise)" if pm < bm else
                   "no help / worse")
        print(f"  {name:10s}  {bm:.4f} +/- {bs:.4f}   {pm:.4f} +/- {ps:.4f}   {delta:+6.1f}%   {verdict}")
    Path("validate_best_results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\ntotal time: {time.time()-t0:.0f}s   written: validate_best_results.json")


if __name__ == "__main__":
    main()
