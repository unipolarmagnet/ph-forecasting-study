"""No-PH benchmark: DLinear on raw OHLCV (no persistent-homology features), run N=10
times (different seeds) across the 12 datasets. Records MSE/MAE/DA/R2/time per run and
mean +/- std, so the evolved best PH config can be judged against this control.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from study.fitness import StudyConfig, _evaluate_steps, METRICS  # noqa: E402

DATA = ROOT / "data"
DATASETS = ["SPY", "AAPL", "JPM"]   # this study uses 3 (all 12 CSVs are present in data/)


def run(datasets, model="dlinear", n_runs=10, epochs=10, seq_len=64, pred_len=5, base_seed=2021, out=None):
    print(f"No-PH benchmark: {model} on raw OHLCV, {n_runs} runs x {len(datasets)} datasets\n")
    rows = []
    t0 = time.time()
    for i in range(n_runs):
        sc = StudyConfig(datasets=datasets, model=model, epochs=epochs, seq_len=seq_len,
                         pred_len=pred_len, seed=base_seed + i)
        agg, _ = _evaluate_steps(sc, [])             # steps=[] -> cleanup only = raw OHLCV
        agg["seed"] = base_seed + i
        rows.append(agg)
        print(f"  run {i+1:>2d}/{n_runs}  seed={base_seed+i}  "
              f"MSE {agg['mse']:.4f}  MAE {agg['mae']:.4f}  DA {agg['da']:.3f}  "
              f"R2 {agg['r2']:.4f}  ({agg['seconds']:.0f}s)")
    summary = {m: dict(mean=float(np.nanmean([r[m] for r in rows])),
                       std=float(np.nanstd([r[m] for r in rows]))) for m in METRICS}
    result = dict(no_ph=True, model=model, n_runs=n_runs, datasets=[Path(d).stem for d in datasets],
                  epochs=epochs, seq_len=seq_len, pred_len=pred_len,
                  per_run=rows, summary=summary, total_seconds=round(time.time() - t0, 1))
    if out:
        out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        with open(out.with_suffix(".csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["run", "seed"] + METRICS + ["seconds"])
            for j, r in enumerate(rows):
                w.writerow([j, r["seed"]] + [f"{r[m]:.6f}" for m in METRICS] + [r["seconds"]])
    print("\n=== no-PH benchmark (mean +/- std over runs) ===")
    for m in METRICS:
        print(f"  {m.upper():4s}  {summary[m]['mean']:.4f} +/- {summary[m]['std']:.4f}")
    if out:
        print(f"written: {out} (+ .csv)")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="*", default=DATASETS)
    p.add_argument("--model", default="dlinear", choices=["dlinear","patchtst","timemixer"])
    p.add_argument("--n_runs", type=int, default=10)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--pred_len", type=int, default=5)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    ds = [str(DATA / f"{t}.csv") if not str(t).endswith(".csv") else t for t in a.datasets]
    run(ds, model=a.model, n_runs=a.n_runs, epochs=a.epochs, seq_len=a.seq_len, pred_len=a.pred_len, out=a.out)
