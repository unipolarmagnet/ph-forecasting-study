# PH study (deployment2) — minimal, self-contained

Finds **which persistent-homology methods and parameters** give the best stock-price
forecasts (DLinear) across the study datasets — **SPY, AAPL, JPM** — vs a no-PH
benchmark. Only the pieces this experiment uses are included.

```
data/           12 ticker CSVs present; this study uses SPY, AAPL, JPM (2016–2025, adj. close)
models/         3 forecasters: DLinear (self-contained) + PatchTST, TimeMixer (TSLib adapter)
                + _common.py harness (MSE/MAE/DA/R2). EA is model-selectable via --model.
preprocessing/  phlib.py (PH toolkit) + nodes.py (normalize, rolling ph_features, cleanup)
study/          genome · fitness · evolve (the EA) · benchmark (no-PH x10) · plots
```

## What it does
- **Search space (genome):** the PH process via `phlib` — point cloud (`takens`,
  `sliding_window`, `takens_pca`, `sliding_window_pca`), filtration (`vietoris_rips`,
  `dtm_rips`, `alpha`, `witness`, `cubical`, `upper_star`, `periodic_cubical`),
  vectorization (`landscape`, `silhouette`, `betti`, `image`, `entropy`,
  `topological_vector`, `complex_polynomial`, `persistence_image`, `statistics`) +
  their parameters (window, dim, delay, stride, resolution, metric, …).
- **Model:** DLinear (+channel-mix), train/val/test 0.7/0.1/0.2, train-only StandardScaler.
- **Fitness:** mean **test MSE** across the 3 study datasets (SPY, AAPL, JPM); lower = better.
- **Per-individual metrics** logged to `results.csv`: **MSE, MAE, DA (directional
  accuracy), R², time**.
- **Benchmark:** DLinear on **raw OHLCV (no PH)**, run **10×** (different seeds) → mean±std.
- **Population monitoring:** every generation's full population is logged
  (`generations.jsonl`, `population.csv`) and visualised (`population_monitor.png`).

## Setup & run (Linux, no sudo)
```bash
bash setup.sh
source .venv/bin/activate
bash run.sh                         # benchmark(10x) -> EA -> plots
# long runs:  nohup bash run.sh > run_full.log 2>&1 & disown
#             tail -f study/runs/ph_study/run.log
```
CPU works; GPU optional. For CPU-only torch see the note in `requirements.txt`.

## Manual commands
```bash
python -m study.benchmark --n_runs 10 --out study/runs/ph_study/benchmark.json
python -m study.evolve --model dlinear --pop 24 --ngen 50 --epochs 10 --seq_len 64 --pred_len 5
# or run the EA on PatchTST / TimeMixer:
#   python -m study.evolve --model timemixer --pop 24 --ngen 30
#   python -m study.benchmark --model timemixer --n_runs 10 --out study/runs/tm/benchmark.json
python -m study.plots  study/runs/ph_study
python preprocessing/verify_phlib.py     # 43/43 correctness checks for phlib
```

## REST API (optional) — `serve.py`
Expose the model as a service (Flask):
```bash
python serve.py --host 0.0.0.0 --port 8000      # needs: pip install flask
```
- `GET  /health` · `GET /models`
- `POST /model/dlinear` → trains + evaluates, returns `{mse, mae, da, r2, n_features, elapsed_s, ...}`
```bash
# raw OHLCV
curl -X POST localhost:8000/model/dlinear -H "Content-Type: application/json" \
     -d '{"data":"AAPL","epochs":20,"seq_len":96,"pred_len":24}'
# with persistent-homology features (phlib)
curl -X POST localhost:8000/model/dlinear -H "Content-Type: application/json" \
     -d '{"data":"SPY","ph":{"filtration":"vietoris_rips","vectorizer":"silhouette",
          "window":96,"stride":5,"dimension":3,"delay":4,"homology_dim":1,"maxdim":1,"resolution":20}}'
```
Add more models by adding rows to `MODELS` in `serve.py`.

## Outputs — `study/runs/ph_study/`
- `results.csv` — one row per evaluated individual: key genes + **MSE, MAE, DA, R², time**
- `population.csv` / `generations.jsonl` — the full population of each generation
- `benchmark.json` (+ `.csv`) — the 10-run no-PH control
- `convergence.png`, `method_perf.png`, `population_monitor.png`, `numeric_trends.png`
- `summary.md` — best pipeline (+ all metrics), method rankings, PH-vs-no-PH verdict
- `hall_of_fame.json`, `logbook.csv`, `config.json`, `run.log`

## Notes
- PH features are **rolling-window**: for each step the window ending there is sent to
  `phlib` → a feature vector, appended to OHLCV (recomputed every `stride` steps).
- `phlib` is a facade over **gudhi / ripser / persim / scikit-learn** — no PH algorithm
  is re-implemented; it is verified by `preprocessing/verify_phlib.py` (43/43).
- Everything is seed-deterministic and logged for reproducibility.
