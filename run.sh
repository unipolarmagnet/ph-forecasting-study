#!/usr/bin/env bash
# Run the full PH study: no-PH benchmark (10x) -> evolutionary PH search -> plots.
#   bash run.sh
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate

# resolve a python interpreter (venv 'python', else system 'python3')
PY=$(command -v python || command -v python3 || true)
if [ -z "$PY" ]; then
    echo "ERROR: no 'python' or 'python3' on PATH. Run 'bash setup.sh' first (creates .venv)." >&2
    exit 1
fi
echo "Using $($PY --version)"

RUN="study/runs/ph_study"
mkdir -p "$RUN"
DATASETS="SPY AAPL JPM"          # this study uses 3 datasets

# 1) no-PH benchmark: DLinear on raw OHLCV, 10 runs across the datasets
"$PY" -m study.benchmark --datasets $DATASETS --n_runs 10 --epochs 10 \
    --seq_len 64 --pred_len 5 --out "$RUN/benchmark.json"

# 2) evolutionary search over PH methods + parameters (DLinear)
"$PY" -m study.evolve --datasets $DATASETS --pop 24 --ngen 50 --epochs 10 \
    --seq_len 64 --pred_len 5 --seed 2021 --out_dir "$RUN"

# 3) plots + summary (convergence, method rankings, population monitor, PH-vs-no-PH verdict)
"$PY" -m study.plots "$RUN"

echo
echo "Done. See $RUN/: results.csv, population.csv, summary.md, *.png, run.log"
