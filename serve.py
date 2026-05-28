"""Minimal REST API exposing the forecasting model(s) — Flask.

Endpoints
  GET  /health                      -> status + available models
  GET  /models                      -> models and their default params
  POST /model/<name>                -> train + evaluate the model, return metrics JSON

POST body (JSON; only `data` is required):
  {
    "data": "AAPL",                 # a ticker in data/ (e.g. SPY/AAPL/JPM) OR a .csv path
    "seq_len": 96, "pred_len": 24, "epochs": 10,
    "features": "MS", "target": "Close",
    "lr": 0.005, "batch_size": 32, "mix_channels": true, "seed": 2021,
    "ph": {                         # OPTIONAL: prepend persistent-homology features (phlib)
        "filtration": "vietoris_rips", "vectorizer": "silhouette",
        "point_cloud": "takens", "window": 96, "stride": 5,
        "dimension": 3, "delay": 4, "homology_dim": 1, "maxdim": 1, "resolution": 20
    }
  }
Returns: { model, dataset, ph, mse, mae, r2, da, n_features, elapsed_s, epochs_ran, ... }

Run:  python serve.py --host 0.0.0.0 --port 8000     (then POST to it)
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))

import _common                                       # noqa: E402
from preprocessing import apply as apply_pipeline    # noqa: E402

DATA = ROOT / "data"

# model registry (name -> build spec + sensible defaults). Add rows to expose more models.
MODELS = {
    "dlinear":   dict(build="dlinear.model:build",   lr=0.005,  epochs=20, mix_channels=True),
    "patchtst":  dict(build="patchtst.model:build",  lr=0.0001, epochs=10, mix_channels=False),
    "timemixer": dict(build="timemixer.model:build", lr=0.01,   epochs=10, mix_channels=False),
}


def _load_build(spec):
    import importlib
    mod, attr = spec.split(":")
    return getattr(importlib.import_module(mod), attr)


def _resolve_data(d):
    p = Path(d) if str(d).endswith(".csv") else DATA / f"{d}.csv"
    if not p.exists():
        raise FileNotFoundError(f"dataset not found: {d}")
    return pd.read_csv(p), p.stem


def _cfg(body, reg):
    import torch
    return types.SimpleNamespace(
        seq_len=int(body.get("seq_len", 96)), pred_len=int(body.get("pred_len", 24)),
        features=body.get("features", "MS"), target=body.get("target", "Close"),
        epochs=int(body.get("epochs", reg["epochs"])), batch_size=int(body.get("batch_size", 32)),
        lr=float(body.get("lr", reg["lr"])), patience=int(body.get("patience", 5)),
        train_ratio=0.7, test_ratio=0.2, no_scale=False,
        individual=bool(body.get("individual", False)),
        mix_channels=bool(body.get("mix_channels", reg["mix_channels"])), kernel_size=25,
        clip_grad=float(body.get("clip_grad", 0.0)), seed=int(body.get("seed", 2021)),
        device=body.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        out_dir=None, quiet=True)


app = Flask(__name__)


@app.get("/")
def explorer():
    """Self-contained web explorer for the API (forms + live responses)."""
    html = (ROOT / "explorer.html").read_text(encoding="utf-8")
    return app.response_class(html, mimetype="text/html")


@app.get("/health")
def health():
    return jsonify(status="ok", models=list(MODELS))


@app.get("/datasets")
def datasets():
    """List available CSV datasets with row counts and date range."""
    out = []
    for p in sorted(DATA.glob("*.csv")):
        rec = dict(filename=p.name, ticker=p.stem)
        try:
            d = pd.read_csv(p, usecols=lambda c: c.lower() == "date")
            col = d.columns[0]
            rec.update(n_rows=int(len(d)), start=str(d[col].iloc[0]), end=str(d[col].iloc[-1]))
        except Exception:
            pass
        out.append(rec)
    return jsonify(out)


@app.get("/models")
def models():
    out = {}
    for n, reg in MODELS.items():
        try:
            _load_build(reg["build"]); ok = True
        except Exception:
            ok = False
        out[n] = dict(available=ok, defaults=dict(lr=reg["lr"], epochs=reg["epochs"],
                                                  mix_channels=reg["mix_channels"]))
    return jsonify(out)


@app.get("/data")
def get_data():
    """Return a dataset's time series as structured JSON.

    Query params: filename (required, e.g. AAPL.csv) ; optional:
      columns=date,Close   subset of columns
      start=YYYY-MM-DD end=YYYY-MM-DD   date range
      tail=N               only the last N rows
      orient=records|columns|split   (default records)
    """
    fn = request.args.get("filename")
    if not fn:
        return jsonify(error="missing query param 'filename' (e.g. AAPL.csv)"), 400
    name = fn if fn.lower().endswith(".csv") else fn + ".csv"
    path = DATA / name
    if not path.exists():
        return jsonify(error=f"file not found: {name}",
                       available=[p.name for p in sorted(DATA.glob("*.csv"))]), 404
    df = pd.read_csv(path)
    cols = request.args.get("columns")
    if cols:
        keep = [c.strip() for c in cols.split(",") if c.strip() in df.columns]
        df = df[keep] if keep else df
    if "date" in df.columns:
        if request.args.get("start"):
            df = df[df["date"] >= request.args["start"]]
        if request.args.get("end"):
            df = df[df["date"] <= request.args["end"]]
    tail = request.args.get("tail", type=int)
    if tail:
        df = df.tail(tail)
    orient = request.args.get("orient", "records")
    payload = dict(filename=name, ticker=name[:-4], columns=list(df.columns), n_rows=int(len(df)))
    if "date" in df.columns and len(df):
        payload["start"] = str(df["date"].iloc[0])
        payload["end"] = str(df["date"].iloc[-1])
    if orient == "columns":
        payload["data"] = {c: df[c].tolist() for c in df.columns}      # column arrays
    elif orient == "split":
        payload["data"] = json.loads(df.to_json(orient="split"))        # {columns,index,data}
    else:
        payload["data"] = json.loads(df.to_json(orient="records"))      # list of row objects
    payload["orient"] = orient
    return jsonify(payload)


@app.post("/model/<name>")
def run_model(name):
    if name not in MODELS:
        return jsonify(error=f"unknown model '{name}'", available=list(MODELS)), 404
    body = request.get_json(force=True, silent=True) or {}
    if "data" not in body:
        return jsonify(error="missing required field 'data' (ticker name or .csv path)"), 400
    try:
        t0 = time.time()
        df, stem = _resolve_data(body["data"])
        ph_secs, n_ph, ph_warn = 0.0, 0, None
        if body.get("ph"):                            # optional PH feature prepend (phlib)
            tph = time.time()
            df = apply_pipeline(df, [("ph_features", dict(body["ph"]))],
                                target=body.get("target", "Close"))
            ph_secs = round(time.time() - tph, 3)
            ph_cols = [c for c in df.columns if c.startswith("ph_")]
            n_ph = len(ph_cols)
            if n_ph and float(np.nanstd(df[ph_cols].to_numpy(dtype=float))) < 1e-9:
                ph_warn = ("PH features are all-constant/zero (degenerate config — e.g. window "
                           "too small for the embedding so the point cloud is empty). They add no info.")
        build = _load_build(MODELS[name]["build"])
        cfg = _cfg(body, MODELS[name])
        m = _common.run_cfg(build, name, cfg, df=df)
        m.pop("preds", None)
        m.update(dataset=stem, ph=bool(body.get("ph")), n_ph_features=n_ph,
                 preprocess_seconds=ph_secs, train_seconds=m.get("elapsed_s"),
                 total_seconds=round(time.time() - t0, 3))
        if ph_warn:
            m["ph_warning"] = ph_warn
        return jsonify(m)
    except Exception as e:
        return jsonify(error=f"{type(e).__name__}: {str(e)[:300]}"), 500


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    print(f"Serving models {list(MODELS)} on http://{a.host}:{a.port}")
    # threaded=False: one training at a time (torch training shouldn't overlap on one GPU)
    app.run(host=a.host, port=a.port, threaded=False)
