"""Minimal REST API exposing the forecasting model(s) — Flask.

Endpoints
  GET  /health                      -> status + available models
  GET  /models                      -> models and their default params
  GET  /datasets · /data · /nodes   -> input introspection
  POST /pipeline/run                -> data -> preprocessing chain -> model (sync, or async via "async":true)
  GET  /pipelines                   -> list recent pipeline jobs (running + finished)
  GET  /pipeline/<id>               -> full details (params + status + result) of a pipeline job
  POST /pipeline/<id>/kill          -> terminate a running pipeline job
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
import subprocess
import sys
import threading
import time
import types
import uuid
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


# ---------------------------------------------------------------------------
# Async pipeline jobs: in-memory registry + subprocess worker.
# Killing happens by terminating the worker process — no torch-thread surgery,
# and CUDA memory is released cleanly when the OS reaps the child.
# ---------------------------------------------------------------------------
JOBS = {}                # id -> dict (status, body, started_at, ended_at, result, error, _proc)
JOBS_LOCK = threading.Lock()
MAX_JOBS = 50            # trim oldest finished jobs above this


def _run_pipeline_logic(body):
    """data -> preprocessing chain -> model. Returns metrics dict, or {'error': ...}.

    Pure function (no Flask state) so the subprocess worker can call it directly.
    """
    from preprocessing import nodes
    if not body.get("data"):
        return dict(error="missing required field 'data'")
    name = body.get("model", "dlinear")
    if name not in MODELS:
        return dict(error=f"unknown model '{name}'", available=list(MODELS))
    steps_spec = body.get("preprocessing", []) or []
    for s in steps_spec:
        if not isinstance(s, dict) or "node" not in s:
            return dict(error="each preprocessing step must be {'node':..., 'params':{}}")
        if s["node"] not in nodes.NODES:
            return dict(error=f"unknown node '{s['node']}'", available=list(nodes.NODES))
    steps = [(s["node"], dict(s.get("params") or {})) for s in steps_spec]
    mp = body.get("model_params", {}) or {}

    t0 = time.time()
    df, stem = _resolve_data(body["data"])
    prep_s, n_ph, ph_warn = 0.0, 0, None
    if steps:
        tp = time.time()
        df = apply_pipeline(df, steps, target=mp.get("target", "Close"))
        prep_s = round(time.time() - tp, 3)
        ph_cols = [c for c in df.columns if c.startswith("ph_")]
        n_ph = len(ph_cols)
        if n_ph and float(np.nanstd(df[ph_cols].to_numpy(dtype=float))) < 1e-9:
            ph_warn = ("PH features are all-constant/zero (degenerate config). "
                       "Verify the parameters (e.g. window too small for the embedding).")
    build = _load_build(MODELS[name]["build"])
    cfg = _cfg(mp, MODELS[name])
    m = _common.run_cfg(build, name, cfg, df=df)
    m.pop("preds", None)
    m.update(dataset=stem, model=name,
             preprocessing_steps=[s["node"] for s in steps_spec],
             n_ph_features=n_ph,
             preprocess_seconds=prep_s, train_seconds=m.get("elapsed_s"),
             total_seconds=round(time.time() - t0, 3))
    if ph_warn:
        m["ph_warning"] = ph_warn
    return m


def _prune_jobs():
    """Trim the OLDEST finished jobs so JOBS stays bounded. Caller holds JOBS_LOCK."""
    if len(JOBS) < MAX_JOBS:
        return
    finished = sorted(((j.get("ended_at") or 0, k) for k, j in JOBS.items()
                       if j["status"] != "running"))
    for _, k in finished[:len(JOBS) - MAX_JOBS + 1]:
        JOBS.pop(k, None)


def _new_job(body):
    """Allocate a job id and register a 'running' job. Returns the id."""
    jid = uuid.uuid4().hex[:8]
    with JOBS_LOCK:
        _prune_jobs()
        JOBS[jid] = dict(id=jid, status="running", body=body,
                         started_at=time.time(), ended_at=None,
                         result=None, error=None,
                         model=body.get("model", "dlinear"),
                         dataset=body.get("data"),
                         n_preprocessing_steps=len(body.get("preprocessing") or []))
    return jid


def _start_job(body):
    """Spawn the worker subprocess for `body` and watch it from a background thread."""
    jid = _new_job(body)
    worker = str(ROOT / "pipeline_worker.py")
    proc = subprocess.Popen([sys.executable, "-u", worker],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, cwd=str(ROOT))
    with JOBS_LOCK:
        JOBS[jid]["_proc"] = proc
        JOBS[jid]["pid"] = proc.pid
    threading.Thread(target=_drive_job, args=(jid, body), daemon=True).start()
    return jid


def _drive_job(jid, body):
    """Send body to worker stdin, read its stdout when it exits, update the job."""
    j = JOBS[jid]
    proc = j["_proc"]
    try:
        stdout, stderr = proc.communicate(input=json.dumps(body).encode("utf-8"))
    except Exception as e:
        with JOBS_LOCK:
            if j["status"] != "killed":
                j["status"] = "error"
                j["error"] = f"subprocess error: {type(e).__name__}: {e}"
            j["ended_at"] = time.time()
        return
    with JOBS_LOCK:
        j["ended_at"] = time.time()
        if j["status"] == "killed":
            if not j.get("error"):
                j["error"] = f"killed by user (worker rc={proc.returncode})"
        elif proc.returncode != 0:
            j["status"] = "error"
            err = (stderr.decode("utf-8", errors="replace").strip() or "")[-500:]
            j["error"] = err or f"worker exited with code {proc.returncode}"
        else:
            try:
                res = json.loads(stdout.decode("utf-8", errors="replace") or "{}")
                if isinstance(res, dict) and "error" in res:
                    j["status"] = "error"
                    j["error"] = res["error"]
                else:
                    j["result"] = res
                    j["status"] = "done"
            except Exception as e:
                j["status"] = "error"
                j["error"] = f"bad worker output: {type(e).__name__}: {str(e)[:200]}"


def _job_view(j, include_body=True):
    """Strip non-JSON-serializable fields (Popen handle) and add live elapsed_s."""
    out = {k: v for k, v in j.items() if not k.startswith("_")}
    if not include_body:
        out.pop("body", None)
    end = out.get("ended_at") or time.time()
    out["elapsed_s"] = round(end - out["started_at"], 2)
    return out


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


@app.get("/nodes")
def list_nodes():
    """List the available preprocessing nodes (for /pipeline/run 'preprocessing' field)."""
    from preprocessing import nodes
    return jsonify(list(nodes.NODES))


@app.post("/pipeline/run")
def pipeline_run():
    """Run the full pipeline in one call: data -> preprocessing -> model train+eval.

    Body JSON (same shape, sync by default):
      { "data": "AAPL",
        "preprocessing": [
          {"node":"normalize","params":{"columns":["Volume"]}},
          {"node":"ph_features","params":{"filtration":"vietoris_rips","vectorizer":"silhouette",
                                          "window":96,"stride":5,"dimension":3,"delay":4,
                                          "homology_dim":1,"maxdim":1,"resolution":20}}
        ],
        "model": "dlinear",
        "model_params": { "seq_len":96, ... , "patience":5 },
        "async": false                 # OPTIONAL — when true returns {id,status:"running"} immediately
      }

    Sync returns: metrics + n_ph_features + preprocess/train/total seconds.
    Async returns: {"id":"<8hex>","status":"running"} — poll GET /pipeline/<id>, kill with
    POST /pipeline/<id>/kill, list with GET /pipelines.
    """
    body = request.get_json(force=True, silent=True) or {}
    async_mode = bool(body.pop("async", False))
    if async_mode:
        try:
            jid = _start_job(body)
        except Exception as e:
            return jsonify(error=f"{type(e).__name__}: {str(e)[:300]}"), 500
        return jsonify(id=jid, status="running", url=f"/pipeline/{jid}")
    try:
        m = _run_pipeline_logic(body)
    except Exception as e:
        return jsonify(error=f"{type(e).__name__}: {str(e)[:300]}"), 500
    if isinstance(m, dict) and "error" in m:
        return jsonify(m), 400
    return jsonify(m)


@app.get("/pipelines")
def list_pipelines():
    """List recent pipeline jobs (running + finished, newest first). Body omitted for brevity."""
    with JOBS_LOCK:
        rows = [_job_view(j, include_body=False) for j in JOBS.values()]
    rows.sort(key=lambda r: r["started_at"], reverse=True)
    return jsonify(rows)


@app.get("/pipeline/<jid>")
def get_pipeline(jid):
    """Full details of a pipeline job: submitted body + status + result/error + elapsed_s."""
    if jid == "run":     # belongs to POST /pipeline/run
        return jsonify(error="use POST /pipeline/run to submit work; GET /pipelines to list"), 405
    with JOBS_LOCK:
        j = JOBS.get(jid)
        if not j:
            return jsonify(error=f"unknown pipeline id {jid!r}"), 404
        return jsonify(_job_view(j))


@app.post("/pipeline/<jid>/kill")
def kill_pipeline(jid):
    """Terminate a running pipeline job. Idempotent on already-finished jobs."""
    with JOBS_LOCK:
        j = JOBS.get(jid)
        if not j:
            return jsonify(error=f"unknown pipeline id {jid!r}"), 404
        if j["status"] != "running":
            view = _job_view(j); view["message"] = "not running"
            return jsonify(view)
        proc = j.get("_proc")
        j["status"] = "killed"
        j["error"] = "killed by user"
    warn = None
    if proc is not None:
        try:
            proc.terminate()          # POSIX: SIGTERM; Windows: TerminateProcess
        except Exception as e:
            warn = f"terminate raised: {type(e).__name__}: {e}"
    with JOBS_LOCK:
        view = _job_view(j)
    if warn:
        view["warning"] = warn
    return jsonify(view)


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
