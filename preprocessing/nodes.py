"""Preprocessing nodes for the PH study: normalization + rolling persistent-homology
features (computed via the verified `phlib` toolkit) + cleanup.

Each node is a pure function (df, **params) -> df. The PH node turns a 1-D series
(a CSV column) into rolling topological feature columns appended to the OHLCV frame.
"""
import numpy as np
import pandas as pd

from . import phlib


def _series(df, source, target="Close"):
    if source == "log_return":
        return np.log(df[target].clip(lower=1e-8)).diff().fillna(0.0).to_numpy(float)
    col = "Close" if source == "close" else source
    return df[col].to_numpy(float) if col in df.columns else df[target].to_numpy(float)


def normalize(df, columns=None, **_):
    """Z-score the numeric columns over the frame. (The model harness ALSO fits a
    train-only StandardScaler before windowing; this node is here if you want to
    normalise earlier, e.g. the PH columns.)"""
    df = df.copy()
    cols = columns or list(df.select_dtypes(include=[np.number]).columns)
    for c in cols:
        sd = df[c].std()
        df[c] = (df[c] - df[c].mean()) / (sd if sd > 1e-8 else 1.0)
    return df


def ph_features(df, source="close", window=64, stride=5, point_cloud="takens",
                filtration="vietoris_rips", vectorizer="statistics",
                homology_dim=1, maxdim=1, target="Close", **phlib_params):
    """Rolling persistent-homology features via phlib.

    For each step t the window ending at t is passed to ``phlib.compute`` -> a fixed
    feature vector; vectors are stacked into aligned columns (every `stride` steps,
    forward-filled in between for speed). Errors on a window -> zeros.
    """
    df = df.copy()
    series = _series(df, source, target=target)
    T = len(series)
    md = max(int(maxdim), int(homology_dim))
    feats, k, last = None, None, None
    for t in range(int(window) - 1, T):
        on_stride = (t - (int(window) - 1)) % max(int(stride), 1) == 0
        if k is not None and not on_stride and last is not None:
            feats[t] = last
            continue
        w = series[t - int(window) + 1: t + 1]
        try:
            v, _, _ = phlib.compute(w, point_cloud=point_cloud, filtration=filtration,
                                    vectorizer=vectorizer, homology_dim=int(homology_dim),
                                    maxdim=md, **phlib_params)
            v = np.ravel(np.asarray(v, dtype=float))
        except Exception:
            v = None
        if k is None:                                   # fix feature length from first window
            k = int(v.size) if v is not None and v.size else 1
            feats = np.full((T, k), np.nan)
        if v is None or v.size != k:
            vv = np.zeros(k)
            if v is not None:
                vv[:min(v.size, k)] = v[:k]
            v = vv
        feats[t] = last = v
    if k is None:                                       # window longer than the series
        return df
    cols = [f"ph_{filtration}_{vectorizer}_{i}" for i in range(k)]
    return pd.concat([df, pd.DataFrame(feats, columns=cols, index=df.index)], axis=1)


def cleanup(df, target="Close", **_):
    """Make the frame model-ready: drop infs, fill warmup NaNs."""
    return df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)


NODES = {
    "normalize": normalize,
    "ph_features": ph_features,
    "cleanup": cleanup,
}
