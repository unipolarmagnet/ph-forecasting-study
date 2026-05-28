"""Adapter that plugs verified Time-Series-Library (TSLib) models into our harness.

The verified implementations live in `vendor/Time-Series-Library/` (single source
of truth — we do NOT re-implement them). Each TSLib forecasting model exposes:

    Model(configs).forward(x_enc, x_mark_enc, x_dec, x_mark_dec) -> [B, pred_len, c_out]

while our `_common` harness calls `model(x)` with `x = [B, seq_len, C]`. This
module bridges the two:
  * builds a TSLib `configs` namespace (mirrors run.py argparse defaults so no
    attribute is missing) from our small `cfg`,
  * loads the model class straight from the vendored file (by path, so it does
    not clash with our own top-level `models/` directory),
  * wraps it so calendar features are unused (x_mark=None — all TSLib embeddings
    handle that) and the decoder input is the standard zero-seeded tensor.

These models are channel-mixing by construction, so unlike DLinear they need no
`--mix_channels` head; with features="MS" they output all channels and the
harness slices the target.
"""
import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

TSLIB = Path(__file__).resolve().parent.parent / "vendor" / "Time-Series-Library"

# Mirror of TSLib run.py argparse defaults — only the modelling-relevant fields,
# so any TSLib Model.__init__ can be constructed without a missing attribute.
_DEFAULTS = dict(
    task_name="long_term_forecast", features="MS", target="Close", freq="h",
    seq_len=96, label_len=48, pred_len=24,
    enc_in=7, dec_in=7, c_out=7, num_class=1,
    d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
    moving_avg=25, factor=1, dropout=0.1, embed="timeF", activation="gelu",
    top_k=5, num_kernels=6,
    channel_independence=1, decomp_method="moving_avg", use_norm=1,
    down_sampling_layers=0, down_sampling_window=1, down_sampling_method=None,
    seg_len=96, patch_len=16, expand=2, d_conv=4, individual=False,
    distil=True, p_hidden_dims=[128, 128], p_hidden_layers=2,
)


def _ensure_path():
    p = str(TSLIB)
    if p not in sys.path:
        sys.path.insert(0, p)  # so the model's `from layers... import` resolves


def _load_model_class(tslib_name):
    """Load `Model` from vendor/Time-Series-Library/models/<tslib_name>.py by path."""
    _ensure_path()
    path = TSLIB / "models" / f"{tslib_name}.py"
    if not path.exists():
        raise FileNotFoundError(f"TSLib model not found: {path}")
    spec = importlib.util.spec_from_file_location(f"tslib_{tslib_name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Model


class TSLibForecast(nn.Module):
    """Wrap a TSLib forecasting Model to accept `x = [B, seq_len, C]`."""

    def __init__(self, net, label_len, pred_len):
        super().__init__()
        self.net = net
        self.label_len = label_len
        self.pred_len = pred_len

    def forward(self, x):  # x: [B, seq_len, C]
        b, _, c = x.shape
        x_dec = torch.zeros(b, self.label_len + self.pred_len, c,
                            device=x.device, dtype=x.dtype)
        if self.label_len > 0:                      # standard zero-seeded decoder input
            x_dec[:, :self.label_len] = x[:, -self.label_len:]
        out = self.net(x, None, x_dec, None)        # x_mark=None -> no calendar features
        return out[:, -self.pred_len:, :]           # [B, pred_len, c_out]


def build_tslib(tslib_name, cfg, **overrides):
    """Construct a harness-ready model wrapping the verified TSLib `tslib_name`.

    cfg must provide seq_len, pred_len, enc_in (set by _common before build).
    `overrides` set the model's hyper-parameters (d_model, e_layers, ...).
    """
    conf = dict(_DEFAULTS)
    conf.update(task_name="long_term_forecast",
                seq_len=cfg.seq_len, pred_len=cfg.pred_len,
                enc_in=cfg.enc_in, dec_in=cfg.enc_in, c_out=cfg.enc_in)
    conf.update(overrides)
    ns = types.SimpleNamespace(**conf)
    Model = _load_model_class(tslib_name)
    net = Model(ns)
    return TSLibForecast(net, ns.label_len, ns.pred_len)
