"""TimeMixer — build factory wrapping the verified TSLib implementation.

Implementation: vendor/Time-Series-Library/models/TimeMixer.py
Paper: Wang et al., "TimeMixer: Decomposable Multiscale Mixing for Time Series
Forecasting" (ICLR 2024).  Hyper-parameters mirror the TSLib ETT config verified
during the reproduction phase (multiscale down-sampling enabled).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # models/ dir
from _tslib import build_tslib  # noqa: E402

HP = dict(d_model=16, d_ff=32, e_layers=2, label_len=0,
          down_sampling_layers=3, down_sampling_window=2, down_sampling_method="avg",
          channel_independence=1, decomp_method="moving_avg", moving_avg=25, use_norm=1)

DEFAULTS = dict(seq_len=96, pred_len=24, features="MS", target="Close",
                epochs=10, batch_size=32, lr=0.01, patience=3)


def build(cfg):
    return build_tslib("TimeMixer", cfg, **HP)
