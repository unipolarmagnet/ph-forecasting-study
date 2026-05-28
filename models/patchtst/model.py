"""PatchTST — build factory wrapping the verified TSLib implementation.

Implementation: vendor/Time-Series-Library/models/PatchTST.py
Paper: Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with
Transformers" (ICLR 2023).  Hyper-parameters mirror the TSLib ETT config verified
during the reproduction phase.  (patch_len=16, stride=8 are fixed in the TSLib
Model constructor.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # models/ dir
from _tslib import build_tslib  # noqa: E402

HP = dict(d_model=16, d_ff=128, e_layers=3, n_heads=4, factor=1,
          dropout=0.2, label_len=48)

DEFAULTS = dict(seq_len=96, pred_len=24, features="MS", target="Close",
                epochs=10, batch_size=32, lr=0.0001, patience=3)


def build(cfg):
    return build_tslib("PatchTST", cfg, **HP)
