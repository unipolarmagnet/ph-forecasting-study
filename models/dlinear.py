"""DLinear wrapper — uniform CLI entry point.

Run:
    python models/dlinear.py --data_path data/AAPL.csv --seq_len 96 --pred_len 24
Implementation lives in models/dlinear/ (verified vs Zeng et al. AAAI 2023).
"""
import sys
from pathlib import Path

# allow running as a script from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dlinear.model import build          # noqa: E402
import _common                            # noqa: E402

DEFAULTS = dict(
    seq_len=96, pred_len=24, features="MS", target="Close",
    epochs=20, batch_size=32, lr=0.005, patience=5, kernel_size=25,
)

if __name__ == "__main__":
    _common.run(build, "DLinear", DEFAULTS)
