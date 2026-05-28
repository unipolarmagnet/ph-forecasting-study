"""PatchTST wrapper — uniform CLI entry point.

Run:
    python models/patchtst.py --data_path data/AAPL.csv --seq_len 96 --pred_len 24
Implementation: verified TSLib model (vendor/Time-Series-Library), wired via models/patchtst/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from patchtst.model import build, DEFAULTS   # noqa: E402
import _common                               # noqa: E402

if __name__ == "__main__":
    _common.run(build, "PatchTST", DEFAULTS)
