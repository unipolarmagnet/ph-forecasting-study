"""TimeMixer wrapper — uniform CLI entry point.

Run:
    python models/timemixer.py --data_path data/AAPL.csv --seq_len 96 --pred_len 24
Implementation: verified TSLib model (vendor/Time-Series-Library), wired via models/timemixer/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from timemixer.model import build, DEFAULTS   # noqa: E402
import _common                                # noqa: E402

if __name__ == "__main__":
    _common.run(build, "TimeMixer", DEFAULTS)
