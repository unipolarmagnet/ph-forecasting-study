"""Download daily (1D) stock data from Yahoo Finance into data/ as CSV.

Source: Yahoo Finance via the `yfinance` library (https://github.com/ranaroussi/yfinance).
Universe: one representative asset per GICS sector + SPY as an all-sectors benchmark.
Range : 2016-01-01 .. 2025-12-31 (END is exclusive -> last bar 2025-12-31).
Prices: auto-adjusted (auto_adjust=True), so `Close` is the Adjusted Close — it
        accounts for corporate actions (dividends, splits) for a consistent signal.
Output: one CSV per ticker, columns: date,Open,High,Low,Close,Volume.

Each download is validated (missing values, logical inconsistencies such as
High < Low, non-positive prices, negative volume, duplicate/unsorted dates).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

OUT = Path(__file__).resolve().parent

# ticker -> GICS sector it represents
SECTORS = {
    "XOM":   "Energy",
    "APD":   "Materials",
    "BA":    "Industrials",
    "AMZN":  "Consumer Discretionary",
    "PG":    "Consumer Staples",
    "JNJ":   "Health Care",
    "JPM":   "Financials",
    "AAPL":  "Information Technology",
    "GOOGL": "Communication Services",
    "NEE":   "Utilities",
    "AMT":   "Real Estate",
    "SPY":   "Benchmark (All Sectors)",
}
TICKERS = list(SECTORS)
START = "2016-01-01"
END = "2026-01-01"   # exclusive -> last bar 2025-12-31

_OHLC = ["Open", "High", "Low", "Close"]
_EPS = 1e-6          # float tolerance for High/Low consistency checks


def fetch(ticker):
    df = yf.download(ticker, start=START, end=END, interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        return None
    if isinstance(df.columns, pd.MultiIndex):   # single-ticker multiindex flatten
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={"Date": "date"})
    keep = ["date", "Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep if c in df.columns]]
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def validate(df):
    """Return a list of (check, count) for any quality issues found (empty = clean)."""
    issues = []
    miss = int(df[_OHLC + ["Volume"]].isna().sum().sum())
    if miss:
        issues.append(("missing values (NaN)", miss))
    o, h, l, c = (df[k] for k in _OHLC)
    hi_lt_lo = int((h < l - _EPS).sum())
    if hi_lt_lo:
        issues.append(("High < Low", hi_lt_lo))
    hi_bad = int((h < np.maximum(o, c) - _EPS).sum())
    if hi_bad:
        issues.append(("High < max(Open,Close)", hi_bad))
    lo_bad = int((l > np.minimum(o, c) + _EPS).sum())
    if lo_bad:
        issues.append(("Low > min(Open,Close)", lo_bad))
    nonpos = int((df[_OHLC] <= 0).sum().sum())
    if nonpos:
        issues.append(("non-positive price", nonpos))
    negvol = int((df["Volume"] < 0).sum())
    if negvol:
        issues.append(("negative volume", negvol))
    dups = int(df["date"].duplicated().sum())
    if dups:
        issues.append(("duplicate dates", dups))
    if not df["date"].is_monotonic_increasing:
        issues.append(("dates not sorted", 1))
    return issues


def main(tickers):
    print(f"Downloading {len(tickers)} tickers  {START}..{END} (exclusive)  [Adjusted Close]\n")
    ok, bad = 0, 0
    for t in tickers:
        df = fetch(t)
        sector = SECTORS.get(t, "?")
        if df is None:
            print(f"  {t:6s} {sector:24s} NO DATA"); bad += 1; continue
        issues = validate(df)
        path = OUT / f"{t}.csv"
        df.to_csv(path, index=False)
        status = "PASS" if not issues else "ISSUES: " + ", ".join(f"{n}×{c}" for c, n in issues)
        print(f"  {t:6s} {sector:24s} {len(df):>4d} rows  "
              f"{df['date'].iloc[0]}..{df['date'].iloc[-1]}  {status}")
        ok += 1
    print(f"\nDone: {ok} written, {bad} failed.  Output dir: {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:] or TICKERS)
