"""Read bar files exported from TradeStation Desktop (Data Window / Export Data).

TradeStation writes CSV or TXT with a header such as
    "Date","Time","Open","High","Low","Close","Up","Down"
or  Date,Time,Open,High,Low,Close,Vol,OI  (futures)
where Up/Down are up-tick and down-tick volume. Times are the chart's time zone,
which for Michael is ET. Returns OHLCV with a tz-aware ET index, oldest first,
so it can be used interchangeably with ``tradestation.TradeStation.bars()``.
"""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")


def read_ts_export(path: str | Path, tz: str = "America/New_York") -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p, sep=None, engine="python", skipinitialspace=True)
    df.columns = [str(c).strip().strip('"').title() for c in df.columns]
    if "Time" in df:
        when = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce")
    else:
        when = pd.to_datetime(df["Date"], errors="coerce")
    df = df.loc[when.notna()].copy()
    df.index = pd.DatetimeIndex(when[when.notna()]).tz_localize(ZoneInfo(tz), ambiguous="NaT",
                                                                 nonexistent="shift_forward").tz_convert(ET)
    df.index.name = "time_et"
    df = df[df.index.notna()]
    ren = {"Vol": "Volume", "Total Volume": "Volume", "Oi": "OpenInterest", "Open Int": "OpenInterest"}
    df = df.rename(columns=ren)
    if "Volume" not in df and {"Up", "Down"} <= set(df.columns):
        df["Volume"] = pd.to_numeric(df["Up"], errors="coerce") + pd.to_numeric(df["Down"], errors="coerce")
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume", "Up", "Down", "OpenInterest") if c in df]
    out = df[keep].apply(pd.to_numeric, errors="coerce").sort_index()
    return out[~out.index.duplicated(keep="last")]
