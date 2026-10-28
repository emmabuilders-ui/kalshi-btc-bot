"""Historical data loading and settlement.

Three CSV files (see README for the exact columns):

* btc.csv      timestamp, price[, brti]   one BTC spot price per second (gaps are forward-filled)
* markets.csv  ticker, open_time, close_time, strike[, result]
* quotes.csv   timestamp, ticker, yes_bid, yes_ask[, yes_bid_size, yes_ask_size, last_price]

If `brti` is missing, settlement is estimated from spot, which is NOT the same as
the CF Benchmarks BRTI that Kalshi settles on. If markets.csv has the official
`result`, that is used instead and the estimate is only reported for comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SettlementConfig
from .models import Market, Quote


def to_epoch_seconds(col: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(col):
        v = col.astype("float64")
        v = np.where(v > 1e12, v / 1000.0, v)       # milliseconds -> seconds
        return pd.Series(np.floor(v).astype("int64"), index=col.index)
    dt = pd.to_datetime(col, utc=True)
    return pd.Series(dt.astype("int64") // 10**9, index=col.index)


def _num(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


@dataclass
class MarketData:
    t0: int
    price: np.ndarray            # 1-second BTC spot closes, index = t - t0
    settle_index: np.ndarray     # BRTI if provided, else spot (proxy)
    has_brti: bool
    filled_seconds: int          # seconds forward-filled because the feed had gaps
    markets: list[Market]
    quotes: list[Quote]          # sorted by time

    @property
    def t_end(self) -> int:
        return self.t0 + len(self.price) - 1

    def settlement_value(self, m: Market, window_s: int) -> float | None:
        lo, hi = m.close_time - window_s - self.t0, m.close_time - self.t0
        if lo < 0 or hi > len(self.settle_index):
            return None
        return float(self.settle_index[lo:hi].mean())


def load_btc(path: str | Path) -> tuple[int, np.ndarray, np.ndarray, bool, int]:
    df = pd.read_csv(path)
    df["t"] = to_epoch_seconds(df["timestamp"])
    df = df.sort_values("t").groupby("t").last()
    full = pd.RangeIndex(df.index.min(), df.index.max() + 1)
    filled = len(full) - len(df)
    df = df.reindex(full).ffill()
    has_brti = "brti" in df.columns and df["brti"].notna().any()
    idx = df["brti"].ffill().bfill() if has_brti else df["price"]
    return int(full[0]), df["price"].to_numpy(float), idx.to_numpy(float), bool(has_brti), filled


def load_markets(path: str | Path) -> list[Market]:
    df = pd.read_csv(path)
    df["open_t"] = to_epoch_seconds(df["open_time"])
    df["close_t"] = to_epoch_seconds(df["close_time"])
    out = []
    for r in df.itertuples(index=False):
        res = getattr(r, "result", None)
        res = str(res).strip().lower() if isinstance(res, str) and res.strip() else None
        out.append(Market(str(r.ticker), int(r.open_t), int(r.close_t), float(r.strike), res))
    return sorted(out, key=lambda m: m.open_time)


def load_quotes(path: str | Path) -> list[Quote]:
    df = pd.read_csv(path)
    df["t"] = to_epoch_seconds(df["timestamp"])
    df = df.sort_values("t", kind="stable")
    for c in ("yes_bid", "yes_ask", "yes_bid_size", "yes_ask_size", "last_price"):
        if c not in df.columns:
            df[c] = np.nan
    out = []
    for r in df.itertuples(index=False):
        bid, ask = _num(r.yes_bid), _num(r.yes_ask)
        out.append(Quote(int(r.t), str(r.ticker),
                         bid if bid and bid > 0 else None,
                         ask if ask and ask < 1 else None,
                         _num(r.yes_bid_size), _num(r.yes_ask_size), _num(r.last_price)))
    return out


def load_dir(path: str | Path) -> MarketData:
    p = Path(path)
    t0, price, idx, has_brti, filled = load_btc(p / "btc.csv")
    return MarketData(t0, price, idx, has_brti, filled,
                      load_markets(p / "markets.csv"), load_quotes(p / "quotes.csv"))


def settle_result(value: float, strike: float, cfg: SettlementConfig) -> str:
    yes = value >= strike if cfg.yes_if == ">=" else value > strike
    return "yes" if yes else "no"
