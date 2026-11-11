"""Two "RVI" indicators, on OHLC bars built from the 1-second BRTI series.

* relative_vigor_index      - Ehlers' Relative Vigor Index (close-open vs high-low)
* relative_volatility_index - Donald Dorsey's Relative Volatility Index (RSI on stdev)

Formulas and parameters: docs/rvi_indicators.md. Pure functions, no look-ahead:
the value on bar t uses bars <= t only. Nothing here is a trading signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TIMEFRAMES = {"1m": "1min", "5m": "5min", "15m": "15min"}


def make_bars(price: pd.Series, timeframe: str = "15m") -> pd.DataFrame:
    """1-second prices (DatetimeIndex) -> open/high/low/close bars.

    Bars are labelled by their END time (label='right', closed='left'), so a bar
    is only available once it has finished - no look-ahead.
    """
    rule = TIMEFRAMES.get(timeframe, timeframe)
    r = price.resample(rule, label="right", closed="left")
    bars = r.ohlc()
    return bars.dropna()


def _swma(x: pd.Series) -> pd.Series:
    """Symmetrically weighted moving average, weights 1-2-2-1 over 4 bars."""
    return (x + 2 * x.shift(1) + 2 * x.shift(2) + x.shift(3)) / 6.0


def relative_vigor_index(bars: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """Ehlers RVI. Returns columns rvi and signal.

    num = SWMA(close-open), den = SWMA(high-low)
    rvi = SMA(num, period) / SMA(den, period)
    signal = SWMA(rvi)
    Bars with zero range for the whole window give NaN.
    """
    num = _swma(bars["close"] - bars["open"])
    den = _swma(bars["high"] - bars["low"])
    den_sum = den.rolling(period).sum()
    rvi = (num.rolling(period).sum() / den_sum.where(den_sum != 0))
    return pd.DataFrame({"rvi": rvi, "signal": _swma(rvi)})


def relative_volatility_index(close: pd.Series, std_period: int = 10,
                              smooth_period: int = 14) -> pd.Series:
    """Dorsey RVI, 0-100 (above ~50 = volatility rising with price).

    std = rolling sample stdev of close over std_period bars.
    Up bar (close > prev close): up = std, down = 0; down bar: the reverse;
    unchanged bar: both 0. Both are Wilder-smoothed (alpha = 1/smooth_period);
    RVI = 100 * up / (up + down).
    """
    std = close.rolling(std_period).std(ddof=1)
    chg = close.diff()
    up = std.where(chg > 0, 0.0).where(std.notna())
    dn = std.where(chg < 0, 0.0).where(std.notna())
    alpha = 1.0 / smooth_period
    up_s = up.ewm(alpha=alpha, adjust=False, min_periods=smooth_period).mean()
    dn_s = dn.ewm(alpha=alpha, adjust=False, min_periods=smooth_period).mean()
    tot = up_s + dn_s
    return (100.0 * up_s / tot.where(tot != 0)).rename("rvol")


def directional_volatility_strength(bars: pd.DataFrame, std_period: int = 10,
                                    smooth_period: int = 14,
                                    vigor_period: int = 10) -> pd.DataFrame:
    """DVS: positive vs negative volatility strength, -1 .. +1.

    dorsey = (relative_volatility_index - 50) / 50   (share of volatility on up bars)
    vigor  = relative_vigor_index (close-open vs high-low conviction)
    dvs    = (dorsey + vigor) / 2     equal-weight blend, both already in -1..+1
    Positive = volatility and bar conviction both lean up; negative = both lean down.
    Returns columns dorsey, vigor, dvs. Untested as a signal.
    """
    dorsey = (relative_volatility_index(bars["close"], std_period, smooth_period) - 50.0) / 50.0
    vigor = relative_vigor_index(bars, vigor_period)["rvi"]
    return pd.DataFrame({"dorsey": dorsey, "vigor": vigor, "dvs": (dorsey + vigor) / 2.0})
