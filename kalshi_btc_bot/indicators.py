"""Streaming indicators.

Two models are kept deliberately separate:

* EntryIndicator: the asymmetrical Bollinger entry model on 1-second closes
  (MA50 / MA250 / MA500, SD250, Z-score, separate upper and lower breach
  tracking, MA50/MA250 golden cross).
* ExitBands + TrailingStop: the traditional Bollinger exit model on
  N-second candles (default 20 x 5s) with the trailing-stop state machine.

All inputs are one price per whole second (integer epoch seconds). The same
classes run in the backtester and in paper trading, so both see identical
signals for identical data.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from .config import EntryConfig, ExitMethod


class RollingStats:
    """O(1) rolling mean / standard deviation over the last `window` values.

    Sums are kept relative to a reference price (BTC ~ 1e5, while SD can be
    a few dollars) and rebuilt periodically so float drift cannot accumulate.
    """

    def __init__(self, window: int, ddof: int = 0):
        if window < 1 + ddof:
            raise ValueError("window too small")
        self.window = window
        self.ddof = ddof
        self._buf: deque[float] = deque()
        self._ref = 0.0
        self._s = 0.0
        self._ss = 0.0
        self._since_rebuild = 0

    def push(self, x: float) -> None:
        if not self._buf:
            self._ref = x
        self._buf.append(x)
        d = x - self._ref
        self._s += d
        self._ss += d * d
        if len(self._buf) > self.window:
            o = self._buf.popleft() - self._ref
            self._s -= o
            self._ss -= o * o
        self._since_rebuild += 1
        if self._since_rebuild >= 4 * self.window:
            self._rebuild()

    def _rebuild(self) -> None:
        self._ref = self._buf[-1]
        self._s = sum(x - self._ref for x in self._buf)
        self._ss = sum((x - self._ref) ** 2 for x in self._buf)
        self._since_rebuild = 0

    @property
    def ready(self) -> bool:
        return len(self._buf) >= self.window

    def mean(self) -> float:
        return self._ref + self._s / len(self._buf)

    def std(self) -> float:
        n = len(self._buf)
        var = (self._ss - self._s * self._s / n) / (n - self.ddof)
        return math.sqrt(var) if var > 0 else 0.0


@dataclass(frozen=True)
class EntrySnapshot:
    t: int
    price: float
    ma50: float | None
    ma250: float | None
    ma500: float | None
    sd250: float | None
    upper: float | None
    lower: float | None
    z: float | None
    lower_breach: bool          # Z(t) <= -2 this second
    upper_breach: bool          # Z(t) >= +2 this second
    last_lower_breach_t: int | None
    last_upper_breach_t: int | None
    recent_lower_breach: bool   # a lower breach within breach_window_s, at or before t
    golden_cross: bool          # MA50(t-1) <= MA250(t-1) and MA50(t) > MA250(t)
    death_cross: bool           # MA50(t-1) >= MA250(t-1) and MA50(t) < MA250(t)

    @property
    def ready(self) -> bool:
        return self.z is not None

    def as_log(self) -> dict:
        r = lambda v: None if v is None else round(v, 4)
        return {"t": self.t, "price": self.price, "ma50": r(self.ma50), "ma250": r(self.ma250),
                "ma500": r(self.ma500), "sd250": r(self.sd250), "upper": r(self.upper),
                "lower": r(self.lower), "z": r(self.z),
                "last_lower_breach_t": self.last_lower_breach_t,
                "last_upper_breach_t": self.last_upper_breach_t}


class EntryIndicator:
    """Asymmetrical Bollinger entry model on 1-second closes."""

    def __init__(self, cfg: EntryConfig):
        self.cfg = cfg
        self._fast = RollingStats(cfg.fast_ma, cfg.ddof)
        self._slow = RollingStats(cfg.slow_ma, cfg.ddof)
        self._long = RollingStats(cfg.long_ma, cfg.ddof)
        self._prev_fast: float | None = None
        self._prev_slow: float | None = None
        self.last_lower_breach_t: int | None = None
        self.last_upper_breach_t: int | None = None

    def update(self, t: int, price: float) -> EntrySnapshot:
        cfg = self.cfg
        for rs in (self._fast, self._slow, self._long):
            rs.push(price)
        ma50 = self._fast.mean() if self._fast.ready else None
        ma250 = self._slow.mean() if self._slow.ready else None
        ma500 = self._long.mean() if self._long.ready else None
        sd = self._slow.std() if self._slow.ready else None

        z = upper = lower = None
        lower_breach = upper_breach = False
        if ma250 is not None and sd is not None:
            upper = ma250 + cfg.band_sd_mult * sd
            lower = ma250 - cfg.band_sd_mult * sd
            if sd > 0:
                z = (price - ma250) / sd
                # Upper and lower breaches are tracked independently.
                if z <= -cfg.breach_z:
                    lower_breach = True
                    self.last_lower_breach_t = t
                if z >= cfg.breach_z:
                    upper_breach = True
                    self.last_upper_breach_t = t

        golden = death = False
        if None not in (ma50, ma250, self._prev_fast, self._prev_slow):
            golden = self._prev_fast <= self._prev_slow and ma50 > ma250
            death = self._prev_fast >= self._prev_slow and ma50 < ma250
        self._prev_fast, self._prev_slow = ma50, ma250

        recent = (self.last_lower_breach_t is not None
                  and 0 <= t - self.last_lower_breach_t <= cfg.breach_window_s)
        return EntrySnapshot(t, price, ma50, ma250, ma500, sd, upper, lower, z,
                             lower_breach, upper_breach, self.last_lower_breach_t,
                             self.last_upper_breach_t, recent, golden, death)


@dataclass(frozen=True)
class ExitBandSnapshot:
    t: int
    price: float                 # latest 1-second BTC price
    upper: float | None          # bands as of the last completed candle
    middle: float | None
    lower: float | None
    candle_closed: bool          # a candle completed at this second
    candle_close: float | None   # that candle's close


class ExitBands:
    """Traditional Bollinger Bands on `candle_seconds` candles, `lookback` candles long.

    A candle covering [k*P, (k+1)*P) closes when the second (k+1)*P - 1 is processed,
    so a decision at that second only uses data up to that second.
    """

    def __init__(self, method: ExitMethod, ddof: int = 0):
        self.method = method
        self.period = method.candle_seconds
        self._stats = RollingStats(method.lookback, ddof)
        self.upper = self.middle = self.lower = None

    def update(self, t: int, price: float) -> ExitBandSnapshot:
        closed = (t % self.period) == self.period - 1
        if closed:
            self._stats.push(price)  # the candle close is the last 1s close in the candle
            if self._stats.ready:
                mid = self._stats.mean()
                sd = self._stats.std()
                self.middle = mid
                self.upper = mid + self.method.sd_mult * sd
                self.lower = mid - self.method.sd_mult * sd
        return ExitBandSnapshot(t, price, self.upper, self.middle, self.lower,
                                closed, price if closed else None)


class TrailingStop:
    """Trailing-stop state machine for one open YES position.

    * Activation: BTC >= upper exit band (touching the band never exits by itself).
      Once active it stays active until the position is closed.
    * Exit: after activation, BTC closes below the middle band on consecutive candles
      totalling >= confirm_seconds (12 x 5s candles for 60s). Any close at or above
      the middle band resets the timer.
    """

    def __init__(self, method: ExitMethod, side: str = "long"):
        if side not in ("long", "short"):
            raise ValueError(side)
        self.method = method
        self.side = side      # "short" mirrors the rules: activate at the lower band, exit above middle
        self.activated = not method.require_activation
        self.activated_t: int | None = None
        self.seconds_below = 0

    def update(self, snap: ExitBandSnapshot) -> bool:
        if snap.middle is None:
            return False
        if not self.activated and snap.upper is not None and (
                snap.price >= snap.upper if self.side == "long" else snap.price <= snap.lower):
            self.activated = True
            self.activated_t = snap.t
        if not self.activated or not snap.candle_closed:
            return False
        against = (snap.candle_close < snap.middle if self.side == "long"
                   else snap.candle_close > snap.middle)
        if against:
            self.seconds_below += self.method.candle_seconds
        else:
            self.seconds_below = 0
        return self.seconds_below >= self.method.confirm_seconds


class IndicatorHub:
    """Computes the entry snapshot and every exit method's band snapshot once per second."""

    def __init__(self, entry_cfg: EntryConfig, exit_methods):
        self.entry = EntryIndicator(entry_cfg)
        self.exit_bands = {m.name: ExitBands(m, entry_cfg.ddof)
                           for m in exit_methods if m.kind != "hold"}

    def update(self, t: int, price: float):
        return (self.entry.update(t, price),
                {name: b.update(t, price) for name, b in self.exit_bands.items()})
