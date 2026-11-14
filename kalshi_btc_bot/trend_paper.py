"""Trend-in YES PAPER traders on the hourly Bitcoin range bands (KXBTC). Read-only: no orders are sent.

Michael asked for these on 2026-11-04 ("put these in as two separate bots") after the trend-state scan of
Jan-Oct 2026 (2.6M band-minutes at every price level): a YES favorite bought 4-20 minutes before the hourly
close wins more often than its price says when BRTI is inside the band now AND the last 15 seconds of BRTI,
carried forward to settlement, still land inside the band ("trend-in"). The NO side showed no edge.

Common setup, checked at each minute mark with 20 down to 4 minutes left before the hourly close:
* look at the 14 bands nearest BRTI (anchor = BRTI 30 min before the close, as the range bot does), two-sided
  bands only (the open-ended end bands have no centre);
* BRTI must be inside the band, and its distance from the band centre (as a share of the half-width) is
  measured;
* slope = least-squares line through the last 15 one-second BRTI prints ($/s); projection = BRTI + slope x
  seconds to the middle of the settlement window; the projection must also be inside the band;
* the YES ask must be in the rule's price window and the YES spread at most 5c;
* buy YES, walking the book up to the top of the price window, 20 contracts, held to settlement;
* one trade per hour (the first band-minute that qualifies), as in the backtest.

Two rules (win % minus price, Jan-Aug / Sep-Oct, one trade per hour, fills at the 1-minute candle ask):
* "broad": YES 61-80c with BRTI within 80% of the centre, or 81-90c within 40%:
  2,605 trades 75.5% at 71.2c +4.3 pts / 593 trades 74.0% at 68.7c +5.4 pts, about 11 trades a day.
* "tight": YES 70-84c with BRTI within 40% of the centre:
  1,149 trades 82.6% at 76.5c +6.1 pts / 270 trades 84.8% at 76.1c +8.7 pts, about 5 trades a day.
Each runs as its own service with its own logs; nothing about the other bots changes.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .late_breakdown import kalshi_fee, log
from .range_paper import CONTRACTS, N_BANDS, RangePaper, _f, _iso, band_mid, event_ticker, hourly_markets, fill

FIRST_MINUTE, LAST_MINUTE = 20, 4   # minutes left before the close: check from 20 down to 4
SLOPE_SECONDS = 15                  # one-second BRTI prints in the trend line
MAX_SPREAD = 0.05                   # YES ask minus YES bid, at most 5c

# (low ask, high ask, max distance from the band centre as a share of the half-width), all inclusive
RULES = {
    "broad": [(0.61, 0.80, 0.80), (0.81, 0.90, 0.40)],
    "tight": [(0.70, 0.84, 0.40)],
}
BOT_NAMES = {"broad": "Trend broad", "tight": "Trend tight"}

_X = np.arange(SLOPE_SECONDS, dtype=float) - (SLOPE_SECONDS - 1) / 2
_XX = float((_X * _X).sum())


def slope_of(prints: list[float]) -> float:
    """Least-squares slope ($ per second) of consecutive one-second prints."""
    y = np.asarray(prints, dtype=float)
    return float((_X * (y - y.mean())).sum() / _XX)


def projection(price: float, slope: float, minutes_left: float) -> float:
    return price + slope * (minutes_left - 0.5) * 60


def inside(x: float, lo: float, hi: float) -> bool:
    return lo <= x <= hi + 0.01


def centre_distance(price: float, lo: float, hi: float) -> float:
    """How far from the band centre, 0 = centre, 1 = at an edge."""
    half = (hi - lo) / 2
    return abs(price - (lo + hi) / 2) / half if half > 0 else 1.0


def window_for(rule: str, ask: float | None, dist: float) -> tuple[float, float] | None:
    """The (low, high) price window the ask qualifies under, or None."""
    if ask is None:
        return None
    c = round(ask * 100)
    for lo, hi, max_dist in RULES[rule]:
        if round(lo * 100) <= c <= round(hi * 100) and dist <= max_dist + 1e-9:
            return lo, hi
    return None


class TrendPaper(RangePaper):
    def __init__(self, out, key_id, key_path, push_logs=False, contracts=CONTRACTS, rule="tight"):
        if rule not in RULES:
            raise ValueError(f"rule must be one of {sorted(RULES)}")
        super().__init__(out, key_id, key_path, push_logs, contracts)
        self.rule = rule
        self.hours_traded: set[int] = {tr["close_ts"] for tr in self.held.values()}

    def prints(self, t: int) -> list[float] | None:
        out = []
        for u in range(t - SLOPE_SECONDS + 1, t + 1):
            v = self.brti.get(u)
            if v is None:
                v = self.price_at(u)        # carry the last print forward over a missing second
                if v is None:
                    return None
            out.append(v)
        return out

    def check(self, t: int):
        close = (t // 3600 + 1) * 3600
        left = (close - t) // 60
        if left > FIRST_MINUTE or left < LAST_MINUTE or close in self.hours_traded:
            return
        price = self.price_at(t)
        prints = self.prints(t)
        if price is None or prints is None:
            log.warning("%s: no BRTI yet", _iso(t))
            return
        slope = slope_of(prints)
        proj = projection(price, slope, left)
        anchor = self.anchor.setdefault(close, self.price_at(close - 1800) or price)
        ev = event_ticker(close)
        markets = self.k.get("/markets", event_ticker=ev, limit=1000).get("markets", [])
        markets = hourly_markets(markets, ev)
        if not markets:
            log.warning("no open markets for %s", ev)
            return
        for m in sorted(markets, key=lambda mm: abs(band_mid(mm) - anchor))[:N_BANDS]:
            tk = m["ticker"]
            lo, hi = _f(m.get("floor_strike")), _f(m.get("cap_strike"))
            if tk in self.held or lo is None or hi is None or m.get("strike_type") in ("greater", "less"):
                continue
            if not inside(price, lo, hi) or not inside(proj, lo, hi):
                continue
            dist = centre_distance(price, lo, hi)
            yb, ya = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
            win = window_for(self.rule, ya, dist)
            if win is None or yb is None or ya - yb > MAX_SPREAD + 1e-9:
                continue
            self.enter(t, close, left, m, ya, price, slope, proj, dist, win[1])
            if close in self.hours_traded:
                return

    def enter(self, t, close, left, m, ask, price, slope, proj, dist, limit):   # noqa: D102
        tk = m["ticker"]
        p_yes, _sd = self.model_yes(t, price, m, left)   # settlement-aware chance, for the dashboard
        try:
            book = self.k.orderbook(tk)
        except Exception as e:
            log.warning("orderbook %s failed: %s", tk, e)
            book = {}
        other = book.get("no_bids") or []
        n, avg = fill(other, self.contracts, limit)
        row = {"time_utc": _iso(t), "ticker": tk, "side": "YES", "mins_left": left,
               "floor": m.get("floor_strike"), "cap": m.get("cap_strike"), "brti": round(price, 2),
               "slope_per_s": round(slope, 4), "projection": round(proj, 2), "centre_dist_pct": round(dist * 100, 1),
               "model": "" if p_yes is None else round(p_yes, 4),
               "ask_snapshot": ask, "book_best": round(1 - other[0][0], 4) if other else "",
               "limit": limit, "filled": n, "avg_price": round(avg, 4) if avg else "", "book": json.dumps(other[:6])}
        self._append("signals.csv", row)
        if n == 0:
            log.info("%s YES %s: ask %.2f but no fill under %.2f", _iso(t), tk, ask, limit)
            return
        fee = kalshi_fee(n, avg)
        self.held[tk] = {"time_utc": _iso(t), "ticker": tk, "side": "YES", "mins_left": left,
                         "contracts": n, "avg_price": round(avg, 4), "fee": fee,
                         "cost": round(n * avg + fee, 2), "model": "" if p_yes is None else round(p_yes, 4), "close_ts": close}
        self.hours_traded.add(close)
        self.hours_traded = {c for c in self.hours_traded if c >= close - 7200}
        self._save_open()
        log.info("PAPER BUY %d YES %s @ %.4f (%s rule, %d min left, slope %.2f $/s, %.0f%% from centre)",
                 n, tk, avg, self.rule, left, slope, dist * 100)


def from_env(out: str, push_logs: bool, contracts: int = CONTRACTS, rule: str = "tight") -> TrendPaper:
    return TrendPaper(out, os.environ.get("KALSHI_KEY_ID"), os.environ.get("KALSHI_KEY_FILE"),
                      push_logs, contracts, rule)
