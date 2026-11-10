"""Quiet-favorite YES PAPER trader on the hourly Bitcoin range bands (KXBTC). Read-only: no orders are sent.

Michael asked for it on 2026-11-05 ("make the bot for 1-18 minutes"), after the recent-move tests: a YES favorite on
a band wins more often than its price says when BTC sits inside the band and has barely moved compared with its
distance to the nearest edge ("quiet").

Rule, checked at each minute mark with 18 down to 1 minutes left before the hourly close:
* the 14 bands nearest BRTI (anchor = BRTI 30 min before the close, as the other band bots), two-sided bands only;
* BRTI inside the band; cushion = distance from BRTI to the nearest band edge;
* lookback = the time left (12 min left -> BRTI now vs 12 min ago); quiet = |move| < 0.25 x cushion;
* YES ask 61-95c and YES spread at most 5c;
* buy YES, walking the book up to 95c, 20 contracts, held to settlement; one trade per hour (first band-minute
  that qualifies).

Backtest (Dec 2025 - Oct 2026, 1-minute candle ask, fee 7% x C x P x (1-P), one trade per hour):
Dec-Aug 2,425 trades 81.6% at 77.7c, +2.8c a contract (s.e. 0.8); Sep-Oct 594 trades 80.0% at 76.3c, +2.5c (s.e. 1.6);
about 10 trades a day, 10 of 10 months positive. Runs as its own service with its own logs.
"""

from __future__ import annotations

import json
import os

from .late_breakdown import kalshi_fee, log
from .range_paper import CONTRACTS, N_BANDS, RangePaper, _f, _iso, band_mid, event_ticker, fill

FIRST_MINUTE, LAST_MINUTE = 18, 1
LOW, HIGH = 0.61, 0.95
QUIET = 0.25                       # |move over the time left| must be under this share of the cushion
MAX_SPREAD = 0.05
BOT_NAME = "Quiet"


def cushion(price: float, lo: float, hi: float) -> float | None:
    """Distance to the nearest band edge, None if BRTI is outside the band (cap strike is e.g. 86399.99)."""
    top = hi + 0.01
    if price < lo or price > top:
        return None
    return min(price - lo, top - price)


def is_quiet(move: float, cush: float) -> bool:
    return cush > 0 and abs(move) < QUIET * cush


def price_ok(ask: float | None) -> bool:
    return ask is not None and round(LOW * 100) <= round(ask * 100) <= round(HIGH * 100)


class QuietPaper(RangePaper):
    def __init__(self, out, key_id, key_path, push_logs=False, contracts=CONTRACTS):
        super().__init__(out, key_id, key_path, push_logs, contracts)
        self.hours_traded: set[int] = {tr["close_ts"] for tr in self.held.values()}

    def check(self, t: int):
        close = (t // 3600 + 1) * 3600
        left = (close - t) // 60
        if left > FIRST_MINUTE or left < LAST_MINUTE or close in self.hours_traded:
            return
        price, before = self.price_at(t), self.price_at(t - 60 * left)
        if price is None or before is None:
            log.warning("%s: no BRTI yet", _iso(t))
            return
        move = price - before
        anchor = self.anchor.setdefault(close, self.price_at(close - 1800) or price)
        ev = event_ticker(close)
        markets = self.k.get("/markets", event_ticker=ev, limit=1000).get("markets", [])
        markets = [m for m in markets if m.get("status") in ("active", "open")]
        if not markets:
            log.warning("no open markets for %s", ev)
            return
        for m in sorted(markets, key=lambda mm: abs(band_mid(mm) - anchor))[:N_BANDS]:
            lo, hi = _f(m.get("floor_strike")), _f(m.get("cap_strike"))
            if m["ticker"] in self.held or lo is None or hi is None or m.get("strike_type") in ("greater", "less"):
                continue
            c = cushion(price, lo, hi)
            if c is None or not is_quiet(move, c):
                continue
            yb, ya = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
            if not price_ok(ya) or yb is None or ya - yb > MAX_SPREAD + 1e-9:
                continue
            self.enter(t, close, left, m, ya, price, move, c)
            if close in self.hours_traded:
                return

    def enter(self, t, close, left, m, ask, price, move, cush):   # noqa: D102
        tk = m["ticker"]
        p_yes, _sd = self.model_yes(t, price, m, left)   # settlement-aware chance, for the dashboard
        try:
            book = self.k.orderbook(tk)
        except Exception as e:
            log.warning("orderbook %s failed: %s", tk, e)
            book = {}
        other = book.get("no_bids") or []
        n, avg = fill(other, self.contracts, HIGH)
        row = {"time_utc": _iso(t), "ticker": tk, "side": "YES", "mins_left": left,
               "floor": m.get("floor_strike"), "cap": m.get("cap_strike"), "brti": round(price, 2),
               "move": round(move, 2), "cushion": round(cush, 2), "move_share": round(abs(move) / cush, 3),
               "model": "" if p_yes is None else round(p_yes, 4),
               "ask_snapshot": ask, "book_best": round(1 - other[0][0], 4) if other else "",
               "limit": HIGH, "filled": n, "avg_price": round(avg, 4) if avg else "", "book": json.dumps(other[:6])}
        self._append("signals.csv", row)
        if n == 0:
            log.info("%s YES %s: ask %.2f but no fill under %.2f", _iso(t), tk, ask, HIGH)
            return
        fee = kalshi_fee(n, avg)
        self.held[tk] = {"time_utc": _iso(t), "ticker": tk, "side": "YES", "mins_left": left,
                         "contracts": n, "avg_price": round(avg, 4), "fee": fee,
                         "cost": round(n * avg + fee, 2), "model": "" if p_yes is None else round(p_yes, 4), "close_ts": close}
        self.hours_traded.add(close)
        self.hours_traded = {c for c in self.hours_traded if c >= close - 7200}
        self._save_open()
        log.info("PAPER BUY %d YES %s @ %.4f (quiet: moved %.0f vs cushion %.0f, %d min left)",
                 n, tk, avg, move, cush, left)


def from_env(out: str, push_logs: bool, contracts: int = CONTRACTS) -> QuietPaper:
    return QuietPaper(out, os.environ.get("KALSHI_KEY_ID"), os.environ.get("KALSHI_KEY_FILE"), push_logs, contracts)
