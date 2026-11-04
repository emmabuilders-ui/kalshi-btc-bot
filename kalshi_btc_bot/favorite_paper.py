"""YES-favorite PAPER trader on the hourly Bitcoin range bands (KXBTC), read-only, no orders are sent.

Michael asked for this on 2026-11-04 ("add it") after the all-price-levels study of Dec 2025-Oct 2026:
YES on a range band priced 75-84c in the last 3 minutes of the hour won 83.2% of the time in both the
fit period (Dec-Aug, 1,104 trades) and out of sample (Sep-Oct, 481 trades), paying 79.6c on average,
about +2.5c per contract after the taker fee. No model is involved.

The rule, checked at each minute mark with 3, 2 and 1 minutes left before the hourly close:
* look at the 14 bands nearest BRTI (same ladder the range bot watches);
* for each band not already held: if the YES ask is 75c to 84c, buy YES, walking the book up to 84c,
  20 contracts, held to settlement. One position per band per hour, first qualifying minute wins.

Backtest caveats: it used the 1-minute candle's closing ask, so a fill at that price is assumed; the
range books are thin, and the edge is 2-3c, so fills decide everything. Own logs
(paper-logs-favorite branch); nothing about the other bots changes.
"""

from __future__ import annotations

import json
import os

from .late_breakdown import kalshi_fee, log
from .range_paper import CONTRACTS, N_BANDS, RangePaper, _f, _iso, band_mid, event_ticker, fill

LOW, HIGH = 0.75, 0.84      # YES ask window, inclusive, to the cent
MINUTES = 3                 # start checking with this many minutes left (then 2, then 1)


def favorite_ok(ask: float | None) -> bool:
    if ask is None:
        return False
    c = round(ask * 100)
    return round(LOW * 100) <= c <= round(HIGH * 100)


class FavoritePaper(RangePaper):
    def check(self, t: int):
        close = (t // 3600 + 1) * 3600
        left = (close - t) // 60
        if left > MINUTES or left < 1:
            return
        price = self.price_at(t)
        if price is None:
            log.warning("%s: no BRTI yet", _iso(t))
            return
        anchor = self.anchor.setdefault(close, self.price_at(close - 1800) or price)
        ev = event_ticker(close)
        markets = self.k.get("/markets", event_ticker=ev, limit=1000).get("markets", [])
        markets = [m for m in markets if m.get("status") in ("active", "open")]
        if not markets:
            log.warning("no open markets for %s", ev)
            return
        for m in sorted(markets, key=lambda mm: abs(band_mid(mm) - anchor))[:N_BANDS]:
            tk = m["ticker"]
            if tk in self.held:
                continue
            ya = _f(m.get("yes_ask_dollars"))
            if favorite_ok(ya):
                self.enter(t, close, left, m, ya, price)

    def enter(self, t, close, left, m, ask, price):   # noqa: D102 (simpler signature than the range bot's)
        tk = m["ticker"]
        try:
            book = self.k.orderbook(tk)
        except Exception as e:
            log.warning("orderbook %s failed: %s", tk, e)
            book = {}
        other = book.get("no_bids") or []
        n, avg = fill(other, self.contracts, HIGH)
        row = {"time_utc": _iso(t), "ticker": tk, "side": "YES", "mins_left": left,
               "floor": m.get("floor_strike"), "cap": m.get("cap_strike"), "brti": round(price, 2),
               "ask_snapshot": ask, "book_best": round(1 - other[0][0], 4) if other else "",
               "limit": HIGH, "filled": n, "avg_price": round(avg, 4) if avg else "", "book": json.dumps(other[:6])}
        self._append("signals.csv", row)
        if n == 0:
            log.info("%s YES %s: ask %.2f but no fill under %.2f", _iso(t), tk, ask, HIGH)
            return
        fee = kalshi_fee(n, avg)
        self.held[tk] = {"time_utc": _iso(t), "ticker": tk, "side": "YES", "mins_left": left,
                         "contracts": n, "avg_price": round(avg, 4), "fee": fee,
                         "cost": round(n * avg + fee, 2), "close_ts": close}
        self._save_open()
        log.info("PAPER BUY %d YES %s @ %.4f (%d min left)", n, tk, avg, left)


def from_env(out: str, push_logs: bool, contracts: int = CONTRACTS) -> FavoritePaper:
    return FavoritePaper(out, os.environ.get("KALSHI_KEY_ID"), os.environ.get("KALSHI_KEY_FILE"),
                         push_logs, contracts)
