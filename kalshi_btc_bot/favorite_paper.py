"""YES-favorite PAPER trader on the hourly Bitcoin range bands (KXBTC), read-only, no orders are sent.

Michael asked for this on 2026-11-04 ("add it") after the all-price-levels study of Dec 2025-Oct 2026:
YES on a range band priced 75-84c in the last 3 minutes of the hour won 83.2% of the time in both the
fit period (Dec-Aug, 1,104 trades) and out of sample (Sep-Oct, 481 trades), paying 79.6c on average,
about +2.5c per contract after the taker fee. No model is involved.

The rule, checked at each minute mark with 3, 2 and 1 minutes left before the hourly close:
* look at the 14 bands nearest BRTI (same ladder the range bot watches);
* for each band not already held: if the YES ask is 75c to 84c, buy YES, walking the book up to 84c,
  20 contracts, held to settlement. One position per band per hour, first qualifying minute wins;
* added 2026-11-04 (Michael: "do it"): skip the band when its YES bid-ask spread is 3c or more. The loser
  study found the 3-5c-spread half of these trades made nothing in both periods (keep half: +5.7c / +5.3c).

Settlement gate (added 2026-11-10, after the 60-second-average study): Kalshi settles on the 60-s BRTI
average, so with 1-3 minutes left a band whose edge is close to the price in volatility terms is a coin
flip whatever the ask says. The bot now also requires BRTI inside the band with a cushion to the nearer
edge of at least MIN_CUSHION_SD standard deviations of the settlement move (price x 100-minute 1-min
volatility x sqrt(minutes left - 40 s)). Dec 2025-Oct 2026 replay: trades with a cushion under 0.5 sd
won 77% at 78.6c (-1.5c), 0.5-1.5 sd won 83% at 80c (+3.3c); 116 trades were taken with BRTI already
outside the band (18% won). The range model's settlement-aware YES chance is logged as `model`.

Backtest caveats: it used the 1-minute candle's closing ask, so a fill at that price is assumed; the
range books are thin, and the edge is 2-3c, so fills decide everything. Own logs
(paper-logs-favorite branch); nothing about the other bots changes.
"""

from __future__ import annotations

import json
import os

from .late_breakdown import kalshi_fee, log
from .range_paper import (CONTRACTS, MAX_SPREAD_LOW, N_BANDS, RangePaper, _f, _iso, band_mid, event_ticker, fill,
                          spread_ok)

LOW, HIGH = 0.75, 0.84      # YES ask window, inclusive, to the cent
MINUTES = 3                 # start checking with this many minutes left (then 2, then 1)
MIN_CUSHION_SD = 0.5        # BRTI must sit inside the band, at least this many settlement sds from the nearer edge


def cushion_sd(price: float, lo: float | None, hi: float | None, sd: float | None) -> float | None:
    """Distance from BRTI to the nearer band edge in settlement standard deviations; negative when
    BRTI is outside the band; None when there is no volatility reading (never skip on a missing sd)."""
    if sd is None or not sd > 0:
        return None
    top = None if hi is None else hi + 0.01
    d = min(price - lo if lo is not None else float("inf"), top - price if top is not None else float("inf"))
    return d / sd


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
            ya, yb = _f(m.get("yes_ask_dollars")), _f(m.get("yes_bid_dollars"))
            if not favorite_ok(ya):
                continue
            if not spread_ok(ya, yb):
                log.info("%s YES %s: ask %.2f in window but spread %s is over %dc, skipped", _iso(t), tk, ya,
                         f"{round((ya - yb) * 100)}c" if yb is not None else "unknown", round(MAX_SPREAD_LOW * 100))
                self._append("skipped.csv", {"time_utc": _iso(t), "ticker": tk, "side": "YES", "mins_left": left,
                                             "ask_snapshot": ya, "bid_snapshot": yb})
                continue
            p_yes, sd = self.model_yes(t, price, m, left)
            lo, hi = self.band_edges(m)
            cush = cushion_sd(price, lo, hi, sd)
            if cush is not None and cush < MIN_CUSHION_SD:
                self._append("skips.csv", {"time_utc": _iso(t), "ticker": tk, "mins_left": left, "ask": ya,
                                           "brti": round(price, 2), "floor": m.get("floor_strike"),
                                           "cap": m.get("cap_strike"), "cushion_sd": round(cush, 2),
                                           "model": "" if p_yes is None else round(p_yes, 4),
                                           "reason": "outside band" if cush < 0 else f"cushion {cush:.2f} sd < {MIN_CUSHION_SD}"})
                continue
            self.enter(t, close, left, m, ya, price, p_yes, cush)

    def enter(self, t, close, left, m, ask, price, p_yes=None, cush=None):   # noqa: D102 (simpler signature than the range bot's)
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
               "model": "" if p_yes is None else round(p_yes, 4), "cushion_sd": "" if cush is None else round(cush, 2),
               "ask_snapshot": ask, "book_best": round(1 - other[0][0], 4) if other else "",
               "limit": HIGH, "filled": n, "avg_price": round(avg, 4) if avg else "", "book": json.dumps(other[:6])}
        self._append("signals.csv", row)
        if n == 0:
            log.info("%s YES %s: ask %.2f but no fill under %.2f", _iso(t), tk, ask, HIGH)
            return
        fee = kalshi_fee(n, avg)
        self.held[tk] = {"time_utc": _iso(t), "ticker": tk, "side": "YES", "mins_left": left,
                         "contracts": n, "avg_price": round(avg, 4), "fee": fee,
                         "cost": round(n * avg + fee, 2), "model": "" if p_yes is None else round(p_yes, 4),
                         "close_ts": close}
        self._save_open()
        log.info("PAPER BUY %d YES %s @ %.4f (%d min left)", n, tk, avg, left)


def from_env(out: str, push_logs: bool, contracts: int = CONTRACTS) -> FavoritePaper:
    return FavoritePaper(out, os.environ.get("KALSHI_KEY_ID"), os.environ.get("KALSHI_KEY_FILE"),
                         push_logs, contracts)
