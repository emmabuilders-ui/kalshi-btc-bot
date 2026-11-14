"""PAPER trader for Kalshi's 5 PM ET Bitcoin range market (daily and weekly), YES side only.

Kalshi's KXBTC series has an hourly $100-band range market every hour EXCEPT the 5 PM ET close, where the
event (KXBTC-yyMONdd17) is the daily "BTC price range at 5pm" market (listed the day before, $250 bands) or,
on Fridays, the weekly one (listed a week before, $500 bands). The hourly-band bots traded it by accident until
2026-11-14 and did well on a small sample (29 trades, 28 won, mostly 90c+ favorites). PR #41 makes them skip it;
this bot trades it on purpose so the record can be kept on its own count.

Research (strategy/five_pm_range/, 2026-11-14, Dec 2025 - Oct 2026, 284 daily + 46 weekly events): Kalshi prices
these markets as well as the hourly ones. Taker buys at any price and time left lose 1-4c after fees on both sides;
the hourly range bot's YES rule replayed on the last 30 minutes of these events was the only rule positive in both
periods: 60 trades 95.0% won at 90.3c (+4.1c) Dec-Aug, 18 trades 94.4% at 90.0c (+3.8c) Sep-Oct. The NO side and
the 75-89c "rebuilt >= ask" account went negative Sep-Oct, so they are not traded here.

Rule (the range bot's YES rule, unchanged):
* only the hour whose close is 5 PM Eastern; checks every minute from 30 minutes left; the 14 bands nearest the
  BRTI price 30 minutes before the close, "between" bands only (no tails);
* buy YES when the ask is 75-89c and the model is 10+ points above it (spread 2c or less), or the ask is 90-99c and
  the model is 2+ points above it; walk the book up to model minus the gap; 20 contracts; one position per band;
  held to settlement (BRTI 60-second average at 5 PM);
* the model is the frozen range model (range_cdf.json, 40-s settlement shrink), same as the range bot.
No orders are ever sent.

Decision rule (docs/paper_test_decision_rule.md): retire under +1c win %-minus-price after fees at 300 trades.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from .late_breakdown import kalshi_fee, log
from .range_paper import (CONTRACTS, ET, MAX_SPREAD_LOW, N_BANDS, RangePaper, _f, _iso, band_mid, event_ticker,
                          fill, price_limit, qualifies, spread_ok)

WINDOW = 30                 # minutes left when checks start (what the hourly backtests could test)
MIN_LIFE = 2 * 3600         # a market open for longer than this before its close is the daily/weekly one


def is_five_pm_close(close_ts: int) -> bool:
    """True when the hour closing at `close_ts` closes at 5 PM Eastern (the daily/weekly market's slot)."""
    return datetime.fromtimestamp(int(close_ts), ET).hour == 17


def market_life(m: dict) -> float | None:
    """Seconds from a market's open_time to its close_time, or None when either is missing."""
    try:
        o = datetime.fromisoformat(str(m["open_time"]).replace("Z", "+00:00"))
        c = datetime.fromisoformat(str(m["close_time"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        return None
    return (c - o).total_seconds()


def market_kind(m: dict) -> str | None:
    """'daily' or 'weekly' for a 5 PM market, None for an hourly market or one without open/close times."""
    life = market_life(m)
    if life is None or life <= MIN_LIFE:
        return None
    return "weekly" if life > 48 * 3600 else "daily"


def daily_markets(markets: list[dict]) -> list[dict]:
    """Open 'between' bands of the daily/weekly market (hourly markets and tail contracts are dropped)."""
    out = []
    for m in markets:
        if m.get("status") not in ("active", "open") or market_kind(m) is None:
            continue
        if m.get("floor_strike") is None or m.get("cap_strike") is None:
            continue
        out.append(m)
    return out


class FivePmPaper(RangePaper):
    def check(self, t: int):
        close = (t // 3600 + 1) * 3600
        if not is_five_pm_close(close):
            return
        left = (close - t) // 60
        if left > WINDOW or left < 1:
            return
        price = self.price_at(t)
        closes = self.minute_closes(t)
        v = self.model.vol(closes) if len(closes) else None
        if price is None or v is None:
            log.warning("%s: not enough BRTI history yet", _iso(t))
            return
        s100, vb = v
        anchor = self.anchor.setdefault(close, self.price_at(close - 1800) or price)
        ev = event_ticker(close)
        markets = daily_markets(self.k.get("/markets", event_ticker=ev, limit=1000).get("markets", []))
        if not markets:
            log.warning("no open daily/weekly markets for %s", ev)
            return
        for m in sorted(markets, key=lambda mm: abs(band_mid(mm) - anchor))[:N_BANDS]:
            tk = m["ticker"]
            if tk in self.held:
                continue
            lo, hi = _f(m.get("floor_strike")), _f(m.get("cap_strike"))
            p_yes = self.model.p_inside(price, lo, hi, left, s100, vb)
            yb, ya = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
            if not qualifies(p_yes, ya):
                continue
            if not spread_ok(ya, yb):
                log.info("%s YES %s: ask %.2f qualifies but spread %s is over %dc, skipped", _iso(t), tk, ya,
                         f"{round((ya - yb) * 100)}c" if yb is not None else "unknown", round(MAX_SPREAD_LOW * 100))
                self._append("skipped.csv", {"time_utc": _iso(t), "ticker": tk, "side": "YES", "mins_left": left,
                                             "ask_snapshot": ya, "bid_snapshot": yb, "model": round(p_yes, 4)})
                continue
            self.enter(t, close, left, m, "YES", p_yes, ya, price, s100, vb)

    def enter(self, t, close, left, m, side, mp, ask, price, s100, vb):
        tk = m["ticker"]
        try:
            book = self.k.orderbook(tk)
        except Exception as e:
            log.warning("orderbook %s failed: %s", tk, e)
            book = {}
        other = book.get("no_bids") or []           # buying YES takes the NO bids
        limit = price_limit(mp)
        n, avg = fill(other, self.contracts, limit)
        kind = market_kind(m) or ""
        row = {"time_utc": _iso(t), "ticker": tk, "side": side, "market": kind, "mins_left": left,
               "floor": m.get("floor_strike"), "cap": m.get("cap_strike"), "brti": round(price, 2),
               "model": round(mp, 4), "ask_snapshot": ask, "book_best": round(1 - other[0][0], 4) if other else "",
               "limit": limit, "filled": n, "avg_price": round(avg, 4) if avg else "",
               "vol_1m": round(s100, 6), "vol_state": vb, "book": json.dumps(other[:6])}
        self._append("signals.csv", row)
        if n == 0:
            log.info("%s YES %s: signal at %.2f but no fill under %.2f", _iso(t), tk, ask, limit)
            return
        fee = kalshi_fee(n, avg)
        self.held[tk] = {"time_utc": _iso(t), "ticker": tk, "side": side, "market": kind, "mins_left": left,
                         "contracts": n, "avg_price": round(avg, 4), "fee": fee,
                         "cost": round(n * avg + fee, 2), "model": round(mp, 4), "close_ts": close}
        self._save_open()
        log.info("PAPER BUY %d YES %s (%s market) @ %.4f (model %.3f, %d min left)", n, tk, kind, avg, mp, left)


def from_env(out: str, push_logs: bool, contracts: int = CONTRACTS) -> FivePmPaper:
    return FivePmPaper(out, os.environ.get("KALSHI_KEY_ID"), os.environ.get("KALSHI_KEY_FILE"), push_logs, contracts)
