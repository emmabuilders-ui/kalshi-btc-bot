"""Fast-push PAPER trader on the hourly Bitcoin RANGE bands (KXBTC), read-only, no orders are sent.

Michael asked for this on 2026-11-04: the fast-push rule that runs on the 15-minute markets, pointed
at the hourly range ladder as a second market. It is a separate service with its own logs, so the
15-minute fast-push bot is untouched.

The rule is the 15-minute bot's rule, unchanged (see fast_push.py):
* levels fixed at each 15-minute session open; break = BRTI $5 through a level; fast push = the move
  over the last 15 s in the break direction is at least 2x a normal 15-s move; window 60-150 s left.
Because the range contracts settle on the hour, only the LAST 15-minute session of each hour
(xx:45 to xx:00) can trigger a range trade: 60-150 s before the hour close.

What it buys, about 1 s after a qualifying break:
* YES on the band BRTI is in now (the band it pushed into), and
* NO on the band BRTI was in 15 s earlier (the band it pushed out of),
but only if the push actually crossed a band edge (otherwise nothing has repriced), and only if that
side's ask is under 50c or 75-96c. 20 contracts each, walking the book up to the price band's limit,
held to settlement. At most one trade per band per hour and one break judged per side per session.
Skipped if the decision comes more than 5 s after the break (the lag is the edge).

Untested in backtests: the hourly tests use 1-minute quotes, too coarse to see a 1-5 s lag. This paper
run is the test, and the thing to watch is fills: range books are thinner than the 15-minute book.
"""

from __future__ import annotations

import json
import os
import time

from .fast_push import (CONTRACTS, MAX_LAG, PUSH_MULT, PUSH_SECS, WINDOW, FastPushPaper, FPSession,
                        minute_vol, price_ok, push_reading)
from .late_breakdown import _iso, kalshi_fee, log
from .range_paper import _f, event_ticker, fill
from .settlement import band_prob, side_prob, sigma_1s, worth_it


def band_edges(m: dict) -> tuple[float | None, float | None]:
    """(floor, cap) of a range market; None for the open end of a tail band."""
    lo, hi = _f(m.get("floor_strike")), _f(m.get("cap_strike"))
    if m.get("strike_type") == "greater":
        hi = None
    elif m.get("strike_type") == "less":
        lo = None
    return lo, hi


def band_of(markets: list[dict], price: float) -> dict | None:
    """The band whose range holds `price` (floor <= price < cap + 0.01)."""
    for m in markets:
        lo, hi = band_edges(m)
        if (lo is None or price >= lo) and (hi is None or price < hi + 0.01):
            return m
    return None


class FastPushRangePaper(FastPushPaper):
    def __init__(self, out, contracts: int = CONTRACTS, key_id=None, key_path=None,
                 push_logs: bool = False, max_lag: float | None = MAX_LAG):
        super().__init__(out, contracts, key_id, key_path, push_logs, max_lag)
        self.traded: dict[str, int] = {}   # range ticker -> hour close it was traded for

    def range_markets(self, close_ts: int) -> list[dict]:
        ev = event_ticker(close_ts)
        markets = self.k.get("/markets", event_ticker=ev, limit=1000).get("markets", [])
        return [m for m in markets if m.get("status") in ("active", "open")]

    def judge(self, s: FPSession, t: int, name: str, side: str, lev: float, price: float):
        left = s.close_ts - (t + 1)
        row = {"break_utc": _iso(t), "session": s.ticker, "side": side.upper(), "level": name,
               "level_price": round(lev, 2), "secs_left": left, "brti": round(price, 2)}
        before = self.series.value(t - PUSH_SECS)
        vol = minute_vol(self.minute_closes(t))
        push = push_reading(price, before, vol, side) if before and vol else None
        row["push"] = round(push, 2) if push is not None else ""
        lag = time.time() - t
        row["lag_s"] = round(lag, 1)
        reason = ""
        if s.close_ts % 3600:
            reason = "not the last 15 minutes of the hour"
        elif not WINDOW[0] <= left < WINDOW[1]:
            reason = "outside 60-150 s window"
        elif side in s.done:
            reason = f"{side.upper()} already judged this session"
        elif push is None:
            reason = "no volatility reading"
        elif push < PUSH_MULT:
            reason = f"push {push:.2f} < {PUSH_MULT}"
        else:
            s.done.add(side)
            if self.max_lag is not None and lag > self.max_lag:
                reason = f"late by {lag:.0f}s"
        if reason:
            self._append("breaks.csv", {**row, "decision": "skip: " + reason})
            return
        try:
            markets = self.range_markets(s.close_ts)
        except Exception as e:
            log.warning("range markets failed: %s", e)
            markets = []
        now_band = band_of(markets, price)
        from_band = band_of(markets, before) if before is not None else None
        if not markets or now_band is None:
            self._append("breaks.csv", {**row, "decision": "skip: no range markets"})
            return
        if from_band is not None and from_band["ticker"] == now_band["ticker"]:
            self._append("breaks.csv", {**row, "decision": "skip: push stayed inside "
                                        f"{band_label(now_band)} (BRTI {before:.2f} -> {price:.2f})"})
            return
        self._append("breaks.csv", {**row, "decision": f"judge: into {band_label(now_band)}"
                                    + (f", out of {band_label(from_band)}" if from_band else "")})
        cands = [("YES", now_band)]
        if from_band is not None:
            cands.append(("NO", from_band))
        for bside, m in cands:
            self.try_band(s, t, name, left, push, lag, price, bside, m)

    def try_band(self, s: FPSession, t: int, name: str, left: int, push, lag, price: float, bside: str, m: dict):
        tk = m["ticker"]
        lo, hi = band_edges(m)
        # settlement-aware chance that the band wins (60-s BRTI average; added 2026-11-10)
        p_band = band_prob(price, lo, hi, left, sigma_1s(self.series.array(t - 900, t + 1)))
        p_side = side_prob(p_band, bside)
        row = {"break_utc": _iso(t), "ticker": tk, "floor": lo, "cap": hi, "side": bside, "level": name,
               "secs_left": left, "brti": round(price, 2), "push": round(push, 2), "lag_s": round(lag, 1),
               "model": round(p_side, 4)}
        reason = ""
        if self.traded.get(tk) == s.close_ts:
            reason = "band already traded this hour"
        book = None
        try:
            book = self.k.orderbook(tk)
        except Exception as e:
            log.warning("orderbook %s failed: %s", tk, e)
        bids = (book or {}).get("no_bids" if bside == "YES" else "yes_bids") or []
        ask = round(1 - bids[0][0], 4) if bids else None
        row.update(ask=ask if ask is not None else "", ask_size=bids[0][1] if bids else "",
                   book=json.dumps(bids[:8]) if bids else "", book_utc=_iso(time.time()))
        ok, limit = price_ok(ask)
        if not reason:
            if ask is None:
                reason = "empty book"
            elif not ok:
                reason = f"{bside} ask {ask:.2f} not under 50c or 75-96c"
            elif not worth_it(p_side, ask):
                reason = f"settlement chance {p_side:.2f} under ask {ask:.2f} + fee"
        row["decision"] = f"BUY {bside}" if not reason else "skip: " + reason
        self._append("signals.csv", row)
        log.info("%s %s %s push %.2f ask %s -> %s", tk, bside, name, push, ask, row["decision"])
        if reason:
            return
        n, avg = fill(bids, self.contracts, limit)
        if not n:
            return
        self.traded[tk] = s.close_ts
        fee = kalshi_fee(n, avg)
        self.unsettled.append({"time_utc": _iso(t + 1), "ticker": tk, "side": bside, "floor": lo, "cap": hi,
                               "level": name, "secs_left": left, "push": round(push, 2), "lag_s": round(lag, 1),
                               "contracts": n, "avg_price": round(avg, 4), "fee": fee,
                               "cost": round(n * avg + fee, 2), "model": round(p_side, 4), "close_ts": s.close_ts,
                               "result": "", "won": "", "pnl": ""})
        log.info("PAPER BUY %d %s %s @ %.2f (fee %.2f)", n, bside, tk, avg, fee)

    def settle(self, now: int):
        # range markets settle a minute or two after the hour; the base loop retries until a result is in
        if now - self._last_settle < 30:
            return
        for tr in self.unsettled:
            if now < tr["close_ts"] + 60:
                return  # wait a full minute after the close before asking
        super().settle(now)


def band_label(m: dict) -> str:
    lo, hi = band_edges(m)
    if lo is None:
        return f"below {hi + 0.01:,.0f}"
    if hi is None:
        return f"above {lo:,.0f}"
    return f"{lo:,.0f}-{hi + 0.01:,.0f}"


def from_env(out: str, push_logs: bool, contracts: int = CONTRACTS) -> FastPushRangePaper:
    return FastPushRangePaper(out, contracts, os.environ.get("KALSHI_KEY_ID"), os.environ.get("KALSHI_KEY_FILE"),
                              push_logs)
