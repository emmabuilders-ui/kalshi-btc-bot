"""Fast-push level-break PAPER trader for Kalshi KXBTC15M (read-only, no orders are sent).

The rule (Michael, 2026-11-01), tested on Dec 2025 - Oct 2026 with prices rebuilt from every trade:

* Levels, fixed at each 15-minute session open from 1-second BRTI. Supports are below the
  price at open by more than $10 (nearest one of each kind), resistances above it:
  hard bottom/top (zigzag clusters, as in the Breakdown Bot), previous 15 min, 60 min and 4 h
  low/high, previous UTC day high/low, today's high/low so far, today's open, floor pivots
  P, R1/S1, R2/S2 (from the previous UTC day), and round $500 / $1,000 numbers.
* Break: BRTI trades $5 through a level. Only a level's first break counts.
* Fast push: at the break second, the move over the last 15 s in the break direction is at
  least 2x a normal 15-s move (1-minute volatility of the last 100 minutes x sqrt(1/4)).
* Window: 60-179 s left. Support break buys NO, resistance break buys YES, the first
  qualifying break per side per session, held to settlement.
* Entry about 1 s after the break (the live BRTI feed is about 1 s behind). The side's ask
  must be under 50c or 75-96c. 20 contracts, walking the book up to the band's limit.

Backtest with this exact timing (push read at the break second, price 1 s later, rebuilt from
every Kalshi trade, Dec 10 2025 - Oct 25 2026): 75-96c 175 trades 97% @ 90c +6.9c/contract
(Dec-Aug), 58 trades 93% @ 88c +4.3c (Sep-Oct); under 50c only 13 trades (+35c, unproven).
About 0.8 trades a day.

Why it works: BTC does not win more often after a fast push than the distance from the strike
says (checked on 2025 BRTI). The edge is Kalshi's price taking a few seconds to catch up: P/L fell
from +5.8c at 1 s to +3.6c at 5 s. So speed and fills are what this paper test measures. Every
signal is logged with the wall-clock delay and the book.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

from .late_breakdown import (BREAK, GAP, SESSION, WARMUP, LateBreakdownPaper, hard_levels,
                             kalshi_fee, _iso)
from .range_paper import fill
from .settlement import settle_prob, side_prob, sigma_1s, worth_it

log = logging.getLogger("kalshi_btc_bot")

PUSH_SECS, PUSH_MULT = 15, 2.0
WINDOW = (60, 180)          # seconds left at entry: 60 <= left < 180
LOW_MAX, HIGH_MIN, HIGH_MAX = 0.4999, 0.75, 0.96
CONTRACTS = 20
MAX_LAG = 5                 # skip if the decision comes more than 5 s after the break (wall clock)


def all_levels(b: np.ndarray, open_ts: int) -> dict[str, tuple[str, float]]:
    """Levels for a session. b holds 1-second prices ending at the second before open_ts
    (b[-1] = last price before the open; b[-k] = second open_ts - k). Returns
    {name: (side, price)}; side "no" for supports below the open, "yes" for resistances above."""
    n = len(b)
    s0 = float(b[-1])
    at = lambda t: n - (open_ts - t)  # index of absolute second t
    cand: dict[str, list[float]] = {}
    cand["hard"] = list(hard_levels(b, n))
    for name, secs in (("prev15", 900), ("prev60", 3600), ("prev4h", 14400)):
        w = b[-secs:]
        cand[name] = [float(w.min()), float(w.max())]
    day0 = open_ts // 86400 * 86400
    if open_ts - day0 >= 60 and at(day0 - 86400) >= 0:
        prev = b[at(day0 - 86400):at(day0)]
        ph, pl, pc = float(prev.max()), float(prev.min()), float(prev[-1])
        p = (ph + pl + pc) / 3
        today = b[at(day0):]
        cand["prevday"] = [ph, pl]
        cand["today"] = [float(today.max()), float(today.min())]
        cand["day_open"] = [float(b[at(day0)])]
        cand["pivot_P"] = [p]
        cand["pivot_R1S1"] = [2 * p - pl, 2 * p - ph]
        cand["pivot_R2S2"] = [p + (ph - pl), p - (ph - pl)]
    cand["round1000"] = [math.floor(s0 / 1000) * 1000, math.ceil(s0 / 1000) * 1000]
    cand["round500"] = [math.floor(s0 / 500) * 500, math.ceil(s0 / 500) * 500]
    out = {}
    for name, vals in cand.items():
        v = np.asarray(vals, float)
        below, above = v[v < s0 - GAP], v[v > s0 + GAP]
        if len(below):
            out[f"{name}_low"] = ("no", float(below.max()))
        if len(above):
            out[f"{name}_high"] = ("yes", float(above.min()))
    return out


def minute_vol(closes: np.ndarray) -> float | None:
    """Std of 1-minute log returns from the last 101 minute closes."""
    if len(closes) < 101 or np.isnan(closes).any():
        return None
    v = float(np.std(np.diff(np.log(closes[-101:])), ddof=1))
    return v if v > 0 else None


def push_reading(p_now: float, p_before: float, vol: float, side: str) -> float:
    """Move over the last 15 s in the break direction, in units of a normal 15-s move."""
    sg = 1 if side == "yes" else -1
    return sg * math.log(p_now / p_before) / (vol * math.sqrt(PUSH_SECS / 60))


def price_ok(ask: float | None) -> tuple[bool, float]:
    """Allowed ask bands; returns (ok, the limit to walk the book to)."""
    if ask is None:
        return False, 0.0
    if ask <= LOW_MAX:
        return True, LOW_MAX
    if HIGH_MIN <= ask <= HIGH_MAX:
        return True, HIGH_MAX
    return False, 0.0


@dataclass
class FPSession:
    ticker: str
    strike: float
    open_ts: int
    close_ts: int
    s0: float
    levels: dict                                  # name -> (side, price)
    broken: set = field(default_factory=set)      # levels whose first break was judged
    done: set = field(default_factory=set)        # sides already paper-traded
    live: bool = True


class FastPushPaper(LateBreakdownPaper):
    def __init__(self, out, contracts: int = CONTRACTS, key_id=None, key_path=None,
                 push_logs: bool = False, max_lag: float | None = MAX_LAG):
        super().__init__(out, key_id=key_id, key_path=key_path, push_logs=push_logs)
        self.contracts = contracts
        self.max_lag = max_lag

    def start_session(self, now: int):
        m = self.k.open_market(now)
        if not m:
            return None
        o = m["open_ts"]
        b = self.series.array(o - WARMUP, o)
        if np.isnan(b).any():
            log.warning("no price history yet for %s", m["ticker"])
            return None
        lv = all_levels(b, o)
        s = FPSession(m["ticker"], m["strike"], o, m["close_ts"], float(b[-1]), lv,
                      live=self.ready_ts is not None and self.ready_ts <= o + 1)
        log.info("session %s strike %.2f open %.2f, %d levels%s", s.ticker, s.strike, s.s0, len(lv),
                 "" if s.live else " (started mid-session, watch only)")
        self._append("sessions.csv", {"ticker": s.ticker, "open_utc": _iso(o), "strike": s.strike,
                                      "price_at_open": round(s.s0, 2), "live": int(s.live),
                                      "levels": json.dumps({k: [sd, round(p, 2)] for k, (sd, p) in lv.items()})})
        return s

    def minute_closes(self, t: int) -> np.ndarray:
        """Closes (last second) of the 101 complete minutes before second t's minute."""
        last = (t // 60) * 60 - 1
        return np.array([self.series.value(last - 60 * k) or np.nan for k in range(100, -1, -1)])

    def step(self, now: int, price: float):
        s = self.session
        if s is None or not s.live or now >= s.close_ts:
            return
        for name, (side, lev) in s.levels.items():
            if name in s.broken:
                continue
            gone = (price - lev) if side == "yes" else (lev - price)
            if gone >= BREAK:
                s.broken.add(name)
                self.judge(s, now, name, side, lev, price)

    def judge(self, s: FPSession, t: int, name: str, side: str, lev: float, price: float):
        left = s.close_ts - (t + 1)  # entry about 1 s after the break
        row = {"break_utc": _iso(t), "ticker": s.ticker, "side": side.upper(), "level": name,
               "level_price": round(lev, 2), "secs_left": left, "brti": round(price, 2),
               "strike": s.strike}
        reason = ""
        before = self.series.value(t - PUSH_SECS)
        vol = minute_vol(self.minute_closes(t))
        push = push_reading(price, before, vol, side) if before and vol else None
        row["push"] = round(push, 2) if push is not None else ""
        lag = time.time() - t
        row["lag_s"] = round(lag, 1)
        # settlement-aware chance of the side (Kalshi settles on the 60-s BRTI average; added 2026-11-10)
        p_side = side_prob(settle_prob(price, s.strike, left, sigma_1s(self.series.array(t - 900, t + 1))), side)
        row["model"] = round(p_side, 4)
        if not WINDOW[0] <= left < WINDOW[1]:
            reason = "outside 1-3 min window"
        elif side in s.done:
            reason = f"{side.upper()} already traded this session"
        elif push is None:
            reason = "no volatility reading"
        elif push < PUSH_MULT:
            reason = f"push {push:.2f} < {PUSH_MULT}"
        else:
            s.done.add(side)  # the first fast-push break per side is the one judged (as in the test)
            if self.max_lag is not None and lag > self.max_lag:
                reason = f"late by {lag:.0f}s"
        if reason and not WINDOW[0] <= left < WINDOW[1]:
            self._append("breaks.csv", {**row, "decision": "skip: " + reason})
            return
        book = None
        try:
            book = self.k.orderbook(s.ticker)
        except Exception as e:
            log.warning("orderbook failed: %s", e)
        bids = (book or {}).get("yes_bids" if side == "no" else "no_bids") or []
        ask = round(1 - bids[0][0], 4) if bids else None
        row.update(ask=ask if ask is not None else "", ask_size=bids[0][1] if bids else "",
                   book=json.dumps(bids[:8]) if bids else "", book_utc=_iso(time.time()))
        ok, limit = price_ok(ask)
        if not reason:
            if ask is None:
                reason = "empty book"
            elif not ok:
                reason = f"{side.upper()} ask {ask:.2f} not under 50c or 75-96c"
            elif not worth_it(p_side, ask):
                reason = f"settlement chance {p_side:.2f} under ask {ask:.2f} + fee"
        row["decision"] = f"BUY {side.upper()}" if not reason else "skip: " + reason
        self._append("signals.csv", row)
        log.info("%s %s %s push %s ask %s -> %s", s.ticker, side.upper(), name, row["push"], ask, row["decision"])
        if reason:
            return
        n, avg = fill(bids, self.contracts, limit)
        if not n:
            return
        fee = kalshi_fee(n, avg)
        self.unsettled.append({"time_utc": _iso(t + 1), "ticker": s.ticker, "side": side.upper(), "level": name,
                               "secs_left": left, "push": row["push"], "lag_s": row["lag_s"], "contracts": n,
                               "avg_price": round(avg, 4), "fee": fee, "cost": round(n * avg + fee, 2),
                               "model": round(p_side, 4), "close_ts": s.close_ts, "result": "", "won": "", "pnl": ""})
        log.info("PAPER BUY %d %s %s @ %.2f (fee %.2f)", n, side.upper(), s.ticker, avg, fee)


def from_env(out: str, push_logs: bool, contracts: int = CONTRACTS) -> FastPushPaper:
    return FastPushPaper(out, contracts, os.environ.get("KALSHI_KEY_ID"), os.environ.get("KALSHI_KEY_FILE"),
                         push_logs)
