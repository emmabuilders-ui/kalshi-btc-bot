"""Longshot PAPER trader for Kalshi KXBTC15M (read-only, no orders are sent).

Michael asked for it on 2026-11-10 ("find me a longshot trade that pays off. Must be 15% or lower." then "build it").
The scan behind it is in research/longshot/; every other longshot setup tested there loses after the fee.

The rule, exactly as tested:

* Only sides priced 15c or less. One side at most can be that cheap, because the two asks always add up to more
  than a dollar, so there is never a choice to make between YES and NO.
* Fair value: how far BTC still has to travel to win, divided by how far it usually travels in the time left
  (the standard deviation of the 1-second BRTI log returns over the trailing 15 minutes, times the square root of
  the seconds left less the 40-second settlement window). The normal curve turns that into a probability.
* Buy when fair value minus the ask minus the taker fee (0.07 x p x (1-p)) is at least 8 points. The 8-point margin
  was picked on Dec 2025 - Aug 2026 and is the first margin whose Dec-Aug edge clears two and a half standard
  errors; the result rises smoothly from a 0-point to a 10-point margin, so it is not a knife edge.
* Checks at each minute mark 1-14 after the open (14 down to 1 minute left). The first minute the side qualifies is
  the contract's one look; later minutes are not checked again, as in the backtest's one trade per contract.
* 20 contracts at the ask (walking the book only while the margin stays at least 8 points), held to settlement.

The normal curve is known to overstate longshot odds here: where it says 17% the contracts win about 12.5%. The
empirical table in research/longshot/emp.py is the honest fair value and prices these same trades at 12.7% against
the 9.55c paid, so the 8-point margin really means "the ask is about 3 points under the true win rate". Trading
straight off that table is weaker (it adds cheap 1-6c looks that lose), which is why the normal curve is the trigger.

Backtest of this module's own decision code (research/longshot/replay_bot.py replays `inputs`, `evaluate` and
`limit_price` over the leak-free snapshots; one trade per contract, first qualifying minute):

  Dec-Aug: 1,497 trades, 12.49% vs 9.94c, +1.94c per contract (t 2.3, 7 of 9 months positive, July -2.8c)
  Sep-Oct:   184 trades, 12.50% vs 9.45c, +2.46c per contract (t 1.0, September -0.1c, October +4.1c)

The first scan reported +2.23c / +1.96c on 1,454 / 174 trades. The difference is sides priced at exactly 15c: in
floating point 1 - 0.85 is a shade above 0.15, so the scan dropped those rows and this module (which rounds book
prices to four decimals) keeps them. They are the worse trades, which is why the edge is a little lower here. The
8-point margin was chosen before that was found and is left as it is; the margin sweep is smooth either way
(4c: +0.3c, 6c: +0.8c, 8c: +1.9c, 10c: +3.2c on Dec-Aug) and only breaks down at 12c.

About 5-6 trades a day. Sep-Oct is thin (standard error 2.4c) and the setup was chosen after scanning about 2,400
cells on this data, so the paper test is the real check. Backtest prices came from Kalshi trade prints; the paper bot
uses the live order book, so fills can be worse.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone

import numpy as np

from .late_breakdown import kalshi_fee, log
from .range_paper import CONTRACTS, _iso, fill
from .squeeze_paper import SqueezePaper, _read_json

CHECK_KS = range(1, 15)          # minute marks after the open: 14 .. 1 minutes left
PX_MAX = 0.15                    # longshots only
MIN_EDGE = 0.08                  # fair value - ask - fee, in dollars
VOL_WINDOW = 900                 # seconds of BRTI behind the volatility estimate
SETTLE_WINDOW = 40               # the last 40 s are the settlement average, so they cannot move the result
MAX_LAG = 20                     # seconds between the check second and acting on it


# ----------------------------------------------------------------------------- the fair value
def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def volatility(sec: np.ndarray) -> float:
    """Standard deviation of the 1-second BRTI log returns over `sec` (gaps already filled)."""
    lr = np.diff(np.log(np.asarray(sec, float)))
    m = float(lr.mean())
    return math.sqrt(max(float((lr * lr).mean()) - m * m, 0.0))


def inputs(sec: np.ndarray, t: int, close_ts: int, strike: float) -> dict:
    """Everything the rule needs at second t. `sec` holds 1-second BRTI for [t-VOL_WINDOW, t] inclusive."""
    brti = float(sec[-1])
    sig = volatility(sec)
    secs_left = close_ts - t
    teff = max(secs_left - SETTLE_WINDOW, 1)
    scale = max(sig, 1e-9) * math.sqrt(teff)
    dist = math.log(brti / strike)                     # positive: BTC is above the strike, so YES is winning
    return {"brti": brti, "strike": float(strike), "sig15": sig, "secs_left": secs_left, "dist": dist,
            "scale": scale, "move_1sd": brti * scale,
            "hour_utc": float(datetime.fromtimestamp(t, timezone.utc).hour)}


def fair(side: str, f: dict) -> float:
    """Chance `side` ends up winning: the normal-curve chance BTC travels the distance that is still needed."""
    gap = -f["dist"] if side == "yes" else f["dist"]   # log distance still to travel, positive when out of the money
    return _norm_cdf(-gap / f["scale"])


def edge(p: float, price: float) -> float:
    return p - price - 0.07 * price * (1 - price)


def book_asks(book: dict) -> dict:
    """Ask ladders for YES and NO from the bid ladders (YES ask = 1 - NO bid)."""
    out = {}
    for side, other in (("yes", "no_bids"), ("no", "yes_bids")):
        out[side] = [[round(1 - p, 4), q] for p, q in (book.get(other) or [])]
    return out


def evaluate(f: dict, asks: dict, min_edge: float = MIN_EDGE) -> dict:
    """Decide one minute check. Returns the look (or None) and every number behind it."""
    out = {"look": None}
    for side in ("yes", "no"):
        lv = asks.get(side) or []
        ask = lv[0][0] if lv else None
        p = fair(side, f)
        out[f"{side}_ask"], out[f"{side}_fair"] = ask, p
        out[f"{side}_edge"] = edge(p, ask) if ask is not None else None
        if ask is not None and 0 < ask <= PX_MAX and out[f"{side}_edge"] >= min_edge:
            out["look"] = side
    side = out["look"]
    out["fair"] = out[f"{side}_fair"] if side else None
    out["zgap"] = (-f["dist"] if side == "yes" else f["dist"]) / f["scale"] if side else None
    return out


def limit_price(p: float, levels: list[list[float]], min_edge: float = MIN_EDGE) -> float:
    """Highest book price that is still a longshot and still leaves an edge of at least min_edge."""
    lim = 0.0
    for price, _ in levels:
        if price <= PX_MAX and edge(p, price) >= min_edge:
            lim = max(lim, price)
    return lim


# ----------------------------------------------------------------------------- the bot
class LongshotPaper(SqueezePaper):
    BOT = "Longshot"
    CHECKS = CHECK_KS
    EDGE = MIN_EDGE

    def __init__(self, out, key_id=None, key_path=None, push_logs: bool = False, contracts: int = CONTRACTS):
        self.looked: set[str] = set()          # tickers whose one look has been used
        super().__init__(out, key_id=key_id, key_path=key_path, push_logs=push_logs, contracts=contracts,
                         max_lag=MAX_LAG)

    def _load_state(self):
        j = _read_json(self.out / "state.json", {})
        self.looked = set(j.get("looked", []))
        self.traded = set(j.get("traded", []))
        self.unsettled = _read_json(self.out / "open_trades.json", [])

    def _save_state(self):
        (self.out / "state.json").write_text(json.dumps(
            {"looked": sorted(self.looked)[-200:], "traded": sorted(self.traded)[-50:]}))

    def on_second(self, t: int, price: float):
        m = self._market(self.wall())
        if not m or not m["open_ts"] <= t < m["close_ts"] or m["ticker"] in self.looked:
            return
        k, r = divmod(t - m["open_ts"], 60)
        if r or k not in self.CHECKS:
            return
        if self.wall() - t > MAX_LAG:
            self._row(t, m, None, None, f"skip: check is {self.wall() - t}s late")
            return
        self.check(t, m)

    def check(self, t: int, m: dict):
        sec = self.series.array(t - VOL_WINDOW, t + 1)
        if np.isnan(sec).any():
            self._row(t, m, None, None, "skip: BRTI history missing")
            return
        f = inputs(sec, t, m["close_ts"], m["strike"])
        if f["sig15"] <= 0:
            self._row(t, m, f, None, "skip: BRTI has not moved, no volatility estimate")
            return
        try:
            book = self.k.orderbook(m["ticker"])
        except Exception as e:
            log.warning("orderbook failed: %s", e)
            self._row(t, m, f, None, "skip: order book unavailable")
            return
        asks = book_asks(book)
        ev = evaluate(f, asks, self.EDGE)
        side = ev["look"]
        if side is None:
            self._row(t, m, f, ev, f"no look (no side at {100 * PX_MAX:g}c or less with a "
                                   f"{100 * self.EDGE:g}-point edge)")
            return
        self.looked.add(m["ticker"])
        self._save_state()
        p = ev["fair"]
        other = book.get("no_bids" if side == "yes" else "yes_bids") or []
        n, avg = fill(other, self.contracts, limit_price(p, asks[side], self.EDGE))
        if not n:
            self._row(t, m, f, ev, f"skip: nothing to buy with a {100 * self.EDGE:g}-point edge")
            return
        fee = kalshi_fee(n, avg)
        self.traded.add(m["close_ts"])
        self._save_state()
        self._row(t, m, f, ev, f"BUY {side.upper()}", contracts=n, avg_price=round(avg, 4))
        self.unsettled.append({"time_utc": _iso(t), "ticker": m["ticker"], "side": side.upper(),
                               "level": f"{ev['zgap']:.2f} vol units from the strike",
                               "secs_left": m["close_ts"] - t, "model_prob": round(p, 4),
                               "gap_pts": round(100 * edge(p, avg), 2), "contracts": n, "avg_price": round(avg, 4),
                               "fee": fee, "cost": round(n * avg + fee, 2), "close_ts": m["close_ts"],
                               "result": "", "won": "", "pnl": ""})
        self._save_open()
        log.info("PAPER BUY %d %s %s @ %.2f (fair %.3f, fee %.2f)", n, side.upper(), m["ticker"], avg, p, fee)

    def _row(self, t, m, f, ev, decision, **extra):
        ev = ev or {}
        r = lambda v, d=4: "" if v is None else round(float(v), d)
        pts = lambda v: "" if v is None else round(100 * float(v), 2)
        row = {"time_utc": _iso(t), "ticker": m["ticker"], "min_left": (m["close_ts"] - t) // 60,
               "secs_left": m["close_ts"] - t, "brti": r(f["brti"] if f else None, 2), "strike": round(m["strike"], 2),
               "distance_usd": r(abs(m["strike"] - f["brti"]) if f else None, 2),
               "one_sd_usd": r(f["move_1sd"] if f else None, 2), "sig15": r(f["sig15"] if f else None, 8),
               "yes_ask": r(ev.get("yes_ask")), "no_ask": r(ev.get("no_ask")),
               "yes_fair_pct": pts(ev.get("yes_fair")), "no_fair_pct": pts(ev.get("no_fair")),
               "yes_edge_pts": pts(ev.get("yes_edge")), "no_edge_pts": pts(ev.get("no_edge")),
               "side": (ev.get("look") or "").upper(), "zgap": r(ev.get("zgap"), 3),
               "decision": decision, "contracts": "", "avg_price": ""}
        row.update(extra)
        self._append("signals.csv", row)
        log.info("%s %s -> %s", m["ticker"], row["side"] or "-", decision)


def from_env(out: str, push_logs: bool, contracts: int = CONTRACTS) -> LongshotPaper:
    return LongshotPaper(out, os.environ.get("KALSHI_KEY_ID"), os.environ.get("KALSHI_KEY_FILE"), push_logs, contracts)
