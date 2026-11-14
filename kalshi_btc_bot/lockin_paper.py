"""Last-minute "lock-in" PAPER trader for Kalshi KXBTC15M (read-only, no orders are sent).

Michael asked for it on 2026-11-13 ("make the bot and add it to webpage") after the settlement-average test of
2026-11-10 (strategy/settlement_average_test.xlsx, sheets under60_rule and under60_by_sec_conf).

Why there is anything to trade: Kalshi settles each 15-minute contract on the AVERAGE of the 60 one-second BRTI
prints before the close, not on the last print. Inside the final minute part of that average is already known
("locked in"): with 30 s left, 31 of the 60 prints are history and only 29 can still move. The settlement-aware
probability (kalshi_btc_bot/settlement.py: locked part + the shrinking variance of the prints still to come) was
better than Kalshi's own price at 10-50 s left on Dec 2025 - Oct 2026 (Brier score). Buying the side it favours
when it beats the ask by a margin was the one new lead of that test.

The rule, exactly as tested (margin 5c, quote age under 10 s, first qualifying second, one trade per contract):

* Checks at 50, 40, 30, 20 and 10 seconds before the close of the open 15-minute market.
* At each check: spot = the BRTI print for that second; locked average = mean of the prints from T-60 up to and
  including this second (61 - seconds left prints); sigma = standard deviation of the 1-second BRTI log returns
  over the trailing 15 minutes; p = settle_prob(spot, strike, seconds left, sigma, locked average, prints known).
* Only the side the settlement math favours (chance 50% or more) can be bought: the locked-in prints have
  picked a side, and the trade is that Kalshi has not yet priced it fully. Edge = that side's chance minus the ask
  minus the taker fee 0.07 x ask x (1-ask); buy when the edge is at least 5 points. The first check that
  qualifies is the contract's one trade; later checks are not made. (The cheap side is never bought: when the
  model is less sure than Kalshi, Kalshi is usually right, as every test at 60 s or more has shown.)
* 20 contracts at the ask, walking the book only while the edge stays at least 5 points, held to settlement.

Backtest (BRTI 1-s, Kalshi trade prints as the price, last print within 10 s, 29.5k contracts):

  Dec-Aug: 498 trades, 80.5% won vs 62.8c paid, +16.5c per contract after fees, 9 of 9 months positive
  Sep-Oct: 160 trades, 76.2% won vs 67.8c paid,  +7.2c per contract after fees, 2 of 2 months positive

About 2-3 trades a day. With a 3c worse price it is still +13.6c / +4.3c. The margin and the seconds grid were
chosen after about 30 looks at this data and the backtest priced at trade prints, not at a live ask, so the paper
test is the real check: the decision rule is in docs/paper_test_decision_rule.md (retire under +1c at 300 trades).

Guards that are not from the backtest, each logged as a skip so their cost can be read later:
* the check must be acted on within MAX_LAG seconds (the live BRTI feed is about 1 s behind; a stale feed or a slow
  order book makes a 10-second-left decision worthless);
* the price for the check second must be a real BRTI print, not the exchange proxy (its $4-5 noise is bigger than
  the moves that decide the last 30 seconds);
* the bought side's bid-ask spread must be 2c or less (the spread skip of PR #35 for the range and favorite bots);
  a spread skip does not use up the contract's look, the next check can still trade. Rows go to skips.csv.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .late_breakdown import kalshi_fee, log
from .range_paper import CONTRACTS, MAX_SPREAD_LOW, _iso, fill
from .settlement import SETTLE_SECONDS, settle_prob, sigma_1s, taker_fee
from .squeeze_paper import SqueezePaper, _read_json

CHECK_SECS = (50, 40, 30, 20, 10)   # seconds left at which the rule is checked (the backtest's grid)
MIN_EDGE = 0.05                     # settlement-aware chance - ask - fee, in dollars
VOL_WINDOW = 900                    # seconds of BRTI behind the volatility estimate
MAX_LAG = 4                         # seconds between the check second and acting on it
MAX_SPREAD = MAX_SPREAD_LOW         # 2c: the spread skip shared with the range and favorite bots


# ----------------------------------------------------------------------------- the rule
def inputs(sec: np.ndarray, t: int, close_ts: int, strike: float) -> dict:
    """Everything the rule needs at second t. `sec` holds 1-second BRTI for [t-VOL_WINDOW, t] inclusive
    (VOL_WINDOW + 1 values, gaps filled)."""
    sec = np.asarray(sec, float)
    spot = float(sec[-1])
    secs_left = int(close_ts - t)
    known_n = max(SETTLE_SECONDS + 1 - secs_left, 0)        # prints T-60 .. t are already in the average
    known = sec[len(sec) - known_n:] if known_n else sec[:0]
    known_avg = float(known.mean()) if known_n else None
    sig = sigma_1s(sec)
    p_yes = settle_prob(spot, strike, secs_left, sig, known_avg, known_n) if sig > 0 else None
    return {"brti": spot, "strike": float(strike), "secs_left": secs_left, "known_n": known_n,
            "known_avg": known_avg, "sigma": sig, "p_yes": p_yes,
            "settle_mean": (known_n * known_avg + (SETTLE_SECONDS - known_n) * spot) / SETTLE_SECONDS if known_n else spot}


def edge(p_side: float, price: float) -> float:
    return p_side - price - taker_fee(price)


def book_sides(book: dict) -> dict:
    """Ask ladder and best bid for each side from the two bid ladders (YES ask = 1 - NO bid)."""
    out = {}
    for side, own, other in (("yes", "yes_bids", "no_bids"), ("no", "no_bids", "yes_bids")):
        asks = [[round(1 - p, 4), q] for p, q in (book.get(other) or [])]
        bids = book.get(own) or []
        out[side] = {"asks": asks, "ask": asks[0][0] if asks else None, "bid": bids[0][0] if bids else None}
    return out


def evaluate(f: dict, sides: dict, min_edge: float = MIN_EDGE) -> dict:
    """Decide one check. Returns the side to buy (or None) and every number behind it. Only the side the
    settlement-aware chance favours (50% or more) is eligible; both sides' numbers are logged."""
    out = {"look": None}
    if f.get("p_yes") is None:
        return out
    best = None
    for side in ("yes", "no"):
        p = f["p_yes"] if side == "yes" else 1.0 - f["p_yes"]
        ask = sides[side]["ask"]
        out[f"{side}_ask"], out[f"{side}_bid"], out[f"{side}_prob"] = ask, sides[side]["bid"], p
        e = edge(p, ask) if ask is not None and 0 < ask < 1 else None
        out[f"{side}_edge"] = e
        if p >= 0.5 and e is not None and e >= min_edge:
            best = side
    out["look"] = best
    out["prob"] = out[f"{best}_prob"] if best else None
    out["ask"] = out[f"{best}_ask"] if best else None
    return out


def limit_price(p_side: float, levels: list[list[float]], min_edge: float = MIN_EDGE) -> float:
    """Highest book price that still leaves an edge of at least min_edge."""
    lim = 0.0
    for price, _ in levels:
        if edge(p_side, price) >= min_edge:
            lim = max(lim, price)
    return lim


def spread_ok(ask: float | None, bid: float | None, max_spread: float = MAX_SPREAD) -> bool:
    """The bought side's bid-ask spread must be known and at most max_spread (on the 1c grid)."""
    if ask is None or bid is None:
        return False
    return round((ask - bid) * 100) <= round(max_spread * 100)


# ----------------------------------------------------------------------------- the bot
class LockinPaper(SqueezePaper):
    BOT = "Lock-in"
    CHECKS = CHECK_SECS
    EDGE = MIN_EDGE

    def __init__(self, out, key_id=None, key_path=None, push_logs: bool = False, contracts: int = CONTRACTS,
                 max_lag: float = MAX_LAG):
        self.looked: set[str] = set()          # tickers whose one trade has been taken
        super().__init__(out, key_id=key_id, key_path=key_path, push_logs=push_logs, contracts=contracts,
                         max_lag=max_lag)

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
        if m["close_ts"] - t not in self.CHECKS:
            return
        lag = self.wall() - t
        if lag > self.max_lag:
            self._row(t, m, None, None, f"skip: check is {lag}s late (limit {self.max_lag}s)")
            return
        if t not in self.series.brti:
            self._row(t, m, None, None, "skip: no real BRTI print for this second (exchange proxy only)")
            return
        self.check(t, m)

    def check(self, t: int, m: dict):
        sec = self.series.array(t - VOL_WINDOW, t + 1)
        if np.isnan(sec).any():
            self._row(t, m, None, None, "skip: BRTI history missing")
            return
        f = inputs(sec, t, m["close_ts"], m["strike"])
        if f["p_yes"] is None:
            self._row(t, m, f, None, "skip: BRTI has not moved, no volatility estimate")
            return
        try:
            book = self.k.orderbook(m["ticker"])
        except Exception as e:
            log.warning("orderbook failed: %s", e)
            self._row(t, m, f, None, "skip: order book unavailable")
            return
        lag = self.wall() - t
        if lag > self.max_lag:
            self._row(t, m, f, None, f"skip: order book arrived {lag}s after the check second")
            return
        sides = book_sides(book)
        ev = evaluate(f, sides, self.EDGE)
        side = ev["look"]
        if side is None:
            self._row(t, m, f, ev, f"no trade (favoured side has no {100 * self.EDGE:g}-point edge)")
            return
        if not spread_ok(ev["ask"], sides[side]["bid"]):
            sp = "unknown" if sides[side]["bid"] is None else f"{round((ev['ask'] - sides[side]['bid']) * 100)}c"
            n, avg = fill(book.get("no_bids" if side == "yes" else "yes_bids") or [], self.contracts,
                          limit_price(ev["prob"], sides[side]["asks"], self.EDGE))
            self._row(t, m, f, ev, f"skip: {side.upper()} spread {sp} is over {round(100 * MAX_SPREAD)}c")
            self._append("skips.csv", {"time_utc": _iso(t), "ticker": m["ticker"], "side": side.upper(),
                                       "secs_left": f["secs_left"], "ask": ev["ask"], "bid": sides[side]["bid"],
                                       "spread_c": sp, "model_prob": round(ev["prob"], 4),
                                       "edge_pts": round(100 * edge(ev["prob"], ev["ask"]), 2),
                                       "would_fill": n, "would_avg_price": round(avg, 4) if avg else ""})
            return
        self.looked.add(m["ticker"])
        self._save_state()
        p = ev["prob"]
        other = book.get("no_bids" if side == "yes" else "yes_bids") or []
        n, avg = fill(other, self.contracts, limit_price(p, sides[side]["asks"], self.EDGE))
        if not n:
            self._row(t, m, f, ev, f"skip: nothing to buy with a {100 * self.EDGE:g}-point edge")
            return
        fee = kalshi_fee(n, avg)
        self.traded.add(m["close_ts"])
        self._save_state()
        self._row(t, m, f, ev, f"BUY {side.upper()}", contracts=n, avg_price=round(avg, 4))
        self.unsettled.append({"time_utc": _iso(t), "ticker": m["ticker"], "side": side.upper(),
                               "level": f"lock-in at {f['secs_left']} s",
                               "secs_left": f["secs_left"], "model_prob": round(p, 4),
                               "gap_pts": round(100 * edge(p, avg), 2), "brti": round(f["brti"], 2),
                               "locked_avg": round(f["known_avg"], 2), "strike": round(f["strike"], 2),
                               "contracts": n, "avg_price": round(avg, 4),
                               "fee": fee, "cost": round(n * avg + fee, 2), "close_ts": m["close_ts"],
                               "result": "", "won": "", "pnl": ""})
        self._save_open()
        log.info("PAPER BUY %d %s %s @ %.2f with %ds left (chance %.3f, fee %.2f)", n, side.upper(), m["ticker"],
                 avg, f["secs_left"], p, fee)

    def _row(self, t, m, f, ev, decision, **extra):
        ev = ev or {}
        r = lambda v, d=4: "" if v is None else round(float(v), d)
        pts = lambda v: "" if v is None else round(100 * float(v), 2)
        row = {"time_utc": _iso(t), "ticker": m["ticker"], "secs_left": m["close_ts"] - t,
               "brti": r(f["brti"] if f else None, 2), "strike": round(m["strike"], 2),
               "locked_avg": r(f["known_avg"] if f else None, 2), "prints_known": f["known_n"] if f else "",
               "settle_mean": r(f["settle_mean"] if f else None, 2), "sigma_1s": r(f["sigma"] if f else None, 8),
               "p_yes_pct": pts(f["p_yes"] if f else None),
               "yes_bid": r(ev.get("yes_bid")), "yes_ask": r(ev.get("yes_ask")),
               "no_bid": r(ev.get("no_bid")), "no_ask": r(ev.get("no_ask")),
               "yes_edge_pts": pts(ev.get("yes_edge")), "no_edge_pts": pts(ev.get("no_edge")),
               "side": (ev.get("look") or "").upper(), "ask": r(ev.get("ask")),
               "level": f"lock-in at {m['close_ts'] - t} s", "lag_s": self.wall() - t,
               "decision": decision, "contracts": "", "avg_price": ""}
        row.update(extra)
        self._append("signals.csv", row)
        log.info("%s %ds left %s -> %s", m["ticker"], m["close_ts"] - t, row["side"] or "-", decision)


def from_env(out: str, push_logs: bool, contracts: int = CONTRACTS) -> LockinPaper:
    return LockinPaper(out, os.environ.get("KALSHI_KEY_ID"), os.environ.get("KALSHI_KEY_FILE"), push_logs, contracts)
