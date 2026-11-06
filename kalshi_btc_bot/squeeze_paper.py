"""Squeeze-break PAPER traders for Kalshi KXBTC15M (read-only, no orders are sent).

Michael asked for them on 2026-11-05 ("lets go with breaks bollinger band when squeeze is on, and that is the
direction we are going with", then "build it both ways"): one bot that takes the ask and one that rests a limit bid.

The signal (tested on Dec 2025 - Oct 2026 with Kalshi's actual trade prints):
* Squeeze on a 5-, 15- or 60-minute bar series of the 1-second BRTI: Bollinger bands (20 bars, 2 standard
  deviations) sit inside the Keltner channel (20-bar EMA, 1.5 x 20-bar average true range).
* Trigger: while the last completed bar of that series was a squeeze bar, the first 1-minute close that is outside
  that bar's Bollinger band. One trigger per squeeze episode and timeframe. Up = YES, down = NO.
* Market: the 15-minute BTC market that is open when the trigger fires. The breakout side is bought if its ask is
  60-89c. There is no minimum time left: the data showed late breaks scoring best, so none is imposed.
* 20 contracts, held to settlement, one trade per 15-minute market.

Taker bot (`kalshi-squeeze`): buys the breakout side at the ask, walking the book up to 89c.
Maker bot (`kalshi-squeeze-maker`): rests a virtual bid on the breakout side 1c under the ask (the best bid is joined
when the spread is only 1c) and keeps it until it fills or the market closes. A fill is counted only when the public
trades show it would have filled, and the queue ahead of us is counted (a trade exactly at our price first uses up the
queue). Maker fee 1.75% x C x P x (1-P), the same assumption as the other maker bot.

Backtest (real trade prints, Dec-Apr and Aug-Oct, every trigger counted, no one-per-market limit):
* Taker, trigger bought at the minute close, ask 60-89c: Dec-Aug 569 trades +4.3 points win % minus price,
  EV +3.1c per contract; Sep-Oct 174 trades +3.2 points, EV +2.0c. About +2 points over all prices.
* Maker, bid 1c under the ask, ask 60-89c: fills 97% / 100%, win % minus price +4.0 / +4.0, EV +3.7c per fill.
  That test did NOT count the queue ahead of us, so real fills will be fewer and slower; every signal that did not
  fill in the backtest won, so missing fills costs money. The paper test measures this.

Safety guard that is not from the backtest: a trigger is dropped if it cannot be acted on within MAX_LAG seconds
(the live BRTI feed stalled, or the next 15-minute market was still not listed). It is logged as a skip.
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .late_breakdown import LateBreakdownPaper, WARMUP, kalshi_fee, log
from .range_paper import CONTRACTS, _f, _iso, fill

KS = (5, 15, 60)                 # bar lengths in minutes
BB_N, BB_SD = 20, 2.0
KC_N, KC_MULT = 20, 1.5
ASK_MIN, ASK_MAX = 0.60, 0.89
MAX_LAG = 60                     # seconds a trigger may wait for the market/book before it is dropped
MAKER_FEE_RATE = 0.0175
POLL_SECONDS = 10


# ----------------------------------------------------------------------------- the signal
def squeeze_bars(sec: np.ndarray, k: int) -> dict | None:
    """Bollinger / Keltner state of the k-minute bars that end exactly at the end of `sec` (1-second prices,
    no gaps). Returns None when there are not enough bars."""
    size = 60 * k
    n = len(sec) // size
    if n < BB_N + 1:
        return None
    x = np.asarray(sec[len(sec) - n * size:], float).reshape(n, size)
    close, hi, lo = x[:, -1], x.max(axis=1), x.min(axis=1)
    s = pd.Series(close)
    sma = s.rolling(BB_N).mean().to_numpy()
    sd = s.rolling(BB_N).std(ddof=0).to_numpy()
    bbu, bbl = sma + BB_SD * sd, sma - BB_SD * sd
    pc = np.concatenate([[np.nan], close[:-1]])
    tr = np.nanmax(np.vstack([hi - lo, np.abs(hi - pc), np.abs(lo - pc)]), axis=0)
    atr = pd.Series(tr).rolling(KC_N).mean().to_numpy()
    ema = s.ewm(span=KC_N, adjust=False).mean().to_numpy()
    kcu, kcl = ema + KC_MULT * atr, ema - KC_MULT * atr
    with np.errstate(invalid="ignore"):
        on = (bbu < kcu) & (bbl > kcl)
    on = np.where(np.isnan(bbu) | np.isnan(kcu), False, on)
    return {"n": n, "on": on, "bbu": bbu, "bbl": bbl}


class SqueezeWatch:
    """Keeps the squeeze state of the last completed bar for each timeframe and fires the break trigger."""

    def __init__(self, ks=KS):
        self.ks = ks
        self.state: dict[int, dict] = {}
        self.fired: set[tuple[int, int]] = set()   # (k, first bar of the squeeze run) already triggered

    def refresh(self, k: int, boundary: int, sec: np.ndarray):
        """A k-minute bar has just completed at `boundary` (unix second); `sec` holds the seconds before it."""
        bars = squeeze_bars(sec, k)
        bar = boundary // (60 * k) - 1                # absolute index of the bar that just completed
        if bars is None or not bars["on"][-1]:
            self.state[k] = {"bar": bar, "on": False}
            return
        on, run = bars["on"], 1
        while run < bars["n"] and on[-1 - run]:
            run += 1
        self.state[k] = {"bar": bar, "on": True, "run": run, "j0": bar - run + 1,
                         "up": float(bars["bbu"][-1]), "dn": float(bars["bbl"][-1])}

    def check(self, t: int, close: float) -> list[dict]:
        """`t` is the last second of a minute and `close` its closing price. Returns the new triggers."""
        out = []
        for k in self.ks:
            st = self.state.get(k)
            if not st or not st["on"] or st["bar"] != t // (60 * k) - 1:
                continue                               # no squeeze, or the state is not for the bar before this one
            ep = (k, st["j0"])
            if ep in self.fired:
                continue
            if close > st["up"] or close < st["dn"]:
                self.fired.add(ep)
                out.append({"k": k, "side": "yes" if close > st["up"] else "no", "up": st["up"], "dn": st["dn"],
                            "run": st["run"], "close": close})
        if len(self.fired) > 400:
            self.fired = set(sorted(self.fired, key=lambda e: e[1])[-200:])
        return out


# ---------------------------------------------------------------------------- maker helpers
def maker_fee(contracts: int, price: float, rate: float = MAKER_FEE_RATE) -> float:
    return math.ceil(round(rate * contracts * price * (1 - price) * 100, 6)) / 100


def bid_price(own_bid: float | None, ask: float) -> float | None:
    """1c under the ask; join the best bid when the spread is only 1c. None when there is nothing sensible to post."""
    if own_bid is None:
        return round(ask - 0.01, 2) if ask >= 0.02 else None
    if round((ask - own_bid) * 100) >= 2:
        return round(ask - 0.01, 2)
    return round(own_bid, 2)


def size_at(levels: list[list[float]], price: float) -> float:
    return sum(q for p, q in levels if abs(p - price) < 0.0005)


def apply_trades(order: dict, trades: list[dict]) -> int:
    """Update a resting bid on order["side"] ("yes" or "no") with new public trades. Returns contracts newly filled.

    A trade fills us when its taker bought the OTHER side (so they hit bids on our side), after we posted, at our
    price or below it (measured on our side's price). A trade at our price first uses up the queue ahead of us."""
    side = order["side"]
    seen = set(order.setdefault("seen", []))
    got = 0
    for tr in sorted(trades, key=lambda x: x.get("created_time", "")):
        tid = tr.get("trade_id") or f'{tr.get("created_time")}|{tr.get("yes_price_dollars")}|{tr.get("count_fp")}'
        if tid in seen:
            continue
        seen.add(tid)
        if tr.get("created_time", "") <= order["posted_iso"] or tr.get("taker_side") == side:
            continue
        yp, q = _f(tr.get("yes_price_dollars")), _f(tr.get("count_fp")) or 0.0
        if yp is None or q <= 0:
            continue
        ours = yp if side == "yes" else round(1 - yp, 4)
        left = order["contracts"] - order["filled"]
        if left <= 0:
            break
        if ours < order["bid"] - 0.0005:                  # traded through us
            n = min(left, q)
        elif abs(ours - order["bid"]) < 0.0005:           # at our price: the queue ahead goes first
            use = min(order["queue_ahead"], q)
            order["queue_ahead"] -= use
            n = min(left, q - use)
        else:
            continue
        n = int(n)
        if n > 0:
            order["filled"] += n
            got += n
    order["seen"] = sorted(seen)[-500:]
    return got


# ----------------------------------------------------------------------------- the taker bot
class SqueezePaper(LateBreakdownPaper):
    BOT = "Squeeze"

    def __init__(self, out, key_id=None, key_path=None, push_logs: bool = False, contracts: int = CONTRACTS,
                 max_lag: float | None = MAX_LAG):
        super().__init__(out, key_id=key_id, key_path=key_path, push_logs=push_logs)
        self.contracts = contracts
        self.max_lag = max_lag
        self.watch = SqueezeWatch()
        self.market: dict | None = None
        self.pending: list[dict] = []
        self.traded: set[int] = set()          # close_ts of the 15-minute markets already traded
        self._inited = False
        self._mk_try = 0
        self._fresh = not (self.out / "state.json").exists()
        self._load_state()

    # ---------------------------------------------------------------- files
    def _load_state(self):
        j = _read_json(self.out / "state.json", {})
        self.watch.fired = {tuple(e) for e in j.get("fired", [])}
        self.traded = set(j.get("traded", []))
        self.unsettled = _read_json(self.out / "open_trades.json", [])

    def _save_state(self):
        (self.out / "state.json").write_text(json.dumps(
            {"fired": sorted(self.watch.fired), "traded": sorted(self.traded)[-50:]}))

    def _save_open(self):
        (self.out / "open_trades.json").write_text(json.dumps(self.unsettled, indent=1))

    # ---------------------------------------------------------------- the second loop
    def process(self, t: int, source: str):
        price = self.series.value(t)
        if price is None:
            return
        self._log_price(t, source)
        try:
            self.on_second(t, price)
        except Exception:
            log.exception("squeeze step failed")

    def on_second(self, t: int, price: float):
        if not self._inited:
            for k in KS:
                b = (t + 1) // (60 * k) * (60 * k)
                self.watch.refresh(k, b, self.series.array(b - WARMUP, b))
            self._inited = True
            if self._fresh:       # first ever start: a squeeze already running may have broken before we watched
                for k, st in self.watch.state.items():
                    if st["on"]:
                        self.watch.fired.add((k, st["j0"]))
                self._save_state()
            log.info("squeeze state ready: %s", {k: ("ON" if s["on"] else "off") for k, s in self.watch.state.items()})
        if t % 60 == 59:
            for sig in self.watch.check(t, price):
                log.info("squeeze break: %d-min %s, close %.2f outside %.2f/%.2f", sig["k"], sig["side"].upper(),
                         price, sig["dn"], sig["up"])
                self.pending.append({**sig, "t": t})
                self._save_state()
        arr = None
        for k in KS:
            if (t + 1) % (60 * k) == 0:
                if arr is None:
                    arr = self.series.array(t + 1 - WARMUP, t + 1)
                self.watch.refresh(k, t + 1, arr)
        if self.pending:
            self.run_pending()

    # ---------------------------------------------------------------- acting on a trigger
    def _market(self, now: int) -> dict | None:
        if self.market and self.market["open_ts"] <= now < self.market["close_ts"]:
            return self.market
        if now - self._mk_try < 2:
            return None
        self._mk_try = now
        try:
            self.market = self.k.open_market(now)
        except Exception as e:
            log.warning("market lookup failed: %s", e)
            self.market = None
        return self.market

    def wall(self) -> int:
        return int(time.time())

    def run_pending(self):
        now = self.wall()
        for p in list(self.pending):
            try:
                done = self.execute(p, now)
            except Exception:
                log.exception("squeeze trigger dropped")
                done = True
            if done:
                self.pending.remove(p)

    def execute(self, p: dict, now: int) -> bool:
        """Try to act on one trigger. Returns True when it is finished (acted on or dropped)."""
        lag = now - p["t"]
        m = self._market(now)
        if m is None:
            if self.max_lag is not None and lag > self.max_lag:
                self._signal(p, now, None, "", [], "skip: no open 15-minute market within %ds" % self.max_lag)
                return True
            return False
        side = p["side"]
        close = m["close_ts"]
        try:
            book = self.k.orderbook(m["ticker"])
        except Exception as e:
            log.warning("orderbook failed: %s", e)
            if self.max_lag is not None and lag > self.max_lag:
                self._signal(p, now, m, "", [], "skip: order book unavailable")
                return True
            return False
        other = book.get("no_bids" if side == "yes" else "yes_bids") or []
        own = book.get("yes_bids" if side == "yes" else "no_bids") or []
        ask = round(1 - other[0][0], 4) if other else None
        reason = ""
        if self.max_lag is not None and lag > self.max_lag:
            reason = f"late by {lag}s"
        elif close in self.traded:
            reason = "already traded this 15-minute market"
        elif ask is None:
            reason = "empty book"
        elif not ASK_MIN <= ask <= ASK_MAX:
            reason = f"{side.upper()} ask {ask:.2f} outside {ASK_MIN:.2f}-{ASK_MAX:.2f}"
        if reason:
            self._signal(p, now, m, ask, other, "skip: " + reason)
            return True
        self.act(p, now, m, ask, other, own)
        return True

    def _signal(self, p, now, m, ask, other, decision, **extra):
        row = {"time_utc": _iso(p["t"] + 1), "ticker": m["ticker"] if m else "", "side": p["side"].upper(),
               "level": f"{p['k']}-min squeeze", "level_price": round(p["up"] if p["side"] == "yes" else p["dn"], 2),
               "secs_left": (m["close_ts"] - now) if m else "", "brti": round(p["close"], 2), "squeeze_bars": p["run"],
               "lag_s": now - p["t"], "ask": ask, "ask_size": other[0][1] if other else "",
               "book": json.dumps(other[:6]) if other else "", "decision": decision}
        row.update(extra)
        self._append("signals.csv", row)
        log.info("%s %s %s -> %s", m["ticker"] if m else "-", p["side"].upper(), row["level"], decision)

    def act(self, p, now, m, ask, other, own):
        """Taker: buy the breakout side at the ask, walking the book up to ASK_MAX."""
        side = p["side"]
        n, avg = fill(other, self.contracts, ASK_MAX)
        if not n:
            self._signal(p, now, m, ask, other, "skip: nothing to buy under the limit")
            return
        self.traded.add(m["close_ts"])
        self._save_state()
        fee = kalshi_fee(n, avg)
        self._signal(p, now, m, ask, other, f"BUY {side.upper()}")
        self.unsettled.append({"time_utc": _iso(now), "ticker": m["ticker"], "side": side.upper(), "level": f"{p['k']}-min squeeze",
                               "secs_left": m["close_ts"] - now, "lag_s": now - p["t"], "contracts": n,
                               "avg_price": round(avg, 4), "fee": fee, "cost": round(n * avg + fee, 2),
                               "close_ts": m["close_ts"], "result": "", "won": "", "pnl": ""})
        self._save_open()
        log.info("PAPER BUY %d %s %s @ %.2f (fee %.2f)", n, side.upper(), m["ticker"], avg, fee)

    def settle(self, now: int):
        before = len(self.unsettled)
        super().settle(now)
        if len(self.unsettled) != before:
            self._save_open()


# ----------------------------------------------------------------------------- the maker bot
class SqueezeMakerPaper(SqueezePaper):
    BOT = "Squeeze maker"

    def __init__(self, out, key_id=None, key_path=None, push_logs: bool = False, contracts: int = CONTRACTS,
                 max_lag: float | None = MAX_LAG, maker_fee_rate: float = MAKER_FEE_RATE):
        self.fee_rate = maker_fee_rate
        self.orders: dict[str, dict] = {}       # ticker -> resting virtual bid
        self.missed: dict[str, dict] = {}       # ticker -> unfilled bid waiting for its result
        self._last_poll = 0
        super().__init__(out, key_id, key_path, push_logs, contracts, max_lag)
        self._load_orders()
        self.traded |= {o["close_ts"] for o in self.orders.values()}

    def _save_orders(self):
        (self.out / "resting_orders.json").write_text(json.dumps(
            {"orders": list(self.orders.values()), "missed": list(self.missed.values())}, indent=1))

    def _load_orders(self):
        j = _read_json(self.out / "resting_orders.json", {})
        self.orders = {o["ticker"]: o for o in j.get("orders", [])}
        self.missed = {o["ticker"]: o for o in j.get("missed", [])}

    def _event(self, t, o, what, note=""):
        self._append("orders.csv", {"time_utc": _iso(t), "event": what, "ticker": o["ticker"], "side": o["side"].upper(),
                                    "bid": o["bid"], "queue_ahead": round(o["queue_ahead"], 2), "filled": o["filled"],
                                    "contracts": o["contracts"], "ask_at_post": o["ask"], "taker_avg": o["taker_avg"],
                                    "note": note})

    def act(self, p, now, m, ask, other, own):
        """Maker: rest a bid on the breakout side, 1c under the ask (join the best bid at a 1c spread)."""
        side, tk = p["side"], m["ticker"]
        own_bid = own[0][0] if own else None
        bid = bid_price(own_bid, ask)
        if bid is None:
            self._signal(p, now, m, ask, other, "skip: no sensible bid price")
            return
        improved = own_bid is None or bid > own_bid + 0.0005
        queue = 0.0 if improved else size_at(own, bid)
        tn, tavg = fill(other, self.contracts, ASK_MAX)
        o = {"ticker": tk, "side": side, "close_ts": m["close_ts"],
             "posted_iso": datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"), "posted_ts": now,
             "bid": bid, "queue_ahead": queue, "contracts": self.contracts, "filled": 0, "ask": ask,
             "taker_avg": round(tavg, 4) if tavg else "", "taker_n": tn, "level": f"{p['k']}-min squeeze",
             "signal_t": p["t"], "first_fill_logged": False}
        self.orders[tk] = o
        self.traded.add(m["close_ts"])
        self._save_state()
        self._save_orders()
        self._signal(p, now, m, ask, other, f"BID {side.upper()} @ {bid:.2f}", bid=bid, queue_ahead=queue,
                     improved=int(improved))
        self._event(now, o, "post")
        log.info("PAPER BID %d %s %s @ %.2f (ask %.2f, queue %.0f)", self.contracts, side.upper(), tk, bid, ask, queue)

    def _signal(self, p, now, m, ask, other, decision, **extra):
        extra.setdefault("bid", "")
        extra.setdefault("queue_ahead", "")
        extra.setdefault("improved", "")
        super()._signal(p, now, m, ask, other, decision, **extra)

    # ---------------------------------------------------------------- fills and settlement
    def _finish(self, now, tk, why):
        o = self.orders.pop(tk)
        self._event(now, o, "done", why)
        if o["filled"] > 0:
            n, pr = o["filled"], o["bid"]
            fee = maker_fee(n, pr, self.fee_rate)
            self.unsettled.append({"time_utc": _iso(o["posted_ts"]), "ticker": tk, "side": o["side"].upper(), "level": o["level"],
                                   "secs_left": o["close_ts"] - o["posted_ts"], "lag_s": o["posted_ts"] - o["signal_t"],
                                   "contracts": n, "avg_price": pr, "fee": fee, "cost": round(n * pr + fee, 2),
                                   "ask_at_post": o["ask"], "taker_avg": o["taker_avg"],
                                   "taker_fee_saved": round(kalshi_fee(n, pr) - fee, 2), "close_ts": o["close_ts"],
                                   "result": "", "won": "", "pnl": ""})
            self._save_open()
            log.info("PAPER FILLED %d/%d %s %s @ %.2f", n, o["contracts"], o["side"].upper(), tk, pr)
        else:
            self.missed[tk] = {"ticker": tk, "side": o["side"].upper(), "close_ts": o["close_ts"],
                               "time_utc": _iso(o["posted_ts"]), "bid": o["bid"], "ask_at_post": o["ask"],
                               "taker_avg": o["taker_avg"], "taker_n": o["taker_n"]}
            log.info("PAPER BID %s not filled (%s)", tk, why)
        self._save_orders()

    def poll_orders(self, now: int):
        for tk, o in list(self.orders.items()):
            if now >= o["close_ts"]:
                self._finish(now, tk, "market closed")
        if not self.orders or now - self._last_poll < POLL_SECONDS:
            return
        self._last_poll = now
        for tk, o in list(self.orders.items()):
            try:
                trades, cursor = [], None
                for _ in range(5):
                    params = {"ticker": tk, "min_ts": o["posted_ts"], "limit": 1000}
                    if cursor:
                        params["cursor"] = cursor
                    j = self.k.get("/markets/trades", **params)
                    trades += j.get("trades") or []
                    cursor = j.get("cursor")
                    if not cursor:
                        break
            except Exception as e:
                log.warning("trades %s failed: %s", tk, e)
                continue
            if apply_trades(o, trades):
                self._event(now, o, "fill")
                if not o["first_fill_logged"]:      # the chart marks the first fill
                    o["first_fill_logged"] = True
                    self._append("signals.csv", {
                        "time_utc": _iso(now), "ticker": tk, "side": o["side"].upper(), "level": o["level"],
                        "level_price": "", "secs_left": o["close_ts"] - now, "brti": "", "squeeze_bars": "",
                        "lag_s": "", "ask": o["ask"], "ask_size": "", "book": "",
                        "decision": f"BUY {o['side'].upper()} (limit fill)", "bid": o["bid"], "queue_ahead": "", "improved": ""})
                self._save_orders()
                if o["filled"] >= o["contracts"]:
                    self._finish(now, tk, "fully filled")

    def settle(self, now: int):
        self.poll_orders(now)
        if now - self._last_settle >= 60:
            for tk, o in list(self.missed.items()):
                if now < o["close_ts"] + 60:
                    continue
                try:
                    res = self.k.result(tk)
                except Exception as e:
                    log.warning("result %s failed: %s", tk, e)
                    continue
                if res is None:
                    continue
                row = {k: v for k, v in o.items() if k != "close_ts"}
                row.update(result=res, won=int(res == o["side"].lower()))
                self._append("missed.csv", row)
                del self.missed[tk]
                self._save_orders()
        super().settle(now)


def _read_json(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def from_env(out: str, push_logs: bool, contracts: int = CONTRACTS) -> SqueezePaper:
    return SqueezePaper(out, os.environ.get("KALSHI_KEY_ID"), os.environ.get("KALSHI_KEY_FILE"), push_logs, contracts)


def maker_from_env(out: str, push_logs: bool, contracts: int = CONTRACTS,
                   maker_fee_rate: float = MAKER_FEE_RATE) -> SqueezeMakerPaper:
    return SqueezeMakerPaper(out, os.environ.get("KALSHI_KEY_ID"), os.environ.get("KALSHI_KEY_FILE"), push_logs,
                             contracts, maker_fee_rate=maker_fee_rate)
