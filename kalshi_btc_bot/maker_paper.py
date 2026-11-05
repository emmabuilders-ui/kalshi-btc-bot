"""Limit-order (maker) PAPER trader on the hourly Bitcoin range bands (KXBTC). Read-only: no orders are sent.

Michael asked for it on 2026-11-05 ("yes", to building a limit-order bot). Same signal as the tight trend bot
(trend_paper.py, rule "tight": YES ask 70-84c, BRTI within 40% of the band centre, BRTI and its 15-second
projection inside the band, spread at most 5c, 20 down to 4 minutes before the hourly close), but instead of
taking the ask it rests a virtual YES bid and only counts a fill when real trades show it would have filled.

The resting bid
* spread 2c or wider: bid 1c above the best YES bid (we are first in line, nobody ahead of us);
* spread 1c: join the best YES bid; the size already showing at that price is ahead of us in the queue.

How a fill is counted (Kalshi public trades for that market, polled every 10 seconds)
* only trades after the bid was posted where the taker sold YES (taker_side "no": they hit the YES bids);
* a trade below our bid went through us, so it fills us (up to what is left);
* a trade at our bid first uses up the queue ahead of us, then fills us;
* partial fills count; whatever is filled at 3 minutes left is held to settlement.

The bid is cancelled at 3 minutes left, or at a later minute check when BRTI or its projection has left the band
or BRTI is more than 40% from the centre. One bid per hour, 20 contracts, held to settlement.

Fees: maker fee = MAKER_FEE_RATE x C x P x (1-P), rounded up to the cent (default 1.75%, a quarter of the 7%
taker fee; Kalshi charges no maker fee on many markets, so this is on the safe side). Change with --maker-fee.

For each bid the logs also keep what the same signal would have cost as a taker (the ask and the book walk), and
unfilled bids are settled into missed.csv, so we can see both the savings and whether the fills we do get are
the losers (adverse selection: the bid fills because the price is falling).
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone

from .late_breakdown import kalshi_fee, log
from .range_paper import CONTRACTS, _f, _iso, fill
from .trend_paper import LAST_MINUTE, TrendPaper, centre_distance, inside, projection, slope_of, RULES

MAKER_FEE_RATE = 0.0175
POLL_SECONDS = 10
CANCEL_AT = LAST_MINUTE - 1          # 3 minutes left
BOT_NAME = "Maker"


def maker_fee(contracts: int, price: float, rate: float = MAKER_FEE_RATE) -> float:
    return math.ceil(round(rate * contracts * price * (1 - price) * 100, 6)) / 100


def bid_price(yes_bid: float, yes_ask: float) -> tuple[float, bool]:
    """(our bid, improves the best bid?)"""
    if round((yes_ask - yes_bid) * 100) >= 2:
        return round(yes_bid + 0.01, 2), True
    return round(yes_bid, 2), False


def size_at(levels: list[list[float]], price: float) -> float:
    return sum(q for p, q in levels if abs(p - price) < 0.0005)


def apply_trades(order: dict, trades: list[dict]) -> int:
    """Update a resting order with new public trades. Returns contracts newly filled."""
    seen = set(order.setdefault("seen", []))
    got = 0
    for tr in sorted(trades, key=lambda x: x.get("created_time", "")):
        tid = tr.get("trade_id") or f'{tr.get("created_time")}|{tr.get("yes_price_dollars")}|{tr.get("count_fp")}'
        if tid in seen:
            continue
        seen.add(tid)
        if tr.get("created_time", "") <= order["posted_iso"] or tr.get("taker_side") != "no":
            continue
        yp, q = _f(tr.get("yes_price_dollars")), _f(tr.get("count_fp")) or 0.0
        if yp is None or q <= 0:
            continue
        left = order["contracts"] - order["filled"]
        if left <= 0:
            break
        if yp < order["bid"] - 0.0005:                 # traded through us
            n = min(left, q)
        elif abs(yp - order["bid"]) < 0.0005:          # at our price: the queue ahead goes first
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


class MakerPaper(TrendPaper):
    def __init__(self, out, key_id, key_path, push_logs=False, contracts=CONTRACTS, maker_fee_rate=MAKER_FEE_RATE):
        super().__init__(out, key_id, key_path, push_logs, contracts, rule="tight")
        self.fee_rate = maker_fee_rate
        self.orders: dict[str, dict] = {}       # ticker -> resting virtual bid
        self.missed: dict[str, dict] = {}       # ticker -> unfilled bid waiting for its result
        self._last_poll = 0
        self._load_orders()
        self.hours_traded |= {o["close_ts"] for o in self.orders.values()}

    # ---------------------------------------------------------------- files
    def _save_orders(self):
        (self.out / "resting_orders.json").write_text(json.dumps(
            {"orders": list(self.orders.values()), "missed": list(self.missed.values())}, indent=1))

    def _load_orders(self):
        p = self.out / "resting_orders.json"
        if p.exists():
            j = json.loads(p.read_text())
            self.orders = {o["ticker"]: o for o in j.get("orders", [])}
            self.missed = {o["ticker"]: o for o in j.get("missed", [])}

    def _event(self, t, o, what, note=""):
        self._append("orders.csv", {"time_utc": _iso(t), "event": what, "ticker": o["ticker"], "bid": o["bid"],
                                    "improved": int(o["improved"]), "queue_ahead": round(o["queue_ahead"], 2),
                                    "filled": o["filled"], "contracts": o["contracts"], "mins_left": o["mins_left"],
                                    "ask_at_post": o["ask"], "taker_avg": o["taker_avg"], "note": note})

    # ---------------------------------------------------------------- the rule
    def check(self, t: int):
        close = (t // 3600 + 1) * 3600
        left = (close - t) // 60
        # manage the resting bid for this hour first
        for tk, o in list(self.orders.items()):
            if o["close_ts"] < close:                  # left over from an earlier hour (e.g. after a restart)
                self._finish(t, tk, "cancel: hour already over")
                continue
            if o["close_ts"] != close:
                continue
            if left <= CANCEL_AT:
                self._finish(t, tk, "cancel at 3 min left")
                continue
            price, prints = self.price_at(t), self.prints(t)
            if price is None or prints is None:
                continue
            proj = projection(price, slope_of(prints), left)
            lo, hi = o["floor"], o["cap"]
            if not inside(price, lo, hi) or not inside(proj, lo, hi) or centre_distance(price, lo, hi) > RULES["tight"][0][2] + 1e-9:
                self._finish(t, tk, "cancel: trend-in conditions failed")
        super().check(t)

    def enter(self, t, close, left, m, ask, price, slope, proj, dist, limit):   # noqa: D102
        tk = m["ticker"]
        try:
            book = self.k.orderbook(tk)
        except Exception as e:
            log.warning("orderbook %s failed: %s", tk, e)
            book = {}
        yb = book.get("yes_bid") or _f(m.get("yes_bid_dollars"))
        ya = book.get("yes_ask") or ask
        if yb is None or ya is None:
            return
        bid, improved = bid_price(yb, ya)
        queue = 0.0 if improved else size_at(book.get("yes_bids") or [], bid)
        tn, tavg = fill(book.get("no_bids") or [], self.contracts, limit)
        o = {"ticker": tk, "close_ts": close, "posted_iso": datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
             "posted_ts": t, "mins_left": left, "floor": _f(m.get("floor_strike")), "cap": _f(m.get("cap_strike")),
             "bid": bid, "improved": improved, "queue_ahead": queue, "contracts": self.contracts, "filled": 0,
             "ask": ya, "yes_bid": yb, "taker_avg": round(tavg, 4) if tavg else "", "taker_n": tn,
             "brti": round(price, 2), "slope": round(slope, 4), "projection": round(proj, 2), "dist": round(dist * 100, 1)}
        self.orders[tk] = o
        self.hours_traded.add(close)
        self.hours_traded = {c for c in self.hours_traded if c >= close - 7200}
        self._event(t, o, "post")
        self._append("signals.csv", {"time_utc": _iso(t), "ticker": tk, "side": "YES", "mins_left": left,
                                     "floor": m.get("floor_strike"), "cap": m.get("cap_strike"), "brti": o["brti"],
                                     "slope_per_s": o["slope"], "projection": o["projection"], "centre_dist_pct": o["dist"],
                                     "ask_snapshot": ya, "bid": bid, "improved": int(improved), "queue_ahead": queue,
                                     "taker_avg": o["taker_avg"], "book": json.dumps((book.get("yes_bids") or [])[:6])})
        self._save_orders()
        log.info("PAPER BID %d YES %s @ %.2f (%s, queue %.0f, ask %.2f, %d min left)",
                 self.contracts, tk, bid, "improves" if improved else "joins", queue, ya, left)

    def _finish(self, t, tk, why):
        o = self.orders.pop(tk)
        self._event(t, o, "done", why)
        if o["filled"] > 0:
            n, p = o["filled"], o["bid"]
            fee = maker_fee(n, p, self.fee_rate)
            self.held[tk] = {"time_utc": _iso(o["posted_ts"]), "ticker": tk, "side": "YES", "mins_left": o["mins_left"],
                             "contracts": n, "avg_price": p, "fee": fee, "cost": round(n * p + fee, 2),
                             "ask_at_post": o["ask"], "taker_avg": o["taker_avg"],
                             "taker_fee_saved": round(kalshi_fee(n, p) - fee, 2), "close_ts": o["close_ts"]}
            self._save_open()
            log.info("PAPER FILLED %d/%d YES %s @ %.2f", n, o["contracts"], tk, p)
        else:
            self.missed[tk] = {"ticker": tk, "close_ts": o["close_ts"], "time_utc": _iso(o["posted_ts"]),
                               "bid": o["bid"], "ask_at_post": o["ask"], "taker_avg": o["taker_avg"],
                               "taker_n": o["taker_n"], "mins_left": o["mins_left"]}
            log.info("PAPER BID %s not filled (%s)", tk, why)
        self._save_orders()

    # ---------------------------------------------------------------- fills and settlement
    def poll_orders(self, now: int):
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
                self._save_orders()
                if o["filled"] >= o["contracts"]:
                    self._finish(now, tk, "fully filled")

    def settle(self, now: int):
        self.poll_orders(now)
        due = now - self._last_settle >= 60
        if due:
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
                row.update(result=res, won=int(res == "yes"))
                self._append("missed.csv", row)
                del self.missed[tk]
                self._save_orders()
        super().settle(now)


def from_env(out: str, push_logs: bool, contracts: int = CONTRACTS, maker_fee_rate: float = MAKER_FEE_RATE) -> MakerPaper:
    return MakerPaper(out, os.environ.get("KALSHI_KEY_ID"), os.environ.get("KALSHI_KEY_FILE"),
                      push_logs, contracts, maker_fee_rate)
