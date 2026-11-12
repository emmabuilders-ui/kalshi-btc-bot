"""Hourly Bitcoin range (Kalshi KXBTC) PAPER trader. Read-only: no orders are ever sent.

Rule (Michael, 2026-11-01; backtest Jan-Oct 2026 with the look-ahead fix):

* Each hourly event has a ladder of range bands ($100 wide since mid-March). The bot watches
  the 14 bands nearest the BRTI price 30 minutes before the close.
* Every minute it estimates the chance BRTI settles inside each band (finish-in-range model:
  empirical standardized return quantiles, scaled by 100-minute 1-minute volatility, bucketed
  by 10-vs-100 minute volatility state). The NO chance is 1 minus that. Since 2026-11-10 the
  time left is shrunk by 40 s (was 30 s): Kalshi settles on the 60-second BRTI average, whose
  variance equals that of the spot price 40 s before the close. On Dec 2025-Oct 2026 BRTI the
  40-s version has the better Brier score at every minute and adds about 25 YES trades a month
  (96% won at 92c) that the 30-s version skipped.
* Buy a side when its ask is 75-89c and the model is 10+ points above the ask, or the ask is
  90-99c and the model is 2+ points above the ask.
* Checks start at 30 minutes left (the earliest the backtest could test); the bot takes the
  first qualifying minute on each band.
* One position per band (YES or NO, never both), held to settlement. 20 contracts.
* Added 2026-11-04 (Michael: "do it"): skip a 75-89c entry when the band's bid-ask spread is 3c or
  more. The loser study (strategy/loser_anatomy_summary.md) found the 3-5c-spread half of the trades
  made nothing in both Dec-Aug and Sep-Oct; 90c+ entries are not filtered (there it cost money).

Backtest (minute bid/ask snapshots, fees per 20-contract order): YES 378 trades, won 93.1% at
89.5c, +3.0c a contract; NO 4,212 trades, won 94.0% at 92.8c, +0.7c.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .late_breakdown import KalshiClient, kalshi_fee
from .logpush import git_push_logs

log = logging.getLogger("kalshi_btc_bot")

SERIES = "KXBTC"
N_BANDS = 14
YES_WINDOW, NO_WINDOW = 30, 30          # start looking at 30 minutes left (earliest tested)
LOW_FLOOR, HIGH_FLOOR = 0.745, 0.895    # 75c and 90c (prices are on a 1c grid)
LOW_GAP, HIGH_GAP = 0.10, 0.02
MAX_SPREAD_LOW = 0.02                   # 75-89c entries need a bid-ask spread of 2c or less
CONTRACTS = 20
ET = ZoneInfo("America/New_York")


# ------------------------------------------------------------------------------ the model
SETTLE_SHRINK_MIN = 40 / 60   # settlement = 60-s BRTI average: its variance is that of the spot 40 s before the close


class RangeModel:
    def __init__(self, path: str | Path | None = None, shrink_minutes: float = SETTLE_SHRINK_MIN):
        """`shrink_minutes` is taken off the time left before scaling the move table: 40 s is the
        random-walk-correct figure for a 60-second settlement average (see settlement.py). The
        rebuilt table was fitted with 30 s and must keep it (range_multi.rebuilt_model)."""
        self.shrink = shrink_minutes
        if path is None:
            raw = resources.files("kalshi_btc_bot").joinpath("data/range_cdf.json").read_text()
        else:
            raw = Path(path).read_text()
        j = json.loads(raw)
        self.qs = np.linspace(0, 1, j["qs_n"])
        self.cdf = {tuple(int(v) for v in k.split("_")): np.array(a) for k, a in j["cdf"].items()}

    @staticmethod
    def bucket(minutes_left: float) -> int:
        return 30 if minutes_left > 20 else (15 if minutes_left > 8 else 5)

    def vol(self, closes: np.ndarray) -> tuple[float, int] | None:
        """closes: the last 101 one-minute closes, ending at or before the decision time."""
        if len(closes) < 101:
            return None
        r = np.diff(np.log(closes[-101:]))
        s100, s10 = r.std(ddof=1), r[-10:].std(ddof=1)
        if not s100 > 0 or not s10 > 0:
            return None
        return float(s100), int(np.digitize(np.log(s10 / s100), [-0.3, 0.3]))

    def p_inside(self, price: float, lo: float | None, hi: float | None, minutes_left: float,
                 s100: float, vb: int) -> float:
        """Chance the settlement lands in [lo, hi] (hi given as the cap strike, e.g. 86399.99)."""
        te = max(minutes_left - self.shrink, 1 / 60)
        scale = s100 * math.sqrt(te)
        c = self.cdf[(self.bucket(minutes_left), vb)]
        F = lambda x: float(np.interp(x, c, self.qs, left=0.0, right=1.0))
        xl = -np.inf if lo is None else math.log(lo / price) / scale
        xh = np.inf if hi is None else math.log((hi + 0.01) / price) / scale
        return (1.0 if xh == np.inf else F(xh)) - (0.0 if xl == -np.inf else F(xl))


def qualifies(model_p: float, ask: float | None) -> bool:
    if ask is None or not LOW_FLOOR <= ask < 1:
        return False
    return model_p >= ask + (LOW_GAP if ask < HIGH_FLOOR else HIGH_GAP)


def spread_ok(ask: float | None, bid: float | None) -> bool:
    """Spread filter for 75-89c entries: ask - bid must be 2c or less (on the 1c grid).

    90c+ entries are not filtered. A missing bid counts as a wide (unknown) book for a 75-89c ask.
    """
    if ask is None:
        return False
    if ask >= HIGH_FLOOR:
        return True
    if bid is None:
        return False
    return round((ask - bid) * 100) <= round(MAX_SPREAD_LOW * 100)


def price_limit(model_p: float) -> float:
    """Highest price that still meets the rule for this model reading."""
    hi = round(model_p - HIGH_GAP, 4)
    if hi >= HIGH_FLOOR:
        return min(hi, 0.99)
    return round(model_p - LOW_GAP, 4)


def fill(other_bids: list[list[float]], contracts: int, limit: float) -> tuple[int, float | None]:
    """Buy up to `contracts` of one side by taking the other side's bids (price = 1 - bid)."""
    got, cost = 0, 0.0
    for b, q in sorted(other_bids, reverse=True):
        p = round(1 - b, 4)
        if p > limit + 1e-9 or got >= contracts:
            break
        n = int(min(q, contracts - got))
        got += n
        cost += n * p
    return got, (cost / got if got else None)


def event_ticker(close_ts: int) -> str:
    e = datetime.fromtimestamp(close_ts, ET)
    return f"{SERIES}-{e:%y}{e:%b}".upper() + f"{e:%d%H}"


def band_mid(m: dict) -> float:
    lo, hi = m.get("floor_strike"), m.get("cap_strike")
    if lo is not None and hi is not None:
        return (float(lo) + float(hi)) / 2
    return float(lo if lo is not None else hi)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------- the trader
class RangePaper:
    def __init__(self, out: str | Path, key_id: str | None, key_path: str | None,
                 push_logs: bool = False, contracts: int = CONTRACTS):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.k = KalshiClient(key_id, key_path)
        self.model = RangeModel()
        self.contracts = contracts
        self.push_logs = push_logs
        self.brti: dict[int, float] = {}
        self.held: dict[str, dict] = {}         # ticker -> open paper position
        self.anchor: dict[int, float] = {}      # close_ts -> BRTI 30 min before close
        self._done_minutes: set[int] = set()
        self._last_push = self._last_settle = 0
        self._load_open()

    # ---------------------------------------------------------------- files
    def _append(self, name: str, row: dict):
        path = self.out / name
        if path.exists():
            with path.open() as f:
                if f.readline().strip() != ",".join(row):
                    path.rename(path.with_suffix(f".{int(time.time())}.csv"))
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    def _save_open(self):
        (self.out / "open_positions.json").write_text(json.dumps(list(self.held.values()), indent=1))

    def _load_open(self):
        p = self.out / "open_positions.json"
        if p.exists():
            for tr in json.loads(p.read_text()):
                self.held[tr["ticker"]] = tr

    def _status(self, now: float, note: str = ""):
        rows = list(csv.DictReader((self.out / "trades.csv").open())) if (self.out / "trades.csv").exists() else []
        by = {}
        for r in rows:
            d = by.setdefault(r["side"], {"trades": 0, "wins": 0, "pnl_dollars": 0.0})
            d["trades"] += 1
            d["wins"] += r["won"] == "1"
            d["pnl_dollars"] = round(d["pnl_dollars"] + float(r["pnl"]), 2)
        (self.out / "status.json").write_text(json.dumps({
            "updated_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds"),
            "settled": by, "open_positions": len(self.held), "note": note}, indent=1))

    def _push(self, now: float):
        if not self.push_logs or now - self._last_push < 3600:
            return
        self._last_push = now
        try:
            git_push_logs(self.out, f"range paper log {datetime.now(timezone.utc):%Y-%m-%d %H:%M}")
        except Exception as e:
            log.warning("log push failed: %s", e)

    # ---------------------------------------------------------------- prices
    def warm_up(self, now: int):
        if not self.k.key:
            raise SystemExit("A Kalshi API key is needed for BRTI (set KALSHI_KEY_ID / KALSHI_KEY_FILE).")
        for h in range((now - 3 * 3600) // 3600 * 3600, now // 3600 * 3600 + 1, 3600):
            try:
                self.brti.update(self.k.brti_hour(h))
            except Exception as e:
                log.warning("BRTI history %s failed: %s", h, e)
        self.poll_brti()
        log.info("warm-up: %d BRTI seconds", len(self.brti))

    def poll_brti(self) -> int | None:
        try:
            self.brti.update(self.k.brti_live())
        except Exception as e:
            log.warning("live BRTI failed: %s", e)
        if self.brti:
            cut = max(self.brti) - 4 * 3600
            if min(self.brti) < cut - 600:
                self.brti = {t: v for t, v in self.brti.items() if t >= cut}
        return max(self.brti) if self.brti else None

    def price_at(self, t: int) -> float | None:
        for u in range(t, t - 30, -1):
            if u in self.brti:
                return self.brti[u]
        return None

    def minute_closes(self, t: int) -> np.ndarray:
        """Closes of the 101 one-minute bars ending at t-1 (no data after the decision time)."""
        out = []
        for k in range(101):
            v = self.price_at(t - 1 - 60 * (100 - k))
            if v is None:
                return np.array([])
            out.append(v)
        return np.array(out)

    def band_edges(self, m: dict) -> tuple[float | None, float | None]:
        lo, hi = _f(m.get("floor_strike")), _f(m.get("cap_strike"))
        if m.get("strike_type") == "greater":
            hi = None
        elif m.get("strike_type") == "less":
            lo = None
        return lo, hi

    def model_yes(self, t: int, price: float, m: dict, left: int) -> tuple[float | None, float | None]:
        """Settlement-aware chance the band wins (range model, 40-s shrink) and the one-sd settlement move in
        dollars at this time left; (None, None) without 100 minutes of BRTI history."""
        closes = self.minute_closes(t)
        v = self.model.vol(closes) if len(closes) else None
        if v is None:
            return None, None
        s100, vb = v
        lo, hi = self.band_edges(m)
        sd = price * s100 * math.sqrt(max(left - self.model.shrink, 1 / 60))
        return self.model.p_inside(price, lo, hi, left, s100, vb), sd

    # ---------------------------------------------------------------- the rule
    def check(self, t: int):
        """Run the rule at minute mark t (a whole minute, before the hourly close)."""
        close = (t // 3600 + 1) * 3600
        left = (close - t) // 60
        if left > NO_WINDOW or left < 1:
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
        markets = self.k.get("/markets", event_ticker=ev, limit=1000).get("markets", [])
        markets = [m for m in markets if m.get("status") in ("active", "open")]
        if not markets:
            log.warning("no open markets for %s", ev)
            return
        markets = sorted(markets, key=lambda m: abs(band_mid(m) - anchor))[:N_BANDS]
        for m in markets:
            tk = m["ticker"]
            if tk in self.held:
                continue
            lo, hi = _f(m.get("floor_strike")), _f(m.get("cap_strike"))
            if m.get("strike_type") == "greater":
                lo, hi = _f(m.get("floor_strike")), None
            elif m.get("strike_type") == "less":
                lo, hi = None, _f(m.get("cap_strike"))
            p_yes = self.model.p_inside(price, lo, hi, left, s100, vb)
            yb, ya = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
            no_ask = round(1 - yb, 4) if yb else None
            no_bid = round(1 - ya, 4) if ya else None
            for side, mp, ask, bid, window in (("YES", p_yes, ya, yb, YES_WINDOW),
                                               ("NO", 1 - p_yes, no_ask, no_bid, NO_WINDOW)):
                if left > window or not qualifies(mp, ask) or tk in self.held:
                    continue
                if not spread_ok(ask, bid):
                    log.info("%s %s %s: ask %.2f qualifies but spread %s is over %dc, skipped", _iso(t), side, tk,
                             ask, f"{round((ask - bid) * 100)}c" if bid is not None else "unknown",
                             round(MAX_SPREAD_LOW * 100))
                    self._append("skipped.csv", {"time_utc": _iso(t), "ticker": tk, "side": side, "mins_left": left,
                                                 "ask_snapshot": ask, "bid_snapshot": bid, "model": round(mp, 4)})
                    continue
                self.enter(t, close, left, m, side, mp, ask, price, s100, vb)

    def enter(self, t, close, left, m, side, mp, ask, price, s100, vb):
        tk = m["ticker"]
        try:
            book = self.k.orderbook(tk)
        except Exception as e:
            log.warning("orderbook %s failed: %s", tk, e)
            book = {}
        other = book.get("no_bids" if side == "YES" else "yes_bids") or []
        limit = price_limit(mp)
        n, avg = fill(other, self.contracts, limit)
        row = {"time_utc": _iso(t), "ticker": tk, "side": side, "mins_left": left,
               "floor": m.get("floor_strike"), "cap": m.get("cap_strike"), "brti": round(price, 2),
               "model": round(mp, 4), "ask_snapshot": ask, "book_best": round(1 - other[0][0], 4) if other else "",
               "limit": limit, "filled": n, "avg_price": round(avg, 4) if avg else "",
               "vol_1m": round(s100, 6), "vol_state": vb, "book": json.dumps(other[:6])}
        self._append("signals.csv", row)
        if n == 0:
            log.info("%s %s %s: signal at %.2f but no fill under %.2f", _iso(t), side, tk, ask, limit)
            return
        fee = kalshi_fee(n, avg)
        self.held[tk] = {"time_utc": _iso(t), "ticker": tk, "side": side, "mins_left": left,
                         "contracts": n, "avg_price": round(avg, 4), "fee": fee,
                         "cost": round(n * avg + fee, 2), "model": round(mp, 4), "close_ts": close}
        self._save_open()
        log.info("PAPER BUY %d %s %s @ %.4f (model %.3f, %d min left)", n, side, tk, avg, mp, left)

    def settle(self, now: int):
        if now - self._last_settle < 60:
            return
        self._last_settle = now
        changed = False
        for tk, tr in list(self.held.items()):
            if now < tr["close_ts"] + 60:
                continue
            try:
                res = self.k.result(tk)
            except Exception as e:
                log.warning("result %s failed: %s", tk, e)
                continue
            if res is None:
                continue
            won = int(res == tr["side"].lower())
            row = {k: v for k, v in tr.items() if k != "close_ts"}
            row.update(result=res, won=won, pnl=round(tr["contracts"] * (won - tr["avg_price"]) - tr["fee"], 2))
            self._append("trades.csv", row)
            del self.held[tk]
            changed = True
            log.info("SETTLED %s %s %s pnl %.2f", tr["side"], tk, res, row["pnl"])
        if changed:
            self._save_open()
        self._status(now)

    # ---------------------------------------------------------------- loop
    def run(self, seconds: float | None = None):
        start = int(time.time())
        self.warm_up(start)
        while seconds is None or time.time() - start < seconds:
            now = time.time()
            t = int(now) // 60 * 60
            # act a few seconds after each minute mark, once BRTI for that second is in
            if t not in self._done_minutes and now - t >= 3:
                latest = self.poll_brti()
                if latest is not None and latest >= t:
                    self._done_minutes.add(t)
                    try:
                        self.check(t)
                    except Exception as e:
                        log.exception("check failed: %s", e)
                elif now - t > 40:
                    self._done_minutes.add(t)
                    log.warning("%s skipped: BRTI not up to date", _iso(t))
                self._done_minutes = {u for u in self._done_minutes if u > t - 7200}
            self.settle(int(now))
            self._push(now)
            time.sleep(1.0)


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def from_env(out: str, push_logs: bool, contracts: int = CONTRACTS) -> RangePaper:
    return RangePaper(out, os.environ.get("KALSHI_KEY_ID"), os.environ.get("KALSHI_KEY_FILE"),
                      push_logs, contracts)
