"""Break-with-gate PAPER trader for the ETH, SOL and XRP 15-minute markets (read-only, no orders are sent).

Michael asked for it on 2026-11-12 ("make the bot for all of them") after the cross-coin test in
strategy/other_coins_lockin_break_test.md. The rule is the BTC "break 60-150 s with the average gate" rule, frozen from
the BTC work and run on each coin's own CF Benchmarks index and its own Kalshi 15-minute market:

* Break: the coin's index crosses the strike and ends up at least $5 x (price / 75,000) beyond it on the new side
  (about 0.0067% of the price, so $5 on BTC at 75k, 17c on ETH at 2,500). "Crosses" means that the last time the index
  was at least that far from the strike it was on the other side. The side the index is on is the side bought. The
  watch starts 210 s before the close; an entry counts only with 60-150 s left; one trade per contract.
* Gate: the settlement-aware chance of that side (the 60-second index average, volatility from the last 300 s of
  1-second prints, variance sigma^2 x (secs left - 40)) must be at least the ask plus the taker fee. That is the same
  check the BTC break bots use since PR #32.
* Entry at the ask of the order book as soon as the break is seen (the live index is about 1 s behind), 20 contracts,
  walking the book only while the gate still holds, held to settlement. Skipped if the decision comes more than
  5 s after the break second, or if the ask is outside 2-98c.

Replay of the first four steps (Dec 2025 - Nov 11 2026, own index and real Kalshi trade prints, one trade per contract),
EV after the fee, Dec-Aug / Sep-Oct / Nov 1-11:

  SOL  +10.3c / +6.8c / +3.6c   (1,376 / 357 / 42 trades, 8/8 and 2/2 months positive)
  XRP   +9.8c / +5.6c / +2.5c   (1,191 / 307 / 37 trades, 7/7 and 2/2)
  ETH   +8.8c / +1.8c / +6.6c   (1,071 / 363 / 57 trades, 8/9 and 1/2; October -3.8c)

The replay priced fills at the last real taker print (within 15 s), not at a live ask, and the books on these coins
are thin, so the paper test is the check on fills. Every edge shrank from Dec-Aug to Sep-Oct. Retire a coin if its
EV is under +1c at 300 trades; ETH is included at Michael's request even though it missed the +3c Sep-Oct bar.
The level-based break bots (hard levels, moving-average check) are not part of this rule.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import os
import resource
import signal
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .late_breakdown import KalshiClient, _iso, kalshi_fee
from .logpush import git_push_logs
from .range_paper import CONTRACTS, fill
from .settlement import settle_prob, side_prob, sigma_1s, taker_fee

log = logging.getLogger("kalshi_btc_bot")

COINS = {"ETH": ("KXETH15M", "ETHUSD_RTI"), "SOL": ("KXSOL15M", "SOLUSD_RTI"), "XRP": ("KXXRP15M", "XRPUSD_RTI")}
WATCH_SECS = 210               # start tracking the side the index is on this long before the close
WINDOW = (60, 150)             # seconds left at entry, inclusive
BREAK_FRAC = 5.0 / 75000.0     # $5 on BTC at 75,000, scaled to each coin's own price
VOL_SECS = 300                 # seconds of 1-s prints behind the volatility estimate
ASK_MIN, ASK_MAX = 0.02, 0.98
MAX_LAG = 5                    # skip if the decision comes more than 5 s after the break (wall clock)
KEEP_SECS = 4000               # seconds of index history kept in memory
POLL_EVERY = 2                 # seconds between live index calls per coin (each returns the last hour)


# ----------------------------------------------------------------------------- pure rule pieces
def break_threshold(price: float) -> float:
    """Distance beyond the strike that counts as a break, scaled to the coin's price."""
    return price * BREAK_FRAC


def side_of(price: float, strike: float) -> str:
    return "yes" if price >= strike else "no"


def next_state(prev: str | None, price: float, strike: float) -> tuple[str | None, str | None]:
    """One second of the break tracker. Returns (new tracked side, break side or None).

    The tracked side is the side the index was last at least one threshold away from the strike. A break is the
    index getting a threshold beyond the strike on the other side from it."""
    thr = break_threshold(price)
    if abs(price - strike) < thr:
        return prev, None
    side = side_of(price, strike)
    return side, (side if prev is not None and side != prev else None)


def gate_limit(p_side: float, levels: list[list[float]]) -> float:
    """Highest book price (the ask side of `levels`, as [[price, qty]]) where the gate still holds."""
    lim = 0.0
    for price, _ in levels:
        if p_side >= price + taker_fee(price):
            lim = max(lim, price)
    return lim


# ----------------------------------------------------------------------------- feed and client
class CoinClient(KalshiClient):
    """Kalshi REST for one coin: its 15-minute series and its CF Benchmarks index."""

    def __init__(self, series: str, index_id: str, key_id: str | None, key_path: str | None):
        super().__init__(key_id, key_path)
        self.series, self.index_id = series, index_id

    def open_market(self, now: float) -> dict | None:
        j = self.get("/markets", series_ticker=self.series, status="open", limit=10)
        best = None
        for m in j.get("markets", []):
            close_ts = datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).timestamp()
            open_ts = datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp()
            if m.get("floor_strike") is None:  # listed a few seconds before its strike is set
                continue
            if open_ts <= now < close_ts and (best is None or close_ts < best["close_ts"]):
                best = {"ticker": m["ticker"], "strike": float(m["floor_strike"]),
                        "open_ts": int(open_ts), "close_ts": int(close_ts)}
        return best

    def index_live(self) -> dict[int, float]:
        """The index for the last hour, one value per second, about 1 s behind real time."""
        j = self.get("/cfbenchmarks/values", retries=6, id=self.index_id)
        return {p["time"] // 1000: float(p["value"]) for p in j["data"]["payload"] if p["time"] % 1000 == 0}


@dataclass
class Session:
    ticker: str
    strike: float
    open_ts: int
    close_ts: int
    cursor: int                      # next second to feed to the rule
    prev: str | None = None          # side the index was last a threshold away from the strike
    done: bool = False               # this contract's one trade (or skip) has been used


@dataclass
class Coin:
    name: str
    client: CoinClient
    out: Path
    series: dict = field(default_factory=dict)
    session: Session | None = None
    unsettled: list = field(default_factory=list)
    last_market_try: float = 0.0
    last_settle: float = 0.0
    last_poll_ok: int = 0
    last_flush: int = 0
    logged_to: int = 0               # last index second written to the price log


class CoinBreakPaper:
    def __init__(self, out: str | Path, key_id: str | None = None, key_path: str | None = None,
                 push_logs: bool = False, contracts: int = CONTRACTS, coins: tuple[str, ...] = tuple(COINS)):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.contracts, self.push_logs = contracts, push_logs
        self.coins: dict[str, Coin] = {}
        for c in coins:
            series, idx = COINS[c]
            d = self.out / c
            d.mkdir(parents=True, exist_ok=True)
            coin = Coin(c, CoinClient(series, idx, key_id, key_path), d)
            f = d / "open_trades.json"
            if f.exists():  # paper trades bought before a restart still get settled
                coin.unsettled = json.loads(f.read_text())
            self.coins[c] = coin
        self.pool = ThreadPoolExecutor(max_workers=len(self.coins))
        self._last_push = 0.0

    # ------------------------------------------------------------------------------ logging
    @staticmethod
    def _append(coin: Coin, name: str, row: dict):
        path = coin.out / name
        if path.exists():
            with path.open() as f:
                header = f.readline().strip()
            if header != ",".join(row):  # columns changed: keep the old file aside
                path.rename(path.with_suffix(f".{int(time.time())}.csv"))
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    @staticmethod
    def _save_open(coin: Coin):
        (coin.out / "open_trades.json").write_text(json.dumps(coin.unsettled, indent=1))

    def _status(self, coin: Coin, now: float, note: str = ""):
        n = pnl = wins = 0
        f = coin.out / "trades.csv"
        if f.exists():
            for r in csv.DictReader(f.open()):
                if r.get("result"):
                    n += 1
                    pnl += float(r["pnl"] or 0)
                    wins += r["won"] == "1"
        (coin.out / "status.json").write_text(json.dumps({
            "updated_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds"),
            "coin": coin.name, "settled_trades": n, "wins": wins, "pnl_dollars": round(pnl, 2),
            "session": coin.session.ticker if coin.session else None,
            "max_mem_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, "note": note}, indent=1))

    def _log_prices(self, coin: Coin, now: int):
        """Append the index seconds not yet logged to hourly files out/<COIN>/prices/YYYY-MM-DD/HH.csv.gz (one gzip
        member per flush, header "unix_s,price"). The dashboard chart reads only these files, so it makes no API
        calls. A restart re-logs the last hour; the reader keeps one value per second."""
        if now - coin.last_flush < 60 or not coin.series:
            return
        coin.last_flush = now
        new = sorted(t for t in coin.series if t > coin.logged_to)
        by_hour: dict[int, list[int]] = {}
        for t in new:
            by_hour.setdefault(t // 3600 * 3600, []).append(t)
        for h, ts_ in by_hour.items():
            path = coin.out / "prices" / datetime.fromtimestamp(h, timezone.utc).strftime("%Y-%m-%d") / (
                datetime.fromtimestamp(h, timezone.utc).strftime("%H") + ".csv.gz")
            path.parent.mkdir(parents=True, exist_ok=True)
            text = ("" if path.exists() else "unix_s,price\n") + "".join(f"{t},{coin.series[t]}\n" for t in ts_)
            with path.open("ab") as f:
                f.write(gzip.compress(text.encode()))
        if new:
            coin.logged_to = new[-1]

    def _push(self, now: float):
        if not self.push_logs or now - self._last_push < 3600:
            return
        self._last_push = now
        try:
            hour = datetime.now(timezone.utc).strftime("%Y-%m-%d/%H")   # this hour's price files are still being written
            git_push_logs(self.out, f"paper log {datetime.now(timezone.utc):%Y-%m-%d %H:%M}",
                          unstage=[f"{c}/prices/{hour}.csv.gz" for c in self.coins])
        except Exception as e:
            log.warning("log push failed: %s", e)

    # ----------------------------------------------------------------------------- the data
    @staticmethod
    def _array(coin: Coin, start: int, end: int) -> np.ndarray:
        """Index values for seconds [start, end), gaps forward-filled; NaN where nothing is known yet."""
        out = np.full(end - start, np.nan)
        last = None
        for i, t in enumerate(range(start, end)):
            v = coin.series.get(t, last)
            if v is not None:
                out[i], last = v, v
        return out

    def poll(self, coin: Coin) -> int | None:
        try:
            coin.series.update(coin.client.index_live())
        except Exception as e:
            log.warning("%s live index failed: %s", coin.name, e)
        if not coin.series:
            return None
        latest = max(coin.series)
        for t in [t for t in coin.series if t < latest - KEEP_SECS]:
            del coin.series[t]
        return latest

    def new_session(self, coin: Coin, now: float) -> Session | None:
        if now - coin.last_market_try < 5:
            return None
        coin.last_market_try = now
        try:
            m = coin.client.open_market(now)
        except Exception as e:
            log.warning("%s market lookup failed: %s", coin.name, e)
            return None
        if not m:
            return None
        s = Session(m["ticker"], m["strike"], m["open_ts"], m["close_ts"], cursor=max(m["open_ts"], m["close_ts"] - WATCH_SECS))
        log.info("%s session %s strike %s", coin.name, s.ticker, s.strike)
        self._append(coin, "sessions.csv", {"ticker": s.ticker, "open_utc": _iso(s.open_ts), "close_utc": _iso(s.close_ts),
                                            "strike": s.strike})
        return s

    # ------------------------------------------------------------------------------ the rule
    def step(self, coin: Coin, t: int):
        s = coin.session
        price = coin.series.get(t)
        if s is None or price is None or not s.cursor <= t < s.close_ts:
            return
        s.prev, brk = next_state(s.prev, price, s.strike)
        left = s.close_ts - t
        if brk is None or s.done or not WINDOW[0] <= left <= WINDOW[1]:
            return
        s.done = True                       # the first break in the window is this contract's one look
        self.decide(coin, s, t, price, brk)

    def decide(self, coin: Coin, s: Session, t: int, price: float, side: str):
        left = s.close_ts - t
        lag = int(time.time()) - t
        b = self._array(coin, t - VOL_SECS, t + 1)
        sig = sigma_1s(b) if not np.isnan(b).any() else 0.0
        p_yes = settle_prob(price, s.strike, left, sig) if sig > 0 else 0.5
        p_side = side_prob(p_yes, side)
        row = {"time_utc": _iso(t), "coin": coin.name, "ticker": s.ticker, "side": side.upper(), "secs_left": left,
               "index": round(price, 6), "strike": s.strike, "break_threshold": round(break_threshold(price), 6),
               "sigma_1s": round(sig, 8), "model": round(p_side, 4), "lag_s": lag}
        reason, ask, bids = "", None, []
        if lag > MAX_LAG:
            reason = f"decision {lag}s after the break"
        elif sig <= 0:
            reason = "no volatility estimate"
        else:
            try:
                book = coin.client.orderbook(s.ticker)
            except Exception as e:
                log.warning("%s orderbook failed: %s", coin.name, e)
                book = None
            bids = (book or {}).get("no_bids" if side == "yes" else "yes_bids") or []
            ask = round(1 - bids[0][0], 4) if bids else None
            if ask is None:
                reason = "empty book"
            elif not ASK_MIN <= ask <= ASK_MAX:
                reason = f"{side.upper()} ask {ask:.2f} outside {ASK_MIN:.2f}-{ASK_MAX:.2f}"
            elif p_side < ask + taker_fee(ask):
                reason = f"settlement chance {p_side:.2f} under ask {ask:.2f} + fee"
        row.update(ask=ask, ask_size=bids[0][1] if bids else "", book=json.dumps(bids[:8]) if bids else "")
        n = avg = 0
        if not reason:
            levels = [[round(1 - p, 4), q] for p, q in bids]
            limit = min(gate_limit(p_side, levels), ASK_MAX)
            n, avg = fill(bids, self.contracts, limit)
            if not n:
                reason = "nothing to buy with the gate holding"
        row["decision"] = f"BUY {side.upper()}" if not reason else "skip: " + reason
        row.update(contracts=n or "", avg_price=round(avg, 4) if n else "")
        self._append(coin, "signals.csv", row)
        log.info("%s %s %s", coin.name, s.ticker, row["decision"])
        if reason:
            return
        fee = kalshi_fee(n, avg)
        coin.unsettled.append({"time_utc": _iso(t), "coin": coin.name, "ticker": s.ticker, "side": side.upper(),
                               "secs_left": left, "contracts": n, "avg_price": round(avg, 4), "fee": fee,
                               "cost": round(n * avg + fee, 2), "model": round(p_side, 4), "close_ts": s.close_ts,
                               "result": "", "won": "", "pnl": ""})
        self._save_open(coin)
        log.info("%s PAPER BUY %d %s %s @ %.2f (fee %.2f)", coin.name, n, side.upper(), s.ticker, avg, fee)

    def settle(self, coin: Coin, now: int):
        if now - coin.last_settle < 30:
            return
        coin.last_settle = now
        for tr in list(coin.unsettled):
            if now < tr["close_ts"] + 20:
                continue
            try:
                res = coin.client.result(tr["ticker"])
            except Exception as e:
                log.warning("%s result lookup failed: %s", coin.name, e)
                continue
            if res is None:
                continue
            won = int(res == tr["side"].lower())
            tr.update(result=res, won=won, pnl=round(tr["contracts"] * (won - tr["avg_price"]) - tr["fee"], 2))
            self._append(coin, "trades.csv", {k: v for k, v in tr.items() if k != "close_ts"})
            coin.unsettled.remove(tr)
            self._save_open(coin)
            log.info("%s SETTLED %s %s pnl %.2f", coin.name, tr["ticker"], res, tr["pnl"])
        self._status(coin, now)

    # ------------------------------------------------------------------------------- the loop
    def _life(self, msg: str):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        with (self.out / "crash.log").open("a") as f:
            f.write(f"{_iso(time.time())} pid {os.getpid()} max_mem {rss}MB: {msg}\n")

    def run(self, seconds: float | None = None):
        def on_signal(num, frame):
            raise SystemExit(f"stopped by {signal.Signals(num).name}")
        signal.signal(signal.SIGTERM, on_signal)
        self._life("start")
        try:
            self._loop(seconds)
        except SystemExit as e:
            self._life(str(e) or "exit")
            raise
        except BaseException:
            self._life("crashed:\n" + traceback.format_exc())
            raise
        self._life("finished")

    def _loop(self, seconds: float | None = None):
        start = time.time()
        while seconds is None or time.time() - start < seconds:
            tick = time.time()
            now = int(tick)
            # each live index call returns the last hour, so a coin is polled every 2 s, staggered, to go easy on
            # the rate limit the other bots share
            due = [c for i, c in enumerate(self.coins.values()) if (now + i) % POLL_EVERY == 0]
            latest = {c.name: max(c.series) if c.series else None for c in self.coins.values()}
            latest.update(zip([c.name for c in due], self.pool.map(self.poll, due)))
            for name, coin in self.coins.items():
                try:
                    self.tick_coin(coin, now, latest[name])
                except Exception:
                    log.exception("%s tick failed", name)
            self._push(now)
            time.sleep(max(0.0, 1.0 - (time.time() - tick)))

    def tick_coin(self, coin: Coin, now: int, latest: int | None):
        if latest is None:
            return
        self._log_prices(coin, now)
        s = coin.session
        if s is None or now >= s.close_ts:
            coin.session = s = self.new_session(coin, now)
        if s is not None:
            # feed every second not yet seen, in order, so a restart mid-session rebuilds the tracked side
            for t in range(s.cursor, min(latest, s.close_ts - 1) + 1):
                self.step(coin, t)
            s.cursor = max(s.cursor, min(latest, s.close_ts - 1) + 1)
        self.settle(coin, now)


def from_env(out: str, push_logs: bool, contracts: int = CONTRACTS, coins: tuple[str, ...] = tuple(COINS)) -> CoinBreakPaper:
    return CoinBreakPaper(out, os.environ.get("KALSHI_KEY_ID"), os.environ.get("KALSHI_KEY_FILE"), push_logs,
                          contracts, coins)
