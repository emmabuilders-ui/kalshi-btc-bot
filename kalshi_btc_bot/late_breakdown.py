"""Late-breakdown NO paper trader for Kalshi KXBTC15M (read-only, no orders are sent).

The rule (locked 2026-10-31 and tested on Dec 2025 - Apr 2026 data it had never seen):

* Levels, fixed at each 15-minute session open, from the 1-second BRTI series:
  - hard bottom: the nearest cluster of >= 2 turns of a $75 zigzag in the prior 24 h,
    turns within a $40 band (first-element span, not chained), more than $10 below the
    price at open;
  - previous 15-minute low, if it is more than $10 below the price at open.
* Trigger: the FIRST time BRTI trades $5 below one of those levels, the entry is the next
  5-second mark. It counts only if that entry is 60-179 s before close.
* Buy NO only if NO is the favourite (BRTI below the strike) and the MA stack is not
  already pointing down (skip if price < SMA15m < SMA60m < SMA24h).
* NO price cap 96c (the edge sits below 97c).

Prices are the real BRTI from Kalshi's CF Benchmarks passthrough (/cfbenchmarks/values,
about 1 s behind, needs an API key). If that feed stalls, the bot falls back to the mean of
exchange mid prices (Coinbase, Kraken, Bitstamp, Gemini) corrected by their measured offset
to BRTI, and the price log marks those seconds as "proxy".
"""

from __future__ import annotations

import base64
import csv
import gzip
import json
import logging
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

log = logging.getLogger("kalshi_btc_bot")

KALSHI = "https://api.elections.kalshi.com"
API = "/trade-api/v2"
SERIES = "KXBTC15M"

ZZ, CLUSTER, MIN_TOUCH, BREAK, GAP = 75.0, 40.0, 2, 5.0, 10.0
SESSION = 900
WARMUP = 2 * 86400


# ----------------------------------------------------------------------------- rule maths
def zigzag(b: np.ndarray, thr: float):
    """Pivot times, prices and the index where each pivot was confirmed."""
    piv_t, piv_p, conf = [0], [b[0]], [0]
    d, ext_t, ext = 0, 0, b[0]
    for t in range(1, len(b)):
        x = b[t]
        if d >= 0:
            if x > ext:
                ext, ext_t = x, t
            if d == 0 and x < ext - thr:
                d = -1
            elif d == 1 and x <= ext - thr:
                piv_t.append(ext_t); piv_p.append(ext); conf.append(t)
                d, ext, ext_t = -1, x, t
                continue
            if d == 0 and x > b[0] + thr:
                d = 1
        if d == -1:
            if x < ext:
                ext, ext_t = x, t
            if x >= ext + thr:
                piv_t.append(ext_t); piv_p.append(ext); conf.append(t)
                d, ext, ext_t = 1, x, t
    return np.array(piv_t), np.array(piv_p), np.array(conf)


def hard_levels(b: np.ndarray, o: int) -> np.ndarray:
    """Hard levels known at index o (session open) from the series b."""
    pt, pp, cf = zigzag(b[:o], ZZ)
    v = np.sort(pp[(cf < o) & (pt >= o - 86400)])
    if len(v) == 0:
        return np.array([])
    groups = [[v[0]]]
    for z in v[1:]:
        if z - groups[-1][0] <= CLUSTER:
            groups[-1].append(z)
        else:
            groups.append([z])
    return np.array([np.mean(g) for g in groups if len(g) >= MIN_TOUCH])


def session_levels(b: np.ndarray, o: int) -> dict[str, float]:
    """Breakdown levels for a session opening at index o of b (b[o-1] = last price before open)."""
    s0 = b[o - 1]
    out = {}
    hl = hard_levels(b, o)
    below = hl[hl < s0 - GAP]
    if len(below):
        out["hard"] = float(below.max())
    low15 = float(np.min(b[o - SESSION:o]))
    if s0 - low15 > GAP:
        out["15min"] = low15
    return out


def ma_stack(b: np.ndarray, i: int) -> str:
    """'down' if b[i] < SMA15m < SMA60m < SMA24h, 'up' for the mirror, else 'mixed'."""
    s = b[i]
    m15, m60, m24 = (float(np.mean(b[i + 1 - n:i + 1])) for n in (900, 3600, 86400))
    if s < m15 < m60 < m24:
        return "down"
    if s > m15 > m60 > m24:
        return "up"
    return "mixed"


def kalshi_fee(contracts: int, price: float) -> float:
    """Kalshi taker fee for one order: 7% x C x P x (1-P), rounded up to the cent."""
    return math.ceil(round(0.07 * contracts * price * (1 - price) * 100, 6)) / 100


def fill_no(yes_bids: list[list[float]], stake: float, max_price: float):
    """Walk the book to buy NO for about `stake` dollars, paying at most max_price per contract.

    Buying NO takes YES bids: a YES bid at y fills NO at 1 - y. Returns (contracts, avg price).
    """
    best = 1 - yes_bids[0][0] if yes_bids else None
    if best is None or best > max_price:
        return 0, best
    want = int(stake // best)
    got, cost = 0, 0.0
    for y, q in sorted(yes_bids, reverse=True):
        p = round(1 - y, 4)
        if p > max_price or got >= want:
            break
        n = int(min(q, want - got))
        got += n
        cost += n * p
    return got, (cost / got if got else best)


# ------------------------------------------------------------------------------- data feeds
class KalshiClient:
    """Read-only Kalshi REST. Signs requests when a key is configured (needed for BRTI history)."""

    def __init__(self, key_id: str | None = None, key_path: str | None = None, timeout: float = 5):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "kalshi-btc-bot-paper"
        self.timeout = timeout
        self.key_id = key_id
        self.key = None
        if key_id and key_path:
            from cryptography.hazmat.primitives.serialization import load_pem_private_key
            self.key = load_pem_private_key(Path(key_path).read_bytes(), None)

    def _sign(self, method: str, path: str) -> dict:
        if not self.key:
            return {}
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa
        ts = str(int(time.time() * 1000))
        msg = (ts + method + path.split("?")[0]).encode()
        if isinstance(self.key, ed25519.Ed25519PrivateKey):
            sig = self.key.sign(msg)
        elif isinstance(self.key, rsa.RSAPrivateKey):
            sig = self.key.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                                 salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
        else:
            raise ValueError("unsupported key type")
        return {"KALSHI-ACCESS-KEY": self.key_id, "KALSHI-ACCESS-TIMESTAMP": ts,
                "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode()}

    def get(self, path: str, retries: int = 4, **params):
        for k in range(retries):
            r = self.s.get(KALSHI + API + path, params=params or None,
                           headers=self._sign("GET", API + path), timeout=self.timeout)
            if r.status_code == 429:
                time.sleep(0.5 * (k + 1))
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()

    def open_market(self, now: float) -> dict | None:
        j = self.get("/markets", series_ticker=SERIES, status="open", limit=10)
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

    def orderbook(self, ticker: str) -> dict:
        from .recorder import parse_book
        return parse_book(self.get(f"/markets/{ticker}/orderbook", depth=20), depth=20)

    def result(self, ticker: str) -> str | None:
        m = self.get(f"/markets/{ticker}")["market"]
        r = (m.get("result") or "").lower()
        return r if r in ("yes", "no") else None

    def brti_live(self) -> dict[int, float]:
        """Real BRTI for the last hour, one value per second, about 1 s behind real time."""
        j = self.get("/cfbenchmarks/values", retries=6, id="BRTI")
        return {p["time"] // 1000: float(p["value"]) for p in j["data"]["payload"] if p["time"] % 1000 == 0}

    def brti_hour(self, hour_ts: int) -> dict[int, float]:
        """Real BRTI, one value per second, for the hour starting at hour_ts (as far as published)."""
        stamp = datetime.fromtimestamp(hour_ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        j = self.get("/cfbenchmarks/history/values", retries=10, id="BRTI", timespan="HOUR", timestamp=stamp)
        return {p["time"] // 1000: float(p["value"]) for p in j["data"]["payload"] if p["time"] % 1000 == 0}


def _mid(b, a):
    return (float(b) + float(a)) / 2


EXCHANGES = {
    "coinbase": ("https://api.exchange.coinbase.com/products/BTC-USD/ticker", lambda j: _mid(j["bid"], j["ask"])),
    "kraken": ("https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
               lambda j: (lambda r: _mid(r["b"][0], r["a"][0]))(next(iter(j["result"].values())))),
    "bitstamp": ("https://www.bitstamp.net/api/v2/ticker/btcusd/", lambda j: _mid(j["bid"], j["ask"])),
    "gemini": ("https://api.gemini.com/v1/pubticker/btcusd", lambda j: _mid(j["bid"], j["ask"])),
}


class ExchangeProxy:
    """Mean of exchange mid prices, polled in parallel."""

    def __init__(self, timeout: float = 0.8):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "kalshi-btc-bot-paper"
        self.timeout = timeout
        self.pool = ThreadPoolExecutor(max_workers=len(EXCHANGES))

    def _one(self, name):
        url, parse = EXCHANGES[name]
        try:
            r = self.s.get(url, timeout=self.timeout)
            r.raise_for_status()
            return parse(r.json())
        except Exception:
            return None

    def price(self) -> tuple[float | None, dict]:
        quotes = dict(zip(EXCHANGES, self.pool.map(self._one, EXCHANGES)))
        ok = [v for v in quotes.values() if v]
        return (float(np.mean(ok)) if len(ok) >= 2 else None), quotes


class Series:
    """1-second price series. Real BRTI wins over the proxy wherever both exist."""

    def __init__(self):
        self.brti: dict[int, float] = {}
        self.proxy: dict[int, float] = {}
        self.bias = 0.0

    def value(self, t: int) -> float | None:
        if t in self.brti:
            return self.brti[t]
        if t in self.proxy:
            return self.proxy[t] - self.bias
        return None

    def array(self, start: int, end: int) -> np.ndarray:
        """Values for seconds [start, end), gaps forward-filled."""
        out = np.empty(end - start)
        last = None
        for i, t in enumerate(range(start, end)):
            v = self.value(t)
            if v is None:
                v = last
            out[i] = np.nan if v is None else v
            last = v
        if np.isnan(out).any():  # back-fill a gap at the very start
            first = out[~np.isnan(out)]
            out[np.isnan(out)] = first[0] if len(first) else np.nan
        return out

    def recalibrate(self, window: int = 1800) -> int:
        """Set the proxy offset from the latest `window` seconds where both prices exist."""
        both = sorted(set(self.brti) & set(self.proxy))
        if not both:
            return 0
        recent = [t for t in both if t > both[-1] - window]
        self.bias = float(np.median([self.proxy[t] - self.brti[t] for t in recent]))
        return len(recent)

    def prune(self, keep_from: int):
        for d in (self.brti, self.proxy):
            for t in [t for t in d if t < keep_from]:
                del d[t]


# ----------------------------------------------------------------------------- the trader
@dataclass
class Session:
    ticker: str
    strike: float
    open_ts: int
    close_ts: int
    s0: float
    levels: dict
    broken: dict = field(default_factory=dict)  # level name -> entry second (te)
    checked: set = field(default_factory=set)   # levels whose break has been evaluated
    done: bool = False                          # a paper trade was taken this session
    live: bool = True                           # False: bot was not calibrated at the open


class LateBreakdownPaper:
    def __init__(self, out: str | Path, stake: float = 20.0, max_price: float = 0.96,
                 key_id: str | None = None, key_path: str | None = None, push_logs: bool = False):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.stake, self.max_price = stake, max_price
        self.k = KalshiClient(key_id, key_path)
        self.px = ExchangeProxy()
        self.series = Series()
        self.session: Session | None = None
        self.unsettled: list[dict] = []
        self.push_logs = push_logs
        self._last_cal = 0
        self.ready_ts: int | None = None  # first second the bot processed with live prices
        self._cursor: int | None = None   # last second fed to the rule
        self._last_proxy = 0
        self._quotes: dict = {}
        self._last_push = 0
        self._last_settle = 0
        self._price_fh = None
        self._price_day = None

    # ---------------------------------------------------------------- logging helpers
    def _append(self, name: str, row: dict):
        path = self.out / name
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    def _log_price(self, t: int, source: str):
        """1-second prices, one gzip file per UTC hour (closed at the hour, so each is complete)."""
        hour = datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d/%H")
        if hour != self._price_day:
            if self._price_fh:
                self._price_fh.close()
            path = self.out / "prices" / f"{hour}.csv.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            new = not path.exists()
            self._price_fh = gzip.open(path, "at")
            if new:
                self._price_fh.write("unix_s,price,source,brti,proxy_raw,proxy_offset\n")
            self._price_day = hour
        v, b, p = self.series.value(t), self.series.brti.get(t), self.series.proxy.get(t)
        f = lambda x: "" if x is None else f"{x:.2f}"
        self._price_fh.write(f"{t},{f(v)},{source},{f(b)},{f(p)},{self.series.bias:.2f}\n")

    def _status(self, now: float, note: str = ""):
        trades = self.out / "trades.csv"
        n = pnl = wins = 0
        if trades.exists():
            for r in csv.DictReader(trades.open()):
                if r.get("result"):
                    n += 1
                    pnl += float(r["pnl"] or 0)
                    wins += r["won"] == "1"
        (self.out / "status.json").write_text(json.dumps({
            "updated_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds"),
            "settled_trades": n, "wins": wins, "pnl_dollars": round(pnl, 2),
            "proxy_bias": round(self.series.bias, 2), "session": self.session.ticker if self.session else None,
            "note": note}, indent=1))

    def _push(self, now: float):
        if not self.push_logs or now - self._last_push < 3600:
            return
        self._last_push = now
        try:
            git = ["git", "-C", str(self.out)]
            subprocess.run(git + ["add", "-A"], check=True, capture_output=True, timeout=60)
            if self._price_day:  # the open hour's file is still being written
                subprocess.run(git + ["reset", "-q", "--", f"prices/{self._price_day}.csv.gz"],
                               capture_output=True, timeout=60)
            subprocess.run(git + ["commit", "-qm", f"paper log {datetime.now(timezone.utc):%Y-%m-%d %H:%M}"],
                           capture_output=True, timeout=60)
            subprocess.run(git + ["push", "-q", "origin", "HEAD"], check=True, capture_output=True, timeout=120)
        except Exception as e:
            log.warning("log push failed: %s", e)

    # ----------------------------------------------------------------------- BRTI sync
    def warm_up(self, now: int):
        """Load real BRTI for the last 48 h (levels need 24 h of zigzag history plus the 24 h SMA)."""
        if not self.k.key:
            raise SystemExit("A Kalshi API key is needed to load BRTI history (set KALSHI_KEY_ID / KALSHI_KEY_FILE).")
        h0 = (now - WARMUP) // 3600 * 3600
        for h in range(h0, now // 3600 * 3600 + 1, 3600):
            self.series.brti.update(self.k.brti_hour(h))
        log.info("warm-up: %d BRTI seconds loaded, latest %s", len(self.series.brti),
                 datetime.fromtimestamp(max(self.series.brti), timezone.utc).isoformat())

    def poll_brti(self) -> int | None:
        """Fetch live BRTI; returns the latest second available."""
        try:
            self.series.brti.update(self.k.brti_live())
        except Exception as e:
            log.warning("live BRTI failed: %s", e)
        return max(self.series.brti) if self.series.brti else None

    def poll_proxy(self, now: int, every: int):
        """Exchange mid prices: a fallback if live BRTI stalls, and its offset to BRTI is tracked."""
        if now - self._last_proxy < every:
            return
        self._last_proxy = now
        raw, self._quotes = self.px.price()
        if raw is not None:
            self.series.proxy[now] = raw
        if now % 120 < every:
            self.series.recalibrate()
            self.series.prune(now - WARMUP - 3600)

    # ---------------------------------------------------------------------- sessions
    def start_session(self, now: int) -> Session | None:
        m = self.k.open_market(now)
        if not m:
            return None
        o = m["open_ts"]
        b = self.series.array(o - WARMUP, o)
        if np.isnan(b).any():
            log.warning("no price history yet for %s", m["ticker"])
            return None
        lv = session_levels(b, len(b))
        s = Session(m["ticker"], m["strike"], o, m["close_ts"], float(b[-1]), lv,
                    live=self.ready_ts is not None and self.ready_ts <= o + 1)
        log.info("session %s strike %.2f open %.2f levels %s%s", s.ticker, s.strike, s.s0,
                 {k: round(v, 2) for k, v in lv.items()}, "" if s.live else " (started mid-session, watch only)")
        self._append("sessions.csv", {"ticker": s.ticker, "open_utc": _iso(o), "strike": s.strike,
                                      "price_at_open": round(s.s0, 2),
                                      "hard_bottom": round(lv["hard"], 2) if "hard" in lv else "",
                                      "prev15_low": round(lv["15min"], 2) if "15min" in lv else "",
                                      "live": int(s.live), "proxy_bias": round(self.series.bias, 2)})
        return s

    def step(self, now: int, price: float):
        s = self.session
        if s is None or not s.live or now >= s.close_ts:
            return
        sec = now - s.open_ts
        for name, lev in s.levels.items():
            if name not in s.broken and lev - price >= BREAK:
                s.broken[name] = (sec // 5 + 1) * 5
                log.info("%s: %s level %.2f broken at %ds (price %.2f)", s.ticker, name, lev, sec, price)
        due = [n for n, te in s.broken.items() if n not in s.checked and sec >= te]
        if s.done or not due:
            return
        name = min(due, key=s.broken.get)
        s.checked.update(due)  # each level's first break is judged once, at its entry time
        te = s.broken[name]
        left = SESSION - te
        b = self.series.array(now - 86400 - 1, now)
        S = float(b[-1])
        stack = ma_stack(b, len(b) - 1)
        row = {"time_utc": _iso(now), "ticker": s.ticker, "level": name, "level_price": round(s.levels[name], 2),
               "secs_left": left, "price": round(S, 2), "strike": s.strike, "stack": stack,
               "proxy_bias": round(self.series.bias, 2)}
        reason = ""
        if te > 840 or not 60 <= left < 180:
            reason = "outside 1-2 min window"
        elif S >= s.strike:
            reason = "NO not favourite"
        elif stack == "down":
            reason = "MA stack already down"
        book = None
        if not reason or 30 <= left < 300:
            try:
                book = self.k.orderbook(s.ticker)
            except Exception as e:
                log.warning("orderbook failed: %s", e)
        yb = (book or {}).get("yes_bids") or []
        no_ask = round(1 - yb[0][0], 4) if yb else None
        row.update(no_ask=no_ask, no_ask_size=yb[0][1] if yb else "",
                   book=json.dumps(yb[:8]) if yb else "")
        if not reason:
            if no_ask is None:
                reason = "empty book"
            elif no_ask > self.max_price:
                reason = f"NO ask {no_ask:.2f} above cap"
        row["decision"] = "BUY NO" if not reason else "skip: " + reason
        self._append("signals.csv", row)
        log.info("%s %s", s.ticker, row["decision"])
        if reason:
            return
        n, avg = fill_no(yb, self.stake, self.max_price)
        if n == 0:
            return
        s.done = True
        fee = kalshi_fee(n, avg)
        trade = {"time_utc": _iso(now), "ticker": s.ticker, "level": name, "secs_left": left,
                 "contracts": n, "avg_price": round(avg, 4), "fee": fee, "cost": round(n * avg + fee, 2),
                 "close_ts": s.close_ts, "result": "", "won": "", "pnl": ""}
        self.unsettled.append(trade)
        log.info("PAPER BUY %d NO %s @ %.2f (fee %.2f)", n, s.ticker, avg, fee)

    def settle(self, now: int):
        if now - self._last_settle < 30:
            return
        self._last_settle = now
        for tr in list(self.unsettled):
            if now < tr["close_ts"] + 20:
                continue
            try:
                res = self.k.result(tr["ticker"])
            except Exception as e:
                log.warning("result lookup failed: %s", e)
                continue
            if res is None:
                continue
            won = int(res == "no")
            tr.update(result=res, won=won,
                      pnl=round(tr["contracts"] * (won - tr["avg_price"]) - tr["fee"], 2))
            self._append("trades.csv", {k: v for k, v in tr.items() if k != "close_ts"})
            self.unsettled.remove(tr)
            log.info("SETTLED %s %s pnl %.2f", tr["ticker"], res, tr["pnl"])
        self._status(now)

    # -------------------------------------------------------------------------- loop
    def run(self, seconds: float | None = None):
        start = int(time.time())
        self.warm_up(start)
        while seconds is None or time.time() - start < seconds:
            tick = time.time()
            now = int(tick)
            latest = self.poll_brti()
            stale = latest is None or latest < now - 5
            self.poll_proxy(now, 1 if stale else 10)
            if stale:  # BRTI feed stalled: run on the corrected exchange price up to now
                if latest is not None and now - latest > 15:
                    log.warning("live BRTI is %ss behind; using exchange prices", now - latest)
                latest = now if now in self.series.proxy else latest
            if latest is not None:
                if self._cursor is None:
                    self._cursor = latest - 1
                    self.ready_ts = latest
                for t in range(self._cursor + 1, latest + 1):
                    self.process(t, "brti" if t in self.series.brti else "proxy")
                self._cursor = max(self._cursor, latest)
            self.settle(now)
            self._push(now)
            time.sleep(max(0.0, 1.0 - (time.time() - tick)))

    def process(self, t: int, source: str):
        """Feed one second to the rule: roll the session at its close, then check for breaks."""
        price = self.series.value(t)
        if price is None:
            return
        self._log_price(t, source)
        if self.session is None or t >= self.session.close_ts:
            try:
                self.session = self.start_session(t)
            except Exception as e:
                log.warning("session start failed: %s", e)
                self.session = None
            # Kalshi often lists the next market ~30 s after it opens: replay the seconds missed.
            if self.session and self.session.live:
                for u in range(self.session.open_ts, t):
                    v = self.series.value(u)
                    if v is not None:
                        self.step(u, v)
        if self.session:
            try:
                self.step(t, price)
            except Exception as e:
                log.exception("step failed: %s", e)


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def from_env(out: str, stake: float, max_price: float, push_logs: bool) -> LateBreakdownPaper:
    return LateBreakdownPaper(out, stake, max_price, os.environ.get("KALSHI_KEY_ID"),
                              os.environ.get("KALSHI_KEY_FILE"), push_logs)
