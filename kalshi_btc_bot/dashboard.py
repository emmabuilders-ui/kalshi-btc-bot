"""Live chart of the paper bots, served from the server itself (read-only, no orders).

Open http://<server-ip>:8080/<token>/ in any browser. The page refreshes every 2 seconds:
the live BRTI price (Kalshi's cfbenchmarks feed, 1-second values), the hourly range bot's open
bands, the 15-minute strike, and the buys of the two 15-minute bots, with settled P/L so far.

The token (a random path, made by setup.sh) keeps the page private to whoever has the link.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .fast_push import all_levels
from .late_breakdown import WARMUP, KalshiClient
from .range_paper import N_BANDS, RangeModel, _f, band_mid, event_ticker, qualifies

log = logging.getLogger("kalshi_btc_bot")

KEEP = 3900          # seconds of price history to serve (a bit over one hour)
HOLD = 3 * 3600      # seconds kept in memory (the range model needs 100 one-minute closes)


def _rows(path: Path) -> list[dict]:
    try:
        with path.open() as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []


def _json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def range_positions(range_dir: Path) -> list[dict]:
    """Open range positions with their band edges (from the signal that opened them)."""
    held = _json(range_dir / "open_positions.json", [])
    edges = {r["ticker"]: (_num(r.get("floor")), _num(r.get("cap"))) for r in _rows(range_dir / "signals.csv")}
    out = []
    for p in held:
        lo, hi = edges.get(p["ticker"], (None, None))
        out.append({**p, "lo": lo, "hi": hi})
    return out


def settled_summary(rows: list[dict]) -> dict:
    pnl = [_num(r.get("pnl")) for r in rows]
    pnl = [x for x in pnl if x is not None]
    won = sum(1 for r in rows if str(r.get("won")) == "1")
    return {"trades": len(pnl), "wins": won, "pnl": round(sum(pnl), 2)}


def pnl_points(rows: list[dict], bot: str) -> list[dict]:
    """Settled trades as (time, bot, pnl) points for the cumulative P/L chart."""
    out = []
    for r in rows:
        x = _num(r.get("pnl"))
        if x is not None and r.get("time_utc"):
            out.append({"bot": bot, "time_utc": r["time_utc"], "pnl": x, "won": str(r.get("won")) == "1"})
    return out


def session_buys(rows: list[dict], ticker: str, bot: str) -> list[dict]:
    out = []
    for r in rows:
        if r.get("ticker") != ticker or not str(r.get("decision", "")).startswith("BUY"):
            continue
        t = r.get("break_utc") or r.get("time_utc")
        out.append({"bot": bot, "time_utc": t, "side": r.get("side"), "level": r.get("level"),
                    "level_price": _num(r.get("level_price")), "ask": _num(r.get("ask"))})
    return out


def _ts(s: str) -> int:
    return int(datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())


def read_price_hour(path: Path, res: int) -> list[list[float]]:
    """One hour of the breakdown bot's 1-second price log, cut to one point per `res` seconds."""
    out: dict[int, float] = {}
    try:
        with gzip.open(path, "rt") as f:
            for line in f:
                a = line.split(",")
                if not a[0].isdigit() or not a[1]:
                    continue
                out[int(a[0]) // res * res] = float(a[1])
    except (OSError, EOFError, ValueError):
        pass  # the current hour is still being written; the live feed covers it
    return [[t, out[t]] for t in sorted(out)]


def _hour_end(key: str) -> int:
    """'YYYY-MM-DD/HH' -> unix time at the end of that UTC hour."""
    d = datetime.strptime(key, "%Y-%m-%d/%H").replace(tzinfo=timezone.utc)
    return int(d.timestamp()) + 3600


def history_events(L: dict[str, Path]) -> dict:
    """Everything the bots did, for the full-history chart."""
    edges = {r["ticker"]: (_num(r.get("floor")), _num(r.get("cap"))) for r in _rows(L["range"] / "signals.csv")}
    rng = []
    for r in _rows(L["range"] / "trades.csv"):
        lo, hi = edges.get(r["ticker"], (None, None))
        t = _ts(r["time_utc"])
        rng.append({"t": t, "close": (t // 3600 + 1) * 3600, "side": r["side"], "lo": lo, "hi": hi,
                    "n": r.get("contracts"), "price": _num(r.get("avg_price")), "model": _num(r.get("model")),
                    "status": "won" if str(r.get("won")) == "1" else "lost", "pnl": _num(r.get("pnl")),
                    "cost": _num(r.get("cost"))})
    for p in range_positions(L["range"]):
        t = _ts(p["time_utc"])
        rng.append({"t": t, "close": p["close_ts"], "side": p["side"], "lo": p["lo"], "hi": p["hi"],
                    "n": p["contracts"], "price": p["avg_price"], "model": p.get("model"), "status": "open", "pnl": None,
                    "cost": p.get("cost")})
    rng += fastpush_range_events(L.get("fastpush_range"))
    sessions = []
    for r in _rows(L["breakdown"] / "sessions.csv"):
        k = _num(r.get("strike"))
        if k and r.get("open_utc"):
            sessions.append([_ts(r["open_utc"]), k])
    # support (below the price at open) and resistance (above) levels per 15-minute session
    lv: dict[int, list] = {}
    for r in _rows(L["breakdown"] / "sessions.csv"):
        if not r.get("open_utc"):
            continue
        out = []
        for name, side in (("hard_bottom", "no"), ("prev15_low", "no"), ("hard_top", "yes"), ("prev15_high", "yes")):
            v = _num(r.get(name))
            if v:
                out.append([name, side, v])
        lv[_ts(r["open_utc"])] = out
    for r in _rows(L["fastpush"] / "sessions.csv"):
        try:
            d = json.loads(r.get("levels") or "{}")
        except ValueError:
            continue
        if r.get("open_utc") and d:
            lv[_ts(r["open_utc"])] = [[k, v[0], float(v[1])] for k, v in d.items()]
    levels = [[t, lv[t]] for t in sorted(lv)]
    buys = []
    for bot, key in (("Breakdown", "breakdown"), ("Fast push", "fastpush")):
        res = {(r.get("ticker"), r.get("side")): r for r in _rows(L[key] / "trades.csv")}
        for r in _rows(L[key] / "signals.csv"):
            if not str(r.get("decision", "")).startswith("BUY"):
                continue
            tr = res.get((r.get("ticker"), r.get("side")))
            st = "open" if tr is None else ("won" if str(tr.get("won")) == "1" else "lost")
            buys.append({"bot": bot, "t": _ts(r.get("break_utc") or r["time_utc"]), "side": r.get("side"),
                         "level": r.get("level"), "ask": _num(r.get("ask")), "status": st,
                         "n": (tr or {}).get("contracts") or r.get("contracts"),
                         "price": _num(tr.get("avg_price")) if tr else None,
                         "cost": _num(tr.get("cost")) if tr else None,
                         "pnl": _num(tr.get("pnl")) if tr else None})
    return {"range": rng, "sessions": sessions, "levels": levels, "buys": buys}


def fastpush_range_events(d: Path | None) -> list:
    """The fast-push range bot's trades in the range-trade shape (band shaded on the chart), bot 'FP range'."""
    if d is None:
        return []
    out = []
    done = set()
    for r in _rows(d / "trades.csv"):
        t = _ts(r["time_utc"])
        done.add((r.get("ticker"), r.get("side")))
        out.append({"t": t, "close": (t // 3600 + 1) * 3600, "side": r["side"], "bot": "FP range",
                    "lo": _num(r.get("floor")), "hi": _num(r.get("cap")), "level": r.get("level"),
                    "n": r.get("contracts"), "price": _num(r.get("avg_price")), "model": None,
                    "status": "won" if str(r.get("won")) == "1" else "lost", "pnl": _num(r.get("pnl")),
                    "cost": _num(r.get("cost"))})
    for r in _rows(d / "signals.csv"):  # a BUY with no settled trade yet is an open position
        if not str(r.get("decision", "")).startswith("BUY") or (r.get("ticker"), r.get("side")) in done:
            continue
        t = _ts(r["break_utc"]) + 1
        n = r.get("contracts") or 20
        out.append({"t": t, "close": (t // 3600 + 1) * 3600, "side": r["side"], "bot": "FP range",
                    "lo": _num(r.get("floor")), "hi": _num(r.get("cap")), "level": r.get("level"),
                    "n": n, "price": _num(r.get("ask")), "model": None, "status": "open", "pnl": None, "cost": None})
    return out


def current_levels(L: dict[str, Path], s: dict) -> list:
    """Support/resistance for the 15-minute session now running, as the fast-push bot logged them at its open."""
    if not s.get("open_ts"):
        return []
    for r in reversed(_rows(L["fastpush"] / "sessions.csv")):
        if r.get("open_utc") and _ts(r["open_utc"]) == s["open_ts"]:
            try:
                return [s["open_ts"], [[k, v[0], float(v[1])] for k, v in json.loads(r["levels"]).items()]]
            except (ValueError, KeyError, TypeError):
                return []
    return []


def hour_seconds(path: Path, h0: int) -> np.ndarray:
    """One hour of the breakdown bot's price log as 3600 one-second values (nan where missing)."""
    a = np.full(3600, np.nan)
    try:
        with gzip.open(path, "rt") as f:
            for line in f:
                x = line.split(",")
                if x[0].isdigit() and x[1]:
                    i = int(x[0]) - h0
                    if 0 <= i < 3600:
                        a[i] = float(x[1])
    except (OSError, EOFError, ValueError):
        pass
    return a


class LevelTracker:
    """Support and resistance recalculated every minute from the price history (the same 13 kinds of
    level the fast-push bot uses, measured from the price at that minute). Results are cached on disk."""

    def __init__(self, prices_dir: Path, cache: Path):
        self.dir, self.cache = prices_dir, cache
        self.rows: dict[int, dict] = {}
        try:
            self.rows = {int(k): v for k, v in json.loads(cache.read_text()).items()}
        except (OSError, ValueError):
            pass
        self.secs: dict[int, np.ndarray] = {}     # hour start -> 3600 one-second prices
        self.live: dict[int, float] = {}

    def _load_hours(self, upto: int):
        cur = upto // 3600 * 3600
        for f in sorted(self.dir.glob("*/*.csv.gz")):
            try:
                h0 = int(datetime.strptime(f"{f.parent.name} {f.name[:2]}", "%Y-%m-%d %H").replace(tzinfo=timezone.utc).timestamp())
            except ValueError:
                continue
            if h0 < upto - WARMUP - 7200:
                continue
            if h0 not in self.secs or h0 >= cur - 3600:
                self.secs[h0] = hour_seconds(f, h0)
        for h0 in [h for h in self.secs if h < upto - WARMUP - 7200]:
            del self.secs[h0]

    def series(self, start: int, end: int) -> np.ndarray:
        """One-second prices for [start, end), gaps filled forward."""
        b = np.full(end - start, np.nan)
        for h0, a in self.secs.items():
            lo, hi = max(start, h0), min(end, h0 + 3600)
            if lo < hi:
                b[lo - start:hi - start] = a[lo - h0:hi - h0]
        for t, v in self.live.items():
            if start <= t < end and np.isnan(b[t - start]):
                b[t - start] = v
        ok = ~np.isnan(b)
        if not ok.any():
            return b[:0]
        first = int(np.argmax(ok))
        b = b[first:]
        idx = np.where(~np.isnan(b), np.arange(len(b)), 0)
        np.maximum.accumulate(idx, out=idx)
        return b[idx]

    def update(self, now: float, live: dict[int, float], budget: float = 20.0):
        """Fill in any missing minutes up to now (oldest first, at most `budget` seconds of work)."""
        self.live = live
        self._load_hours(int(now))
        if not self.secs:
            return
        first = min(self.secs)
        t_end = int(now) // 60 * 60
        todo = [t for t in range(max(first + 900, t_end - 30 * 86400) // 60 * 60, t_end + 1, 60) if t not in self.rows]
        t0, done = time.time(), 0
        for t in todo:
            b = self.series(t - WARMUP, t)
            if len(b) < 900:
                continue
            try:
                lv = all_levels(b, t)
            except Exception:
                continue
            self.rows[t] = {k: [sd, round(p, 2)] for k, (sd, p) in lv.items()}
            done += 1
            if time.time() - t0 > budget:
                break
        if done:
            try:
                keep = {k: v for k, v in self.rows.items() if k >= t_end - 30 * 86400}
                self.cache.write_text(json.dumps(keep))
            except OSError:
                pass

    def since(self, t: int) -> list:
        return [[k, self.rows[k]] for k in sorted(self.rows) if k > t]


class Dashboard:
    def __init__(self, logs: dict[str, Path], key_id=None, key_path=None):
        self.k = KalshiClient(key_id, key_path)
        self.logs = logs
        self.prices: dict[int, float] = {}
        self.session: dict | None = None
        self.lock = threading.Lock()
        self.err = ""
        self._hours: dict[str, tuple] = {}   # hour -> (read time, points)
        self.ladders: dict[int, list[float]] = {}   # hour close -> Kalshi range band edges
        self._ladder_at: dict[int, float] = {}
        self.first_ts: int | None = None
        self.model = RangeModel()
        self.levels = LevelTracker(logs["breakdown"] / "prices", Path.home() / ".kalshi-dash-levels.json")
        self.quotes: dict = {}
        self.strikes: dict[int, float] = {}   # 15-minute session open -> Kalshi target (strike)
        self._strikes_at = 0.0
        self.q15: list[list] = []     # [t, yes_bid, yes_ask] for the current 15-minute contract
        self._quotes_at = 0.0

    def history(self, res: int = 5) -> dict:
        """Price history from the breakdown bot's hourly logs (closed hours are cached)."""
        now = time.time()
        pts = []
        for f in sorted((self.logs["breakdown"] / "prices").glob("*/*.csv.gz")):
            key = f"{f.parent.name}/{f.name[:2]}"
            # an hour read while the bot was still writing it is incomplete: keep re-reading it
            # until a read made at least 2 minutes after that hour ended
            if key not in self._hours or self._hours[key][0] < _hour_end(key) + 120:
                self._hours[key] = (now, read_price_hour(f, res))
            pts += self._hours[key][1]
        with self.lock:
            live = sorted(self.prices)
            pts += [[t, self.prices[t]] for t in live[::res] if not pts or t > pts[-1][0]]
        if pts:
            self.first_ts = pts[0][0]
        lad = [[c, e] for c, e in sorted(self.ladders.items())]
        ev = history_events(self.logs)
        have = {t: k for t, k in ev["sessions"]}
        have.update(self.strikes)  # Kalshi's own list wins; the bot's log fills any gap
        ev["sessions"] = [[t, have[t]] for t in sorted(have)]
        return {"prices": pts, "ladders": lad, **ev}

    def update_strikes(self, now: float):
        """Every 15-minute target since the history starts, straight from Kalshi (one call a minute)."""
        if now - self._strikes_at < 60:
            return
        self._strikes_at = now
        lo = int(self.first_ts or now - 86400)
        cursor = None
        try:
            for _ in range(10):
                j = self.k.get("/markets", series_ticker="KXBTC15M", min_close_ts=lo, max_close_ts=int(now) + 1800,
                               limit=1000, **({"cursor": cursor} if cursor else {}))
                for m in j.get("markets", []):
                    k = _f(m.get("floor_strike"))
                    if k:
                        close = int(datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).timestamp())
                        self.strikes[close - 900] = k
                cursor = j.get("cursor")
                if not cursor:
                    break
        except Exception as e:
            self.err = f"15-min targets: {e}"

    def fetch_ladder(self, close_ts: int) -> list[float]:
        """Edges of every Kalshi hourly range band (KXBTC) for the hour that closes at close_ts."""
        edges, cursor = set(), None
        for _ in range(10):
            j = self.k.get("/markets", event_ticker=event_ticker(close_ts), limit=200, **({"cursor": cursor} if cursor else {}))
            for m in j.get("markets", []):
                for key, add in (("floor_strike", 0.0), ("cap_strike", 0.01)):
                    v = m.get(key)
                    if v is not None:
                        edges.add(float(round(float(v) + add)))
            cursor = j.get("cursor")
            if not cursor:
                break
        return sorted(edges)

    def warm_up(self, now: float):
        """Load the last two hours of BRTI so the range model has its 100 minutes of history at once."""
        for h in range((int(now) - 2 * 3600) // 3600 * 3600, int(now) // 3600 * 3600 + 1, 3600):
            try:
                got = self.k.brti_hour(h)
                with self.lock:
                    self.prices.update(got)
            except Exception as e:
                log.warning("BRTI history %s failed: %s", h, e)

    def _price_at(self, t: int) -> float | None:
        for u in range(t, t - 30, -1):
            if u in self.prices:
                return self.prices[u]
        return None

    def model_above(self, now: float, strike, close_ts: int):
        """Range model's chance the 15-minute contract settles at or above its target."""
        try:
            if not strike:
                return None
            t = int(now)
            with self.lock:
                last = max(self.prices) if self.prices else None
                price = self._price_at(last) if last else None
                closes = [self._price_at(t - 1 - 60 * (100 - k)) for k in range(101)]
            if price is None or None in closes:
                return None
            v = self.model.vol(np.array(closes))
            if not v:
                return None
            left = max(1.0, (close_ts - now) / 60)
            return round(self.model.p_inside(price, float(strike), None, left, *v), 4)
        except Exception:
            return None

    def range_quotes(self, now: float) -> dict:
        """Kalshi's price and the range model's value for the bands nearest the price, this hour."""
        close = (int(now) // 3600 + 1) * 3600
        t = int(now)
        with self.lock:
            last = max(self.prices) if self.prices else None
            price = self._price_at(last) if last else None
            closes = [self._price_at(t - 1 - 60 * (100 - k)) for k in range(101)]
        if price is None:
            return {}
        left = max(1, (close - t) // 60)
        v = self.model.vol(np.array(closes)) if None not in closes else None
        markets = self.k.get("/markets", event_ticker=event_ticker(close), limit=1000).get("markets", [])
        markets = [m for m in markets if m.get("status") in ("active", "open")]
        markets = sorted(markets, key=lambda m: abs(band_mid(m) - price))[:N_BANDS]
        rows = []
        for m in sorted(markets, key=band_mid, reverse=True):
            lo, hi = _f(m.get("floor_strike")), _f(m.get("cap_strike"))
            if m.get("strike_type") == "greater":
                hi = None
            elif m.get("strike_type") == "less":
                lo = None
            yb, ya = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
            p = self.model.p_inside(price, lo, hi, left, *v) if v else None
            no_ask = round(1 - yb, 4) if yb else None
            rows.append({"ticker": m["ticker"], "lo": lo, "hi": hi, "yes_bid": yb, "yes_ask": ya, "no_ask": no_ask,
                         "model_yes": None if p is None else round(p, 4),
                         "yes_ok": bool(p is not None and left <= 30 and qualifies(p, ya)),
                         "no_ok": bool(p is not None and left <= 30 and qualifies(1 - p, no_ask))})
        return {"close": close, "mins_left": left, "price": price, "at": t, "bands": rows}

    def update_ladders(self, now: float):
        """One Kalshi request per call at most: fill missing hours, refresh the current one every 10 min."""
        cur = (int(now) // 3600 + 1) * 3600
        start = ((self.first_ts or int(now)) // 3600 + 1) * 3600
        todo = [c for c in range(start, cur + 1, 3600) if c not in self.ladders]
        if not todo and now - self._ladder_at.get(cur, 0) > 600:
            todo = [cur]
        if not todo:
            return
        c = todo[-1]
        try:
            self.ladders[c] = self.fetch_ladder(c)
            self._ladder_at[c] = now
        except Exception as e:
            self.err = f"range bands: {e}"

    def poll(self):
        now = time.time()
        try:
            got = self.k.brti_live()
            with self.lock:
                self.prices.update(got)
                for t in [t for t in self.prices if t < now - HOLD]:
                    del self.prices[t]
            self.err = ""
        except Exception as e:  # keep serving what we have
            self.err = f"price feed: {e}"
        self.update_ladders(now)
        self.update_strikes(now)
        s15 = self.session
        if s15:
            try:
                m = self.k.get(f"/markets/{s15['ticker']}")["market"]
                row = [int(now), _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars")),
                       self.model_above(now, s15.get("strike"), s15["close_ts"])]
                if self.q15 and self.q15[0][0] < s15["open_ts"]:
                    self.q15 = []
                self.q15.append(row)
            except Exception as e:
                self.err = f"15-min price: {e}"
        if now - self._quotes_at >= 2:
            self._quotes_at = now
            try:
                self.quotes = self.range_quotes(now)
            except Exception as e:
                self.err = f"range prices: {e}"
        s = self.session
        if s is None or now >= s["close_ts"]:
            try:
                self.session = self.k.open_market(now)
            except Exception as e:
                self.err = f"market list: {e}"

    def level_loop(self):
        while True:
            try:
                with self.lock:
                    live = dict(self.prices)
                self.levels.update(time.time(), live)
            except Exception as e:
                log.warning("levels: %s", e)
            time.sleep(15)

    def loop(self, every: float = 2.0):
        while True:
            t0 = time.time()
            self.poll()
            time.sleep(max(0.2, every - (time.time() - t0)))

    def data(self) -> dict:
        with self.lock:
            cut = time.time() - KEEP
            ts = sorted(t for t in self.prices if t >= cut)
            pts = [[t, self.prices[t]] for t in ts]
        L = self.logs
        s = self.session or {}
        bd = _rows(L["breakdown"] / "signals.csv")
        fp = _rows(L["fastpush"] / "signals.csv")
        fpr = L.get("fastpush_range") or Path("/nonexistent")
        hour_close = (int(time.time()) // 3600 + 1) * 3600
        return {
            "now": time.time(), "prices": pts, "error": self.err, "hour_close": hour_close,
            "range": {"open": range_positions(L["range"]),
                      "settled": settled_summary(_rows(L["range"] / "trades.csv"))},
            "session": s, "quotes": self.quotes, "q15": self.q15,
            "buys": session_buys(bd, s.get("ticker"), "Breakdown") + session_buys(fp, s.get("ticker"), "Fast push"),
            "pnl": sorted(pnl_points(_rows(L["range"] / "trades.csv"), "Range")
                          + pnl_points(_rows(L["breakdown"] / "trades.csv"), "Breakdown")
                          + pnl_points(_rows(L["fastpush"] / "trades.csv"), "Fast push")
                          + pnl_points(_rows(fpr / "trades.csv"), "FP range"),
                          key=lambda r: r["time_utc"]),
            "breakdown": settled_summary(_rows(L["breakdown"] / "trades.csv")),
            "fastpush": settled_summary(_rows(L["fastpush"] / "trades.csv")),
            "fastpush_range": settled_summary(_rows(fpr / "trades.csv")),
        }


def range5_events(d: Path) -> list:
    """The 5-account bot's trades in the main chart's range-trade shape, with the account number."""
    edges = {r["ticker"]: (_num(r.get("floor")), _num(r.get("cap"))) for r in _rows(d / "signals.csv")}
    out = []
    for r in _rows(d / "trades.csv"):
        lo, hi = edges.get(r.get("ticker"), (None, None))
        t = _ts(r["time_utc"])
        out.append({"t": t, "close": (t // 3600 + 1) * 3600, "side": r.get("side"), "lo": lo, "hi": hi,
                    "n": r.get("contracts"), "price": _num(r.get("avg_price")), "model": _num(r.get("model")),
                    "status": "won" if str(r.get("won")) == "1" else "lost", "pnl": _num(r.get("pnl")),
                    "cost": _num(r.get("cost")), "acct": int(r.get("account") or 0)})
    for p in _json(d / "open_positions.json", []):
        lo, hi = edges.get(p.get("ticker"), (None, None))
        t = _ts(p["time_utc"])
        out.append({"t": t, "close": p.get("close_ts") or (t // 3600 + 1) * 3600, "side": p.get("side"), "lo": lo,
                    "hi": hi, "n": p.get("contracts"), "price": _num(p.get("avg_price")), "model": _num(p.get("model")),
                    "status": "open", "pnl": None, "cost": _num(p.get("cost")), "acct": int(p.get("account") or 0)})
    return out


def _range5(self) -> dict:
    from .dashboard_range5 import range5_data
    with self.lock:
        cut = time.time() - KEEP
        ts = sorted(t for t in self.prices if t >= cut)
        pts = [[t, self.prices[t]] for t in ts[::5]]
    d = self.logs.get("range5")
    return range5_data(d, pts) if d else {"now": time.time(), "prices": pts, "status": {}, "open": [], "trades": [],
                                          "signals": 0, "unfilled": 0}


Dashboard.range5 = _range5


def make_handler(dash: Dashboard, token: str):
    from .dashboard_range5 import PAGE5
    page = PAGE.encode()
    page5 = PAGE5.encode()
    root = f"/{token}/"

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            # ?v=range5 shows the same chart with the 5-account bot's trades (embedded at the top of /range5/)
            r5 = self.path.endswith("?v=range5") and "range5" in dash.logs
            if r5:
                self.path = self.path[: -len("?v=range5")]
            if self.path == root:
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path == root + "history.json":
                h = dash.history()
                if r5:
                    h["range"], h["buys"] = range5_events(dash.logs["range5"]), []
                self._send(200, json.dumps(h).encode(), "application/json")
            elif self.path.startswith(root + "levels.json"):
                since = 0
                if "?since=" in self.path:
                    try:
                        since = int(self.path.split("?since=")[1])
                    except ValueError:
                        since = 0
                self._send(200, json.dumps(dash.levels.since(since)).encode(), "application/json")
            elif self.path == root + "data.json":
                self._send(200, json.dumps(dash.data()).encode(), "application/json")
            elif self.path == root + "range5/":
                self._send(200, page5, "text/html; charset=utf-8")
            elif self.path == root + "range5/data.json":
                self._send(200, json.dumps(dash.range5()).encode(), "application/json")
            elif self.path == root + "range5":
                self.send_response(301)
                self.send_header("Location", root + "range5/")
                self.end_headers()
            elif self.path == root[:-1]:
                self.send_response(301)
                self.send_header("Location", root)
                self.end_headers()
            else:
                self._send(404, b"not found", "text/plain")

    return H


def serve(port: int, token: str, logs: dict[str, Path]):
    dash = Dashboard(logs, os.environ.get("KALSHI_KEY_ID"), os.environ.get("KALSHI_KEY_FILE"))
    dash.warm_up(time.time())
    dash.poll()
    threading.Thread(target=dash.loop, daemon=True).start()
    threading.Thread(target=dash.level_loop, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", port), make_handler(dash, token))
    log.info("live chart on port %d", port)
    srv.serve_forever()


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bot Live Chart</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1b1f24;--mut:#66707a;--line:#1b1f24;--grid:#e6e8ec;--grid2:#c3c8cf;
--no:#c0392b;--yes:#1f8a4c;--strike:#2f6fb2;--target:#b8875a;--sup:#1f9a8a;--res:#c2477f;--ladder:#7fa3cf;--ladder-t:#1d4f91;--raw:#b9bec5;--buyzone:#14632f;--fp:#8e44ad;--fpr:#d35400;--win:#1f8a4c;--lose:#c0392b;--open:#ebb800;--win-bg:#e6f4ec;--lose-bg:#fbe9e7}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--card:#1b1e22;--fg:#e8eaed;--mut:#9aa3ad;
--line:#e8eaed;--grid:#262a30;--grid2:#3d434b;--no:#ef6b5b;--yes:#4cc27e;--strike:#6ea8e8;--target:#d2a87a;--sup:#3cc4b1;--res:#e57aa8;--ladder:#5b80ab;--ladder-t:#7aa7e0;--raw:#555c65;--buyzone:#3fbf6a;--fp:#c08be0;--fpr:#f39c5a;--win:#4cc27e;--lose:#ef6b5b;--open:#ffd84d;
--win-bg:#17301f;--lose-bg:#3a1c19;color-scheme:dark}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1300px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0}
.top{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 20px;margin-bottom:10px}
.px{font-size:26px;font-weight:700;font-variant-numeric:tabular-nums}
.mut{color:var(--mut)}
.bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:6px 0}
.bar button{border:1px solid var(--grid2);background:var(--card);color:var(--fg);padding:5px 11px;border-radius:6px;cursor:pointer;font:inherit}
.bar button:focus-visible{outline:2px solid var(--strike)}
.card{background:var(--card);border:1px solid var(--grid);border-radius:8px;padding:12px}
#sc{overflow-x:auto;overflow-y:hidden}
#stick{position:sticky;left:0}
canvas{display:block}
.lbl{font-weight:600;margin:4px 0 2px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin-top:12px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
td,th{padding:6px 8px;text-align:left;border-bottom:1px solid var(--grid);white-space:nowrap}
th{font-size:12px;color:var(--mut);font-weight:600}
.pill{padding:2px 8px;border-radius:10px;font-weight:600;font-size:12px}
.w{background:var(--win-bg);color:var(--win)}.l{background:var(--lose-bg);color:var(--lose)}
.scroll{overflow-x:auto}
.tscroll{max-height:380px;overflow:auto;overscroll-behavior:contain;border:1px solid var(--grid);border-radius:6px}
.tscroll th{position:sticky;top:0;background:var(--card);z-index:1;white-space:normal;vertical-align:bottom;line-height:1.2}
.tscroll td,.tscroll th{padding:4px 6px;font-size:12px}
.big{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}
.key{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--mut)}
.key i{display:inline-block;width:12px;height:10px;margin-right:4px;vertical-align:middle;border-radius:2px}
.navbtn{padding:5px 12px;border:1px solid var(--grid2);border-radius:6px;background:var(--card);color:var(--fg);text-decoration:none;font-weight:600;font-size:13px}.navbtn:hover{border-color:var(--ladder-t)}
body.r5 .nor5{display:none!important}body.r5 .wrap{padding:0}body.r5{background:var(--card)}
</style></head><body><div class="wrap">
<div class="top"><h1 class="nor5">Paper bots, live</h1><a class="nor5 navbtn" href="range5/">5-account range bot →</a><span class="px" id="px">…</span><span class="mut" id="pxraw"></span>
<span class="mut" id="clock"></span><span class="mut" id="err"></span></div>
<div class="bar"><button id="zin">Zoom in</button><button id="zout">Zoom out</button><button id="back">Back 1 hour</button><button id="live">Jump to now</button><button id="lvbtn">Hide support/resistance</button><button id="tlbtn">Hide trend lines</button>
<span class="mut" id="zinfo"></span></div>
<div class="card">
<div class="key r5only" style="display:none"><span>trade bands: light shade = <b>won</b>, dark shade = <b>lost</b>, dashed outline = open; color = account (1 blue · 2 orange · 3 green · 4 gold · 5 purple)</span></div>
<div class="key"><span class="nor5"><i style="background:var(--no);opacity:.35"></i>range NO band</span><span class="nor5"><i style="background:var(--yes);opacity:.35"></i>range YES band</span>
<span class="nor5">band outline: <b style="color:var(--win)">won</b> / <b style="color:var(--lose)">lost</b> / <b style="color:var(--open)">open</b></span><span class="nor5">markers: ● Range bot · ▲ Breakdown bot · ■ Fast-push bot · ◆ Fast-push range bot, colored <b style="color:var(--win)">won</b> / <b style="color:var(--lose)">lost</b> / <b style="color:var(--open)">open</b></span>
<span><i style="background:var(--ladder)"></i>Kalshi hourly range bands (every other band shaded)</span><span><i style="background:var(--sup)"></i>hard support</span><span><i style="background:var(--res)"></i>hard resistance (nearest hard level below and above BTC, recalculated every minute)</span><span><b style="color:var(--sup)">╌</b> rising support / <b style="color:var(--res)">╌</b> falling resistance trend lines (5-min swings), × = break</span><span><i style="background:var(--target)"></i>15-min target (strike) for each contract: above at the close = YES wins</span><span>thin vertical lines = 15-min contract closes</span></div>
<div id="sc"><div id="inner"><div id="stick">
<div class="lbl">Bitcoin: 60-second BRTI average (black, what Kalshi settles on), BRTI each second (light gray), and every trade</div><canvas id="c"></canvas>
<div class="nor5"><div class="lbl">Profit and loss, after fees <span class="mut" id="pl-legend" style="font-weight:400"></span></div><canvas id="pl"></canvas></div>
</div></div></div>
<div style="display:flex;align-items:center;gap:10px;margin-top:6px"><span class="mut" id="slL" style="font-size:12px;white-space:nowrap"></span>
<input type="range" id="slider" min="0" max="1000" value="1000" style="flex:1;accent-color:var(--ladder-t);height:22px">
<span class="mut" id="slR" style="font-size:12px;white-space:nowrap">now</span></div>
<div class="mut" style="font-size:12px;margin-top:2px">Drag the slider (or scroll sideways) to move through the full history. Both graphs move together.</div>
</div>
<div class="nor5">
<div class="card" style="margin-top:12px"><b>All trades</b> <span class="mut" id="tinfo"></span>
<div class="key" id="wl" style="margin:6px 0 8px;font-size:13px;color:var(--fg)"></div>
<div class="tscroll"><table id="trades"></table></div></div>
<div class="card scroll" style="margin-top:12px"><b>Open range positions</b><table id="pos"></table></div>
<div class="card scroll" style="margin-top:12px"><b>15-minute contract now</b> <span class="mut" id="q15info"></span>
<div style="display:flex;flex-wrap:wrap;gap:8px 28px;align-items:center;margin:8px 0"><div><div class="mut">YES bid / ask</div><div class="big" id="q15y"></div></div><div><div class="mut">NO bid / ask</div><div class="big" id="q15n"></div></div><div><div class="mut">Model YES / NO</div><div class="big" id="q15m"></div></div>
<div style="flex:1;min-width:260px"><canvas id="q15c" style="width:100%;height:130px"></canvas>
<div class="mut" style="font-size:11px"><b style="color:var(--yes)">━</b> Kalshi YES ask &nbsp; <b style="color:var(--fg)">┅</b> model YES &nbsp; faint line along 0¢ = model minus Kalshi, colored <span id="q15scale"></span></div></div></div>
<b>15-minute bots, this session</b><table id="pos15"></table></div>
<div class="grid">
<div class="card"><div class="mut">Range bot, settled</div><div class="big" id="sr"></div></div>
<div class="card"><div class="mut">Breakdown bot, settled</div><div class="big" id="sb"></div></div>
<div class="card"><div class="mut">Fast-push bot, settled</div><div class="big" id="sf"></div></div>
<div class="card"><div class="mut">Fast-push range bot, settled</div><div class="big" id="sfr"></div></div>
</div>
<p class="mut">BRTI is the index Kalshi settles on; settlement is its average over the last 60 seconds. Paper trades only. Updates every 2 seconds.</p>
</div></div>
<script>
let D=null,H=null,P=[];          // live data, history, merged prices [[t,p]]
const R5=location.search.includes('v=range5'),QS=R5?'?v=range5':'';   // chart-only view for the 5-account bot page
if(R5){document.body.classList.add('r5');document.querySelector('.r5only').style.display=''}
// 5-account view: each account has its own color; a winning trade is a light shade of it, a losing trade a dark shade
const ACOL={1:[42,120,214],2:[235,104,52],3:[27,175,122],4:[201,133,0],5:[98,80,214]};
function acol(r,kind){const c=ACOL[r.acct]||[120,120,120],m=kind==='light'?c.map(v=>Math.round(v+(255-v)*.55)):kind==='dark'?c.map(v=>Math.round(v*.55)):c;return `rgb(${m.join(',')})`}
const ZOOMS=[2,4,8,16,32,64,128,256,512];let zi=6;  // pixels per 15 minutes
let follow=true;
const $=id=>document.getElementById(id);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
// all times shown in US Eastern time (EST/EDT handled by the browser)
const TZ='America/New_York';
const fHM=new Intl.DateTimeFormat('en-US',{timeZone:TZ,hour:'numeric',minute:'2-digit',hour12:true});
const fHMS=new Intl.DateTimeFormat('en-US',{timeZone:TZ,hour:'numeric',minute:'2-digit',second:'2-digit',hour12:true});
const fDay=new Intl.DateTimeFormat('en-US',{timeZone:TZ,weekday:'short',month:'short',day:'numeric'});
const fH=new Intl.DateTimeFormat('en-US',{timeZone:TZ,hour:'numeric',minute:'numeric',hour12:false});
const hm=t=>fHM.format(new Date(t*1000)).replace(' AM','a').replace(' PM','p');
const hms=t=>fHMS.format(new Date(t*1000));
const etMidnight=t=>{const [h,m]=fH.format(new Date(t*1000)).split(':').map(Number);return (h%24)===0&&m===0};
const money=x=>(x<0?'-$':'+$')+Math.abs(x).toFixed(2);
const usd=x=>'$'+x.toLocaleString('en-US',{maximumFractionDigits:0});
const tsOf=s=>Date.parse(s.replace(' ','T')+'Z')/1000;
const sc=$('sc');
function pps(){return ZOOMS[zi]/900}
function range(){const first=Math.min(P.length?P[0][0]:D.now,...(H?H.range.map(r=>r.close-3600):[]));
 return [Math.floor(first/900)*900,Math.ceil(D.now/900)*900+300]}
function sum(id,s){$(id).textContent=s.trades?`${s.wins}/${s.trades} won, ${money(s.pnl)}`:'no trades yet'}
// trailing 60-second average of BRTI at each price point (Kalshi settles on the average of the 60 seconds before the close)
function avg60(tL,tR){let a=0;while(a<P.length&&P[a][0]<tL-120)a++;const out=[];let j=a,sum=0,n=0;
 for(let i=a;i<P.length&&P[i][0]<=tR+60;i++){sum+=P[i][1];n++;while(P[j][0]<=P[i][0]-60){sum-=P[j][1];n--;j++}
  if(P[i][0]>=tL-60)out.push([P[i][0],P[i][0]-P[j][0]>=40||i-j>=50?sum/n:null])}return out}
function merge(pts){if(!pts.length)return;const last=P.length?P[P.length-1][0]:0;for(const p of pts)if(p[0]>last)P.push(p)}
async function loadHistory(){try{const r=await fetch('history.json'+QS,{cache:'no-store'});H=await r.json();
 const live=P.filter(p=>p[0]>(H.prices.length?H.prices[H.prices.length-1][0]:0));P=H.prices.concat(live);layout()}catch(e){}}
let LVR=[];
async function loadLevels(){try{const since=LVR.length?LVR[LVR.length-1][0]:0;const r=await fetch('levels.json?since='+since,{cache:'no-store'});
 const add=await r.json();if(add.length){LVR=LVR.concat(add);draw()}}catch(e){}}
async function tick(){try{const r=await fetch('data.json',{cache:'no-store'});D=await r.json();merge(D.prices);
 if(!H)await loadHistory();layout()}catch(e){$('err').textContent='connection lost, retrying'}}
function layout(){
 if(!D)return;const [T0,T1]=range();const w=sc.clientWidth;
 const total=Math.max(w,(T1-T0)*pps()+90);$('inner').style.width=total+'px';$('stick').style.width=w+'px';
 if(follow)sc.scrollLeft=total;syncSlider();draw()}
sc.addEventListener('scroll',()=>{follow=sc.scrollLeft+sc.clientWidth>=sc.scrollWidth-30;syncSlider();draw()});
// time slider: 0 = start of the history, 1000 = now
const sl=$('slider');let slDrag=false;
function syncSlider(){if(slDrag)return;const m=sc.scrollWidth-sc.clientWidth;sl.value=m>0?Math.round(sc.scrollLeft/m*1000):1000}
sl.addEventListener('input',()=>{slDrag=true;const m=sc.scrollWidth-sc.clientWidth;sc.scrollLeft=sl.value/1000*m;follow=sl.value>=999;draw()});
sl.addEventListener('change',()=>{slDrag=false});
function zoom(d){if(!D)return;const [T0,T1]=range();
 if(d<0&&(T1-T0)*pps()+90<=sc.clientWidth)return;  // the whole history is already on screenconst mid=T0+(sc.scrollLeft+sc.clientWidth/2)/pps();
 zi=Math.max(0,Math.min(ZOOMS.length-1,zi+d));layout();if(!follow)sc.scrollLeft=(mid-T0)*pps()-sc.clientWidth/2;draw()}
$('zin').onclick=()=>zoom(1);$('zout').onclick=()=>zoom(-1);
$('back').onclick=()=>{follow=false;sc.scrollLeft-=3600*pps()};
$('live').onclick=()=>{follow=true;layout()};
let showLv=true;let showTl=true;$('tlbtn').onclick=()=>{showTl=!showTl;$('tlbtn').textContent=(showTl?'Hide':'Show')+' trend lines';draw()};
// Trend lines, same definition as the trend line research (research/trend_lines/tl.py), 5-minute swings:
// swing low = lowest BRTI within 5 min either side (known 5 min later); support = line from an earlier, lower swing low
// (up to 10 back, within 4 h) to the latest one, no print more than $15 below it in between, most touches wins
// (longest on ties). Resistance mirrors with falling swing highs. Drawn until the first print $5 through it
// (marked ×) or until the next swing point takes over.
const TLN=300,TLTOL=15,TLB=5,TLSPAN=4*3600;let TL=[],TLkey='';
function trendLines(){if(P.length<3)return TL;const key=P[0][0]+':'+Math.floor(P[P.length-1][0]/30);  // recompute at most every 30 sif(key===TLkey)return TL;TLkey=key;
 const n=P.length,t=P.map(p=>p[0]),v=P.map(p=>p[1]);
 // centered rolling min/max over +/-TLN seconds (two-pointer deques)
 const swing=(isLow)=>{const out=[];const dq=[];let r=0,l=0;
  for(let i=0;i<n;i++){while(r<n&&t[r]<=t[i]+TLN){while(dq.length&&(isLow?v[dq[dq.length-1]]>=v[r]:v[dq[dq.length-1]]<=v[r]))dq.pop();dq.push(r);r++}
   while(dq.length&&t[dq[0]]<t[i]-TLN)dq.shift();
   if(t[i]-t[0]>=TLN&&t[n-1]-t[i]>=TLN&&dq.length&&dq[0]===i&&(!out.length||t[i]-t[out[out.length-1]]>TLN))out.push(i)}
  return out};
 const lines=[];
 for(const [side,piv] of [['sup',swing(true)],['res',swing(false)]]){const sg=side==='sup'?1:-1;
  for(let k=1;k<piv.length;k++){const i2=piv[k],p2=v[i2];let best=null;
   for(let j=k-1;j>=Math.max(k-10,0);j--){const i1=piv[j],p1=v[i1];if(t[i2]-t[i1]>TLSPAN)break;if(sg*(p2-p1)<=0)continue;
    const sl=(p2-p1)/(t[i2]-t[i1]);let ok=true;for(let q=i1;q<=i2;q++)if(sg*(v[q]-(p2+sl*(t[q]-t[i2])))<-TLTOL){ok=false;break}
    if(!ok)continue;let tc=2;for(let m=j+1;m<k;m++)if(Math.abs(v[piv[m]]-(p2+sl*(t[piv[m]]-t[i2])))<=TLTOL)tc++;
    if(!best||tc>=best.tc)best={tc,i1,p1,sl}}
   if(!best)continue;
   // active from now until the next swing point of this side is confirmed; ends early at a break
   const until=k+1<piv.length?t[piv[k+1]]+TLN:t[n-1];let end=until,brk=null;
   for(let q=i2;q<n&&t[q]<=until;q++)if(sg*(v[q]-(p2+best.sl*(t[q]-t[i2])))<-TLB){end=t[q];brk=[t[q],v[q]];break}
   lines.push({side,t0:t[best.i1],v0:best.p1,sl:best.sl,t1:end,brk,tc:best.tc})}}
 TL=lines;return TL}$('lvbtn').onclick=()=>{showLv=!showLv;$('lvbtn').textContent=(showLv?'Hide':'Show')+' support/resistance';draw()};
addEventListener('resize',layout);
function setup(cv,h){const dpr=devicePixelRatio||1,w=$('stick').clientWidth;cv.style.width=w+'px';cv.style.height=h+'px';
 cv.width=w*dpr;cv.height=h*dpr;const g=cv.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,w,h);return [g,w]}
function timegrid(g,X,tL,tR,T,B,h,labels){
 // grid and labels follow Eastern time; zoomed far out, fewer lines and labels
 const z=ZOOMS[zi],gl=z>=16?900:z>=8?3600:z>=4?3*3600:6*3600,lb=z>=64?900:z>=32?3600:z>=16?3*3600:z>=8?6*3600:z>=4?12*3600:86400;
 const etH=t=>{const [hh,mm]=fH.format(new Date(t*1000)).split(':').map(Number);return (hh%24)*3600+mm*60};
 for(let t=Math.floor(tL/900)*900;t<=tR;t+=900){const e=etH(t);if(e%gl&&gl>900||(gl===900?false:t%3600))continue;
  const hr=t%3600===0,day=hr&&e===0;
  g.strokeStyle=css(hr?'--grid2':'--grid');g.lineWidth=1;g.beginPath();g.moveTo(X(t),T);g.lineTo(X(t),h-B);g.stroke();
  if(labels&&(e%lb===0||day)){g.fillStyle=css(day?'--fg':'--mut');g.font=(day||hr?'600 ':'')+'11px system-ui';
   const d=new Date(t*1000);g.fillText(day?fDay.format(d):hm(t),X(t)-14,h-6)}}}
// right-hand gutter (same on both charts so their time axes line up): the current hour's band labels live here, clear of the price line
const GUT=110;
function draw(){
 if(!D)return;
 const last=P.length?P[P.length-1]:null;if(last){const a=avg60(last[0]-1,last[0]).at(-1);$('px').textContent=a&&a[1]!=null?usd(a[1]):usd(last[1]);$('pxraw').textContent=`60-s average · BRTI now ${usd(last[1])}`}
 $('err').textContent=D.error||'';
 $('clock').textContent=`${last?hms(last[0]):''} ET · 15-min closes ${hm(D.session&&D.session.close_ts||0)} · hour closes ${hm(D.hour_close)}`;
 $('zinfo').textContent=`${Math.round(sc.clientWidth/pps()/3600*10)/10} hours on screen`;
 {const [a]=range();const d=new Date(a*1000);$('slL').textContent=fDay.format(d)+' '+hm(a)}
 sum('sr',D.range.settled);sum('sb',D.breakdown);sum('sf',D.fastpush);sum('sfr',D.fastpush_range||{trades:0,wins:0,pnl:0});
 const [T0,T1]=range(),fits=(T1-T0)*pps()+90<=sc.clientWidth;
 // when the whole history fits on screen, keep "now" at the right edge (blank before the history starts, not after now)
 const tL=fits?T1+90/pps()-sc.clientWidth/pps():T0+sc.scrollLeft/pps(),tR=tL+sc.clientWidth/pps();
 // ---------------- price chart
 const [g,w]=setup($('c'),430),h=430,L=66,R=GUT,T=8,B=22;const gut=[];
 const X=t=>L+(t-tL)/(tR-tL)*(w-L-R);
 const vis=P.filter(p=>p[0]>=tL-60&&p[0]<=tR+60);
 const ses=(H?H.sessions:[]).concat(D.session&&D.session.strike?[[D.session.open_ts,D.session.strike]]:[]).filter(s=>s[0]+900>=tL&&s[0]<=tR);
 const rng=(H?H.range:[]).filter(r=>r.close>=tL&&r.close-3600<=tR&&r.lo!=null);
 let ys=vis.map(p=>p[1]);if(!ys.length&&last)ys=[last[1]];
 for(const s of ses)ys.push(s[1]);for(const r of rng)ys.push(r.lo,r.hi);
 let lo=Math.min(...ys),hi=Math.max(...ys);const pad=(hi-lo)*0.06+5;lo-=pad;hi+=pad;
 const Y=v=>T+(hi-v)/(hi-lo)*(h-T-B);
 timegrid(g,X,tL,tR,T,B,h,true);
 g.font='11px system-ui';
 const step=[5,10,20,25,50,100,200,250,500,1000,2000,5000].find(s=>(hi-lo)/s<=9)||10000;
 for(let v=Math.ceil(lo/step)*step;v<=hi;v+=step){g.strokeStyle=css('--mut');g.beginPath();g.moveTo(L-5,Y(v));g.lineTo(L,Y(v));g.stroke();g.fillStyle=css('--mut');g.fillText(usd(v),4,Y(v)+4)}
 g.save();g.beginPath();g.rect(L,0,w-L-R,h);g.clip();
 const st={won:'--win',lost:'--lose',open:'--open'};
 // Kalshi's hourly range bands (whether or not there is a trade): alternate shading per band, edge lines, price labels
 for(const [c,edges] of (H&&H.ladders||[])){if(c<tL||c-3600>tR)continue;const x0=X(c-3600),x1=X(c);
  const gaps=edges.slice(1).map((e,k)=>e-edges[k]).sort((a,b)=>a-b),med=gaps[gaps.length>>1]||100;
  for(let k=0;k+1<edges.length;k++){const a=edges[k],b=edges[k+1];if(b<lo||a>hi||b-a>1.5*med)continue;
   if(k%2===0){g.fillStyle=css('--ladder');g.globalAlpha=.13;g.fillRect(x0,Y(b),x1-x0,Y(a)-Y(b));g.globalAlpha=1}
   const q=bandQ(c,a,b),t=`${usd(a)}–${usd(b)}`+(q?`  K ${q.yes_ask!=null?Math.round(q.yes_ask*100)+'¢':'–'} · M ${q.model_yes!=null?Math.round(q.model_yes*100)+'¢':'–'}`:'');
   if(c>D.now||x1>w-R-2){if(Y(a)-Y(b)>=11)gut.push([`${usd(a)}–${usd(b)}`,q?`K ${q.yes_ask!=null?Math.round(q.yes_ask*100)+'¢':'–'} · M ${q.model_yes!=null?Math.round(q.model_yes*100)+'¢':'–'}`:'',Y(b),Y(a),k%2===0])}  // hour runs off the right edge: label it in the gutter
   else if(ZOOMS[zi]>=64&&Y(a)-Y(b)>=13&&x1-x0>120){g.fillStyle=css('--ladder-t');g.font='10px system-ui';
    const tx=x1-g.measureText(t).width-6;if(tx>=Math.max(x0,L)+4)g.fillText(t,tx,(Y(a)+Y(b))/2+3.5)}}
  g.strokeStyle=css('--ladder');g.globalAlpha=.8;g.lineWidth=1;
  for(const e of edges){if(e<lo||e>hi)continue;const y=Math.round(Y(e))+.5;g.beginPath();g.moveTo(x0,y);g.lineTo(x1,y);g.stroke()}
  g.globalAlpha=1}

 for(const r of rng){const x0=X(r.close-3600),x1=X(r.close);
  if(R5){g.fillStyle=acol(r,r.status==='won'?'light':r.status==='lost'?'dark':'');g.globalAlpha=r.status==='won'?.45:r.status==='lost'?.6:.12;
   g.fillRect(x0,Y(r.hi),x1-x0,Y(r.lo)-Y(r.hi));g.globalAlpha=1;g.strokeStyle=acol(r,r.status==='lost'?'dark':'');g.lineWidth=1.5;
   if(r.status==='open')g.setLineDash([4,3]);g.strokeRect(x0,Y(r.hi),x1-x0,Y(r.lo)-Y(r.hi));g.setLineDash([]);continue}
  const c=css(r.side==='NO'?'--no':'--yes');
  g.fillStyle=c;g.globalAlpha=.10;g.fillRect(x0,Y(r.hi),x1-x0,Y(r.lo)-Y(r.hi));g.globalAlpha=1;
  g.strokeStyle=css(st[r.status]);g.lineWidth=1.5;g.strokeRect(x0,Y(r.hi),x1-x0,Y(r.lo)-Y(r.hi));
}
 g.strokeStyle=css('--target');g.lineWidth=2.2;
 for(const s of ses){g.beginPath();g.moveTo(X(s[0]),Y(s[1]));g.lineTo(X(s[0]+900),Y(s[1]));g.stroke()}
 if(showLv&&LVR.length){g.lineWidth=1.3;g.globalAlpha=.8;
  // each level kind is a step line: one value per minute, joined while it stays the same
  let a=0,b=LVR.length;while(a<b){const m=(a+b)>>1;LVR[m][0]<tL-60?a=m+1:b=m}
  const open={};
  const flush=(k,e)=>{const o=open[k];if(!o)return;g.strokeStyle=css(o.side==='no'?'--sup':'--res');g.beginPath();g.moveTo(X(o.t0),Y(o.v));g.lineTo(X(e),Y(o.v));g.stroke();delete open[k]};
  for(let i=a;i<LVR.length&&LVR[i][0]<=tR;i++){const [t,lv]=LVR[i];
   for(const k in open){const n=lv[k];if(!n||n[1]!==open[k].v||n[0]!==open[k].side)flush(k,t)}
   for(const k in lv){if(!k.startsWith('hard'))continue;const [sd,v]=lv[k];if(v<lo||v>hi){if(open[k])flush(k,t);continue}if(!open[k])open[k]={t0:t,v,side:sd}}}
  const tend=Math.min(tR,(LVR[LVR.length-1][0])+60);for(const k in open)flush(k,tend);
  g.globalAlpha=1;g.lineWidth=2.2;g.strokeStyle=css('--target')}
 
 if(showTl){g.lineWidth=1.4;g.setLineDash([6,4]);
  for(const l of trendLines()){if(l.t1<tL||l.t0>tR)continue;g.strokeStyle=css(l.side==='sup'?'--sup':'--res');g.globalAlpha=.9;
   g.beginPath();g.moveTo(X(l.t0),Y(l.v0));g.lineTo(X(l.t1),Y(l.v0+l.sl*(l.t1-l.t0)));g.stroke();
   if(l.brk){g.setLineDash([]);const bx=X(l.brk[0]),by=Y(l.v0+l.sl*(l.brk[0]-l.t0));g.beginPath();g.moveTo(bx-4,by-4);g.lineTo(bx+4,by+4);g.moveTo(bx+4,by-4);g.lineTo(bx-4,by+4);g.stroke();g.setLineDash([6,4])}}
  g.setLineDash([]);g.globalAlpha=1}
 // BRTI itself in light gray, the 60-second average (what Kalshi settles on) as the main line
 g.strokeStyle=css('--raw');g.lineWidth=1;g.beginPath();vis.forEach((p,i)=>{const x=X(p[0]),y=Y(p[1]);i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke();
 const av=avg60(tL,tR);g.strokeStyle=css('--line');g.lineWidth=1.6;g.beginPath();let pen=false;
 for(const [t,v] of av){if(v==null){pen=false;continue}const x=X(t),y=Y(v);pen?g.lineTo(x,y):g.moveTo(x,y);pen=true}g.stroke();
 const at=t=>{let a=0,b=P.length-1;while(a<b){const m=(a+b)>>1;P[m][0]<t?a=m+1:b=m}return P[a]&&Math.abs(P[a][0]-t)<=60?P[a][1]:null};
 const placed=[],colMin=new Float32Array(Math.ceil(w)+2).fill(1e9),colMax=new Float32Array(Math.ceil(w)+2).fill(-1e9);
 for(let i=0;i<vis.length;i++){const x0=X(vis[i][0]),y0=Y(vis[i][1]),p1=vis[Math.min(i+1,vis.length-1)],x1=X(p1[0]),y1=Y(p1[1]);
  for(let c=Math.max(0,Math.floor(Math.min(x0,x1)));c<=Math.min(colMin.length-1,Math.ceil(Math.max(x0,x1)));c++){
   colMin[c]=Math.min(colMin[c],y0,y1);colMax[c]=Math.max(colMax[c],y0,y1)}}
 const over=(x,yy,tw,th)=>{let n=0;const a=Math.max(0,Math.floor(x)-2),b=Math.min(colMin.length-1,Math.ceil(x+tw)+2);
  for(let c=a;c<=b;c++)if(colMin[c]<=yy+th+3&&colMax[c]>=yy-3)n++;
  if(placed.some(p=>x<p[0]+p[2]&&x+tw>p[0]&&yy<p[1]+p[3]&&yy+th>p[1]))n+=1e6;return n};
 const drawBox=(x,yy,tw,th,txt,col)=>{placed.push([x,yy,tw,th]);g.fillStyle=css('--card');g.globalAlpha=.9;g.fillRect(x,yy,tw,th);g.globalAlpha=1;
  g.strokeStyle=col;g.lineWidth=1;g.strokeRect(x,yy,tw,th);g.fillStyle=col;g.fillText(txt,x+4,yy+11);return [x,yy+th/2,tw]};
 // free-floating label (15-minute buys): nearest spot off the price line
 const label=(x,y,txt,col)=>{g.font='600 10.5px system-ui';const tw=g.measureText(txt).width+8,th=15;
  if(x+tw>w-R)x=Math.max(L,x-tw-28);
  let yy=y-th/2;for(let k=1;k<80&&(yy<T||yy+th>h-B||over(x,yy,tw,th));k++)yy=y-th/2+(k%2?1:-1)*Math.ceil(k/2)*8;
  return drawBox(x,yy,tw,th,txt,col)};
 // range label: stays inside its band box, off the price line where it can
 const boxLabel=(dx,texts,col,bx0,bx1,by0,by1)=>{g.font='600 10.5px system-ui';const th=15;
  bx0=Math.max(bx0,L)+3;bx1=Math.min(bx1,w-R)-3;
  let txt=texts.find(t=>g.measureText(t).width+8<=bx1-bx0)||texts[texts.length-1];const tw=g.measureText(txt).width+8;
  const ymin=by0+1,ymax=Math.max(ymin,by1-th-1),xmax=Math.max(bx0,bx1-tw),cx=(ymin+ymax)/2;
  const xs=[Math.min(Math.max(dx+14,bx0),xmax),Math.min(Math.max(dx-tw-14,bx0),xmax),bx0,xmax];for(let q=bx0;q<xmax;q+=30)xs.push(q);
  let best=null;
  for(const x of xs){for(let k=0;k<=Math.ceil((ymax-ymin)/2)+1;k++){const yy=Math.min(ymax,Math.max(ymin,cx+(k%2?1:-1)*Math.ceil(k/2)*2-th/2+th/2));
    const sc=over(x,yy,tw,th);if(!best||sc<best[2])best=[x,yy,sc];if(sc===0)break}if(best[2]===0)break}
  return drawBox(best[0],best[1],tw,th,txt,col)};
 const full=ZOOMS[zi]>=128,short0=ZOOMS[zi]>=32;
 for(const r of rng){const y=at(r.t)??(r.lo+r.hi)/2;const x=X(r.t),yd=Y(y),col=R5?acol(r,'dark'):css(st[r.status]),mid=Y((r.lo+r.hi)/2),short=short0&&x>=L&&x<=w-R;
  const res=r.status==='open'?'OPEN':r.status.toUpperCase()+' '+money(r.pnl),ac=R5?`#${r.acct} `:'';
  const fpr=r.bot==='FP range',md=fpr?`, push ${r.level||''}`:`, model ${r.model?(r.model*100).toFixed(1):'?'}¢`;
  const texts=[`${ac}${hm(r.t)} ${fpr?'FP range ':''}${r.side} ${usd(r.lo)}–${usd(r.hi+0.01)} x${r.n} @${Math.round(r.price*100)}¢${md} → ${res}`,
   `${ac}${hm(r.t)} ${fpr?'FP range ':''}${r.side} x${r.n} @${Math.round(r.price*100)}¢${md} → ${res}`,`${ac}${fpr?'FPR ':''}${r.side} @${Math.round(r.price*100)}¢ ${res}`];
  if(!full)texts.splice(0,2);
  if(short){const [lx,ly,tw]=boxLabel(x,texts,col,X(r.close-3600),X(r.close),Y(r.hi),Y(r.lo));const ex=lx>x?lx:(lx+tw<x?lx+tw:x);
   g.strokeStyle=col;g.lineWidth=1.2;g.setLineDash([3,3]);g.beginPath();g.moveTo(x,yd);g.lineTo(x,ly);g.lineTo(ex,ly);g.stroke()}
  g.setLineDash([]);marker(g,r.bot||'Range',x,yd,R5?acol(r,r.status==='won'?'light':r.status==='lost'?'dark':''):col)}
 for(const b of (H?H.buys:[]).filter(b=>b.t>=tL&&b.t<=tR)){const y=at(b.t);if(y==null)continue;const x=X(b.t),yy=Y(y),up=b.side==='YES',short=short0&&x>=L&&x<=w-R;
  marker(g,b.bot,x,yy,css(st[b.status]));
  if(short){const col=css(st[b.status]),ty=yy+(up?-28:28);
   const [lx,ly,tw]=label(x+12,ty,`${hm(b.t)} ${b.bot} ${b.side} @${b.ask?Math.round(b.ask*100)+'¢':''} ${b.level||''} → ${b.status==='open'?'OPEN':b.status.toUpperCase()+(b.pnl!=null?' '+money(b.pnl):'')}`,col);g.strokeStyle=col;g.setLineDash([3,3]);g.beginPath();g.moveTo(x,yy);g.lineTo(lx>x?lx:lx+tw,ly);g.stroke();g.setLineDash([])}}
 // live 15-min contract: Kalshi YES price, model and difference, in a small tag just right of the live price, by its target line
 const S=D.session,Q=(D.q15||[]).at(-1);
 if(S&&S.strike&&Q&&last&&S.open_ts<=tR&&S.close_ts>=tL){const yb=Q[1],ya=Q[2],m=Q[3];
  // Y-kp/mp/d: Kalshi YES ask / model YES / model minus Kalshi, all in cents
  const c1=v=>v==null?'–':String(Math.round(v*1000)/10),dd=ya!=null&&m!=null?Math.round((m-ya)*1000)/10:null;
  const na=yb!=null?1-yb:null,mn=m!=null?1-m:null,dn=na!=null&&mn!=null?Math.round((mn-na)*1000)/10:null,sg=d=>d==null?'–':(d>0?'+':'')+d;
  const l1=`Y-${c1(ya)}/${c1(m)}/${sg(dd)}`,l2=`N-${c1(na)}/${c1(mn)}/${sg(dn)}`;  // NO ask = 1 - YES bid
  g.font='9px system-ui';const tw=Math.max(g.measureText(l1).width,g.measureText(l2).width)+6,th=22,ys=Y(S.strike);
  let x=Math.min(X(last[0])+8,w-R-4-tw);x=Math.max(L+4,x);
  const pv=vis.filter(p=>X(p[0])>=x-4&&X(p[0])<=x+tw+4).map(p=>Y(p[1])).concat([Y(last[1])]);
  const clear=y0=>!pv.some(py=>py>=y0-4&&py<=y0+th+4)&&y0>=T&&y0+th<=h-B;
  const cands=[ys-th-3,ys+3,ys-th-18,ys+18,ys-th-36,ys+36];const y=cands.find(clear)??Math.max(T,Math.min(h-B-th,cands[0]));
  g.fillStyle=css('--card')||'#fff';g.fillRect(x,y,tw,th);
  const lft=S.close_ts-last[0],buy=lft>=0&&lft<180;  // green from the start of the 15-min buy rule (3 min left) to the close
  g.strokeStyle=css(buy?'--buyzone':'--target');g.lineWidth=buy?1.5:1;g.strokeRect(x+.5,y+.5,tw-1,th-1);
  g.fillStyle=css(buy?'--buyzone':'--fg');g.fillText(l1,x+3,y+9.5);g.fillText(l2,x+3,y+19)}
 if(last&&last[0]>=tL&&last[0]<=tR){const la=av.length?av[av.length-1][1]:null;g.fillStyle=css('--fg');g.beginPath();g.arc(X(last[0]),Y(la??last[1]),4,0,7);g.fill()}
 g.restore();
 // gutter: current hour's band labels, outside the plot so the price line never covers them
 g.save();g.beginPath();g.rect(w-R,T,R,h-T-B);g.clip();g.font='10px system-ui';
 for(const [t1,t2,yt,yb,sh] of gut){if(sh){g.fillStyle=css('--ladder');g.globalAlpha=.13;g.fillRect(w-R,yt,R,yb-yt);g.globalAlpha=1}
  const ym=(yt+yb)/2,two=t2&&yb-yt>=24;if(ym-12<T||ym+13>h-B)continue;g.fillStyle=css('--ladder-t');g.fillText(t1,w-R+5,two?ym-1.5:ym+3.5);if(two)g.fillText(t2,w-R+5,ym+10.5)}
 g.restore();
 plchart(tL,tR);tables(last?last[1]:null);
}
// one shape per bot: Range circle, Breakdown triangle, Fast push square, Fast push on range bands diamond
function marker(g,bot,x,y,col){g.fillStyle=col;g.strokeStyle=css('--card');g.lineWidth=1.5;g.beginPath();
 if(bot==='Breakdown'){g.moveTo(x,y-7);g.lineTo(x-6.5,y+5);g.lineTo(x+6.5,y+5);g.closePath()}
 else if(bot==='Fast push'){g.rect(x-5,y-5,10,10)}
 else if(bot==='FP range'){g.moveTo(x,y-7);g.lineTo(x+7,y);g.lineTo(x,y+7);g.lineTo(x-7,y);g.closePath()}
 else g.arc(x,y,5,0,7);g.fill();g.stroke()}
const BOTS=[['Total','--line'],['Range','--yes'],['Breakdown','--no'],['Fast push','--fp'],['FP range','--fpr']];
function plchart(tL,tR){
 const [g,w]=setup($('pl'),240),h=240,L=66,R=GUT,T=8,B=22;
 const X=t=>L+(t-tL)/(tR-tL)*(w-L-R);
 const pts=D.pnl.map(r=>({...r,t:tsOf(r.time_utc)}));
 const [T0]=range();const series={},run={};for(const [b] of BOTS){series[b]=[[T0,0]];run[b]=0}
 for(const p of pts){run[p.bot]+=p.pnl;run.Total+=p.pnl;series[p.bot].push([p.t,run[p.bot]]);series.Total.push([p.t,run.Total])}
 for(const [b] of BOTS){const s=series[b];s.push([D.now,s[s.length-1][1]])}
 const valAt=(s,t)=>{let v=0;for(const p of s){if(p[0]>t)break;v=p[1]}return v};
 let vals=[0];for(const [b] of BOTS){const s=series[b];vals.push(valAt(s,tL),valAt(s,tR));for(const p of s)if(p[0]>=tL&&p[0]<=tR)vals.push(p[1])}
 let lo=Math.min(...vals),hi=Math.max(...vals);const pad=(hi-lo)*0.12+0.5;lo-=pad;hi+=pad;
 const Y=v=>T+(hi-v)/(hi-lo)*(h-T-B);
 timegrid(g,X,tL,tR,T,B,h,true);
 g.font='11px system-ui';
 const step=[0.5,1,2,5,10,20,50,100,200,500,1000].find(s=>(hi-lo)/s<=6)||2000;
 for(let v=Math.ceil(lo/step)*step;v<=hi;v+=step){g.strokeStyle=css('--grid');g.beginPath();g.moveTo(L,Y(v));g.lineTo(w-R,Y(v));g.stroke();g.fillStyle=css('--mut');g.fillText(money(v).replace('+',''),4,Y(v)+4)}
 g.save();g.beginPath();g.rect(L,0,w-L-R,h);g.clip();
 g.strokeStyle=css('--mut');g.beginPath();g.moveTo(L,Y(0));g.lineTo(w-R,Y(0));g.stroke();
 const leg=[];
 for(const [b,col] of BOTS){const s=series[b];g.strokeStyle=css(col);g.lineWidth=b==='Total'?2.5:1.8;g.setLineDash(b==='Total'?[7,5]:[]);g.beginPath();
  s.forEach((p,i)=>{if(!i){g.moveTo(X(p[0]),Y(p[1]));return}g.lineTo(X(p[0]),Y(s[i-1][1]));g.lineTo(X(p[0]),Y(p[1]))});g.stroke();g.setLineDash([]);
  leg.push(`<span style="color:var(${col})">■</span> ${b==='Total'?'Cumulative, all bots (dashed)':b} ${money(s[s.length-1][1])}`)}
 for(const p of pts){if(p.won)continue;marker(g,p.bot,X(p.t),Y(valAt(series.Total,p.t)),css('--lose'))}
 g.restore();
 $('pl-legend').innerHTML=leg.join(' &nbsp; ')+' &nbsp; red marker = losing trade';
}
function bandQ(c,a,b){const Q=D&&D.quotes;if(!Q||Q.close!==c)return null;
 return (Q.bands||[]).find(r=>r.lo!=null&&r.hi!=null&&Math.abs(r.lo-a)<1&&Math.abs(r.hi+0.01-b)<1)||null}
const cents=x=>{if(x==null)return '–';const c=Math.round(x*1000)/10;return (Number.isInteger(c)?c:c.toFixed(1))+'¢'};
const edge=(m,a)=>{if(m==null||a==null)return '<span class="mut">–</span>';const e=Math.round((m-a)*1000)/10;return `<span style="color:var(${e>0?'--win':'--mut'})">${e>0?'+':''}${e.toFixed(1)}¢</span>`};
// model-minus-Kalshi color scale: -10¢ red, -5 orange, 0 yellow, +5 blue, +10 green
const DSTOP=[[-10,[215,48,31]],[-5,[242,142,43]],[0,[232,200,26]],[5,[59,130,246]],[10,[26,152,80]]];
function dcol(d){d=Math.max(-10,Math.min(10,d));for(let i=1;i<DSTOP.length;i++){const [a,ca]=DSTOP[i-1],[b,cb]=DSTOP[i];
 if(d<=b){const f=(d-a)/(b-a);return `rgb(${ca.map((c,k)=>Math.round(c+(cb[k]-c)*f)).join(',')})`}}return 'rgb(26,152,80)'}
function q15(){const R=D.q15||[],s=D.session||{};
 if(!$('q15scale').innerHTML)$('q15scale').innerHTML=DSTOP.map(([v])=>`<b style="color:${dcol(v)}">${v>0?'+':''}${v}¢</b>`).join(' ');
 if(!R.length){$('q15y').textContent='–';$('q15n').textContent='–';$('q15m').textContent='–';return}
 const r=R[R.length-1];$('q15y').textContent=`${cents(r[1])} / ${cents(r[2])}`;
 $('q15n').textContent=`${r[2]!=null?cents(1-r[2]):'–'} / ${r[1]!=null?cents(1-r[1]):'–'}`;
 $('q15m').textContent=r[3]!=null?`${cents(r[3])} / ${cents(1-r[3])}`:'–';
 $('q15info').textContent=s.ticker?`target ${usd(s.strike)} · closes ${hm(s.close_ts)} ET · BTC is ${P.length&&P[P.length-1][1]>s.strike?'above':'below'} the target`:'';
 const cv=$('q15c'),dpr=devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;cv.width=w*dpr;cv.height=h*dpr;const g=cv.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);
 const t0=s.open_ts||R[0][0],t1=s.close_ts||R[R.length-1][0],X=t=>4+(t-t0)/(t1-t0)*(w-40),Y=v=>6+(1-v)/1.12*(h-12);  // scale runs from +100¢ down to -12¢
 g.strokeStyle=css('--grid');g.font='10px system-ui';g.fillStyle=css('--mut');
 for(const v of [0,.5,1]){g.beginPath();g.moveTo(4,Y(v));g.lineTo(w-36,Y(v));g.stroke();g.fillText(Math.round(v*100)+'¢',w-32,Y(v)+3)}
 const line=(k,col,dash)=>{g.strokeStyle=col;g.lineWidth=1.6;g.setLineDash(dash);g.beginPath();let f=true;
  for(const q of R){if(q[k]==null){f=true;continue}f?g.moveTo(X(q[0]),Y(q[k])):g.lineTo(X(q[0]),Y(q[k]));f=false}g.stroke();g.setLineDash([])};
 // model minus Kalshi YES ask, drawn on the same cents scale around the 0¢ line, colored by size
 g.globalAlpha=.55;g.lineWidth=2.5;
 for(let i=1;i<R.length;i++){const a=R[i-1],b=R[i];if(a[2]==null||a[3]==null||b[2]==null||b[3]==null)continue;
  const da=a[3]-a[2],db=b[3]-b[2];g.strokeStyle=dcol(db*100);g.beginPath();g.moveTo(X(a[0]),Y(da));g.lineTo(X(b[0]),Y(db));g.stroke()}
 g.globalAlpha=1;
 line(2,css('--yes'),[]);line(3,css('--fg'),[4,3])}
let TSIG='';
function trades(){if(!H)return;const sig=JSON.stringify([H.range,H.buys]);if(sig===TSIG)return;TSIG=sig;
 const R=[];
 for(const r of H.range||[])R.push({t:r.t,bot:r.bot||'Range',mkt:`${r.lo==null?'':usd(r.lo)}–${r.hi==null?'':usd(r.hi+0.01).slice(1)} · ${hm(r.close)}${r.bot==='FP range'&&r.level?' · '+r.level:''}`,
  side:r.side,n:r.n,price:r.price,model:r.model,status:r.status,pnl:r.pnl,cost:r.cost});
 for(const b of H.buys||[])R.push({t:b.t,bot:b.bot,mkt:`15-min ${hm(Math.floor(b.t/900)*900+900)} · ${b.level||''}`,
  side:b.side,n:b.n,price:b.price!=null?b.price:b.ask,model:null,status:b.status,pnl:b.pnl,cost:b.cost});
 R.sort((a,b)=>a.t-b.t);
 let cum=0,spent=0,nset=0,wins=0,sp=0,sm=0,nm=0;const W={};const z=()=>({w:0,l:0,o:0,p:0,s:0,sp:0,sm:0,nm:0});for(const k of ['Range','Breakdown','Fast push','FP range'])W[k]=z();
 for(const r of R){const w=W[r.bot]||(W[r.bot]=z());
  if(r.cost!=null){spent+=r.cost;w.s+=r.cost}r.spent=r.cost!=null?spent:null;
  if(r.status==='open'){w.o++;r.cum=null;continue}nset++;
  r.status==='won'?(w.w++,wins++):w.l++;w.p+=r.pnl||0;cum+=r.pnl||0;r.cum=cum;r.avg=cum/nset;
  if(r.price!=null){sp+=r.price;w.sp+=r.price}if(r.model!=null){sm+=r.model;nm++;w.sm+=r.model;w.nm++}
  r.aprice=sp/nset;r.amodel=nm?sm/nm:null;r.awin=wins/nset}
 const mk=r=>r.bot==='Range'?'●':r.bot==='Breakdown'?'▲':r.bot==='FP range'?'◆':'■';
 $('wl').innerHTML=Object.entries(W).map(([k,v])=>{const n=v.w+v.l;
  return `<span><b>${k}</b>: <span style="color:var(--win)">${v.w} won</span> / <span style="color:var(--lose)">${v.l} lost</span>${n?' ('+Math.round(v.w/n*100)+'% win)':''}${v.o?' · '+v.o+' open':''} · ${money(v.p)}${n?' · avg '+money(v.p/n)+'/trade · avg price '+cents(v.sp/n)+(v.nm?' · avg model '+cents(v.sm/v.nm):''):''} · spent $${v.s.toFixed(2)}</span>`}).join('');
 $('tinfo').textContent=`${R.length} trades · total ${money(cum)} settled · avg ${nset?money(cum/nset):'–'} per trade · $${spent.toFixed(2)} spent · newest first`;
 let rows='<tr><th>Time (ET)</th><th>Bot</th><th>Market</th><th>Side</th><th>Qty</th><th>Price</th><th>Model</th><th>Win/ Loss</th><th>Avg model</th><th>Avg price</th><th>Win %</th><th>Spent</th><th>Total spent</th><th>P/L</th><th>Total P/L</th><th>Avg P/L per trade</th></tr>';
 for(let i=R.length-1;i>=0;i--){const r=R[i];const d=new Date(r.t*1000).toLocaleDateString('en-US',{timeZone:'America/New_York',month:'numeric',day:'numeric'});
  const res=r.status==='open'?'<span class="pill" style="color:var(--open)">open</span>':`<span class="pill ${r.status==='won'?'w':'l'}">${r.status==='won'?'Won':'Lost'}</span>`;
  const pc=v=>v==null?'<span class="mut">–</span>':`<span style="color:var(${v>=0?'--win':'--lose'})">${money(v)}</span>`;
  rows+=`<tr><td>${d} ${hm(r.t)}</td><td>${mk(r)} ${r.bot}</td><td>${r.mkt}</td><td>${r.side||''}</td><td>${r.n||''}</td><td>${r.price!=null?cents(r.price):'–'}</td><td>${r.model!=null?cents(r.model):'<span class="mut">–</span>'}</td><td>${res}</td><td>${r.amodel!=null?cents(r.amodel):'–'}</td><td>${r.aprice!=null?cents(r.aprice):'–'}</td><td>${r.awin!=null?(r.awin*100).toFixed(1)+'%':'–'}</td><td>${r.cost!=null?'$'+r.cost.toFixed(2):'–'}</td><td>${r.spent!=null?'$'+r.spent.toFixed(2):'–'}</td><td>${pc(r.pnl)}</td><td>${pc(r.cum)}</td><td>${pc(r.avg)}</td></tr>`}
 if(!R.length)rows+='<tr><td colspan="16" class="mut">No trades yet.</td></tr>';
 $('trades').innerHTML=rows}
function tables(px){q15();trades();
 let rows='<tr><th>Bought (ET)</th><th>Side</th><th>Band</th><th>Contracts</th><th>Paid</th><th>Model</th><th>Now</th><th>Closes</th></tr>';
 const ps=D.range.open.filter(p=>p.lo!=null);
 for(const p of ps){if(px==null)break;const inside=px>=p.lo&&px<=p.hi,win=(p.side==='NO')!==inside;
  const dist=inside?Math.min(px-p.lo,p.hi-px):(px<p.lo?p.lo-px:px-p.hi);
  rows+=`<tr><td>${hm(tsOf(p.time_utc))}</td><td>${p.side}</td><td>${usd(p.lo)}–${usd(p.hi+0.01)}</td><td>${p.contracts}</td><td>${Math.round(p.avg_price*100)}¢</td><td>${(p.model*100).toFixed(1)}¢</td>
  <td><span class="pill ${win?'w':'l'}">${win?'winning':'losing'} ${win?money(p.contracts*(1-p.avg_price)-p.fee):money(-p.cost)}</span> <span class="mut">${usd(dist)} ${inside?'inside':'outside'}</span></td><td>${hm(p.close_ts)}</td></tr>`}
 if(!ps.length)rows+='<tr><td colspan="8" class="mut">None open. The range bot starts checking at 30 minutes left in each hour.</td></tr>';
 $('pos').innerHTML=rows;
 const s=D.session||{};rows='<tr><th>Break (ET)</th><th>Bot</th><th>Side</th><th>Level</th><th>Ask</th><th>Now</th></tr>';
 for(const b of D.buys){const win=(b.side==='YES')===(px>s.strike);
  rows+=`<tr><td>${hms(tsOf(b.time_utc))}</td><td>${b.bot}</td><td>${b.side}</td><td>${b.level} ${b.level_price?usd(b.level_price):''}</td><td>${b.ask?Math.round(b.ask*100)+'¢':''}</td><td><span class="pill ${win?'w':'l'}">${win?'winning':'losing'}</span></td></tr>`}
 if(!D.buys.length)rows+=`<tr><td colspan="6" class="mut">No buys this session${s.strike?' (strike '+usd(s.strike)+')':''}. Both bots only buy in the last 1–3 minutes.</td></tr>`;
 $('pos15').innerHTML=rows;
}
tick();setInterval(tick,2000);setInterval(loadHistory,60000);loadLevels();setInterval(loadLevels,15000);
</script></body></html>
"""
