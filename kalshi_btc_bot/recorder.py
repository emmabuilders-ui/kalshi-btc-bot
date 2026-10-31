"""Record Kalshi KXBTC15M order books and BTC exchange prices, once a second.

Kalshi keeps no order-book history, so the only way to study it is to record it.
Each second this writes one JSON line with:

- the open KXBTC15M contract's ticker, strike and close time
- its order book (YES bids and NO bids, best levels first), best YES bid/ask
- Coinbase, Kraken, Bitstamp and Binance BTC prices (whatever responds in time)

Files rotate hourly: <out>/<YYYY-MM-DD>/<HH>.jsonl.gz (UTC). Read-only: no orders.
"""

from __future__ import annotations

import gzip
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger("kalshi_btc_bot")

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXBTC15M"

PRICE_URLS = {
    "coinbase": ("https://api.exchange.coinbase.com/products/BTC-USD/ticker", lambda j: float(j["price"])),
    "kraken": ("https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
               lambda j: float(next(iter(j["result"].values()))["c"][0])),
    "bitstamp": ("https://www.bitstamp.net/api/v2/ticker/btcusd/", lambda j: float(j["last"])),
    "binance": ("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", lambda j: float(j["price"])),
}


def _levels(raw) -> list[list[float]]:
    """Kalshi returns [[price, size], ...] ascending by price; return best (highest) first."""
    return sorted(([float(p), float(q)] for p, q in (raw or [])), reverse=True)


def parse_book(j: dict, depth: int = 10) -> dict:
    ob = j.get("orderbook_fp") or j.get("orderbook") or {}
    yes = _levels(ob.get("yes_dollars") or ob.get("yes"))
    no = _levels(ob.get("no_dollars") or ob.get("no"))
    # Integer-cent fallback for the older API shape.
    if yes and yes[0][0] > 1:
        yes = [[p / 100, q] for p, q in yes]
    if no and no[0][0] > 1:
        no = [[p / 100, q] for p, q in no]
    best_bid = yes[0][0] if yes else None
    best_ask = round(1 - no[0][0], 4) if no else None
    return {"yes_bids": yes[:depth], "no_bids": no[:depth], "yes_bid": best_bid, "yes_ask": best_ask}


class Recorder:
    def __init__(self, out: str | Path, interval: float = 1.0, depth: int = 10,
                 session: requests.Session | None = None, timeout: float = 0.8):
        self.out = Path(out)
        self.interval = interval
        self.depth = depth
        self.s = session or requests.Session()
        self.s.headers["User-Agent"] = "kalshi-btc-bot-recorder"
        self.timeout = timeout
        self.market: dict | None = None
        self.pool = ThreadPoolExecutor(max_workers=len(PRICE_URLS) + 1)
        self._fh = None
        self._fh_key = None

    # ------------------------------------------------------------------ helpers
    def _get(self, url: str, **params):
        r = self.s.get(url, params=params or None, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _current_market(self, now: float) -> dict | None:
        if self.market and now < self.market["close_ts"]:
            return self.market
        j = self._get(f"{KALSHI}/markets", series_ticker=SERIES, status="open", limit=10)
        opens = []
        for m in j.get("markets", []):
            close_ts = datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).timestamp()
            open_ts = datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp()
            if open_ts <= now < close_ts:
                opens.append({"ticker": m["ticker"], "strike": m.get("floor_strike"),
                              "open_ts": open_ts, "close_ts": close_ts})
        self.market = min(opens, key=lambda m: m["close_ts"]) if opens else None
        if self.market:
            log.info("recording %s (strike %s)", self.market["ticker"], self.market["strike"])
        return self.market

    def _price(self, name: str):
        url, parse = PRICE_URLS[name]
        try:
            return parse(self._get(url))
        except Exception:
            return None

    def _file(self, now: float):
        dt = datetime.fromtimestamp(now, tz=timezone.utc)
        key = dt.strftime("%Y-%m-%d/%H")
        if key != self._fh_key:
            if self._fh:
                self._fh.close()
            path = self.out / f"{key}.jsonl.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = gzip.open(path, "at")
            self._fh_key = key
        return self._fh

    # --------------------------------------------------------------------- loop
    def snapshot(self, now: float) -> dict | None:
        try:
            m = self._current_market(now)
        except Exception as e:  # network hiccup: skip this second
            log.warning("market lookup failed: %s", e)
            return None
        if not m:
            return None
        prices = {n: self.pool.submit(self._price, n) for n in PRICE_URLS}
        try:
            book = parse_book(self._get(f"{KALSHI}/markets/{m['ticker']}/orderbook", depth=self.depth), self.depth)
        except Exception as e:
            log.warning("orderbook failed: %s", e)
            book = None
        row = {"ts": round(now, 3), "ticker": m["ticker"], "strike": m["strike"],
               "secs_left": round(m["close_ts"] - now, 1), "book": book,
               "btc": {n: f.result() for n, f in prices.items()}}
        return row

    def run(self, seconds: float | None = None) -> int:
        start = time.time()
        n = 0
        while seconds is None or time.time() - start < seconds:
            now = time.time()
            row = self.snapshot(now)
            if row:
                fh = self._file(now)
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")
                n += 1
                if n % 300 == 0:
                    fh.flush()
                    log.info("%d snapshots recorded", n)
            time.sleep(max(0.0, self.interval - (time.time() - now)))
        if self._fh:
            self._fh.close()
        return n
