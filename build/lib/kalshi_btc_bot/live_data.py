"""Live market data: BTC spot feeds and Kalshi public market data (read-only).

None of this places orders. Kalshi market data endpoints are public; the parser
accepts both the older integer-cent fields and the newer *_dollars fields.
These clients were written against Kalshi's and the exchanges' documented public
APIs but could not be exercised from the build environment, so check the first
paper session's log before trusting them.
"""

from __future__ import annotations

import logging
import statistics
from datetime import datetime

import requests

from .config import KalshiConfig
from .models import Market, Quote

log = logging.getLogger("kalshi_btc_bot")


# --------------------------------------------------------------------------- BTC
class RestPriceFeed:
    name = "base"
    url = ""

    def __init__(self, session: requests.Session | None = None, timeout: float = 0.8):
        self.s = session or requests.Session()
        self.timeout = timeout

    def get_price(self) -> float:
        r = self.s.get(self.url, timeout=self.timeout)
        r.raise_for_status()
        return self.parse(r.json())

    def parse(self, j) -> float:
        raise NotImplementedError


class CoinbaseFeed(RestPriceFeed):
    name = "coinbase"
    url = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"

    def parse(self, j) -> float:
        return float(j["price"])


class KrakenFeed(RestPriceFeed):
    name = "kraken"
    url = "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"

    def parse(self, j) -> float:
        return float(next(iter(j["result"].values()))["c"][0])


class BitstampFeed(RestPriceFeed):
    name = "bitstamp"
    url = "https://www.bitstamp.net/api/v2/ticker/btcusd/"

    def parse(self, j) -> float:
        return float(j["last"])


class CompositeFeed:
    """Median of several BRTI constituent exchanges. Closer to BRTI than any one
    exchange, but still not BRTI (which uses order books and volume weighting)."""

    name = "composite"

    def __init__(self, feeds):
        self.feeds = feeds

    def get_price(self) -> float:
        px = []
        for f in self.feeds:
            try:
                px.append(f.get_price())
            except Exception as e:      # one exchange failing should not stop the loop
                log.debug("feed %s failed: %s", f.name, e)
        if not px:
            raise RuntimeError("all BTC feeds failed")
        return statistics.median(px)


FEEDS = {"coinbase": CoinbaseFeed, "kraken": KrakenFeed, "bitstamp": BitstampFeed}


def make_feed(name: str):
    if name == "composite":
        return CompositeFeed([cls() for cls in FEEDS.values()])
    return FEEDS[name]()


# ------------------------------------------------------------------------ Kalshi
def _ts(v) -> int:
    return int(datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp())


def _px(d: dict, key: str) -> float | None:
    """Read a price as dollars from either `<key>_dollars` (string) or `<key>` (cents)."""
    v = d.get(f"{key}_dollars")
    if v not in (None, ""):
        return float(v)
    v = d.get(key)
    if v in (None, ""):
        return None
    return float(v) / 100.0


def parse_market(d: dict) -> Market:
    strike = None
    for k in ("floor_strike", "cap_strike", "functional_strike", "custom_strike"):
        v = d.get(k)
        if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace(".", "", 1).isdigit()):
            strike = float(v)
            break
    if strike is None:
        raise ValueError(f"no numeric strike on market {d.get('ticker')}")
    res = d.get("result") or None
    return Market(d["ticker"], _ts(d["open_time"]), _ts(d["close_time"]), strike,
                  res.lower() if res in ("yes", "no", "YES", "NO") else None)


def _levels(raw) -> list[tuple[float, float]]:
    out = []
    for p, q in raw or []:
        # Integer prices are cents (legacy "yes"/"no" arrays); strings are dollars.
        out.append((p / 100.0 if isinstance(p, int) else float(p), float(q)))
    return out


def parse_orderbook(t: int, ticker: str, j: dict, last_price: float | None) -> Quote:
    """Kalshi books list resting bids for YES and NO. A NO bid at p is a YES ask at 1 - p."""
    ob = j.get("orderbook_fp") or j.get("orderbook") or {}
    yes = _levels(ob.get("yes_dollars") or ob.get("yes"))
    no = _levels(ob.get("no_dollars") or ob.get("no"))
    bids = tuple(sorted(yes, key=lambda x: -x[0]))
    asks = tuple(sorted(((round(1.0 - p, 4), q) for p, q in no), key=lambda x: x[0]))
    return Quote(t, ticker,
                 bids[0][0] if bids else None, asks[0][0] if asks else None,
                 bids[0][1] if bids else None, asks[0][1] if asks else None,
                 last_price, ask_levels=asks, bid_levels=bids)


class KalshiMarketData:
    def __init__(self, cfg: KalshiConfig, session: requests.Session | None = None):
        self.cfg = cfg
        self.s = session or requests.Session()

    def _get(self, path: str, **params) -> dict:
        r = self.s.get(self.cfg.api_base + path, params=params, timeout=self.cfg.timeout_s)
        r.raise_for_status()
        return r.json()

    def open_markets(self) -> list[Market]:
        j = self._get("/markets", series_ticker=self.cfg.series_ticker, status="open", limit=20)
        out = []
        for d in j.get("markets", []):
            try:
                out.append(parse_market(d))
            except (KeyError, ValueError) as e:
                log.warning("skipping market: %s", e)
        return sorted(out, key=lambda m: m.close_time)

    def market(self, ticker: str) -> tuple[Market, dict]:
        d = self._get(f"/markets/{ticker}")["market"]
        return parse_market(d), d

    def quote(self, t: int, ticker: str, last_price: float | None) -> Quote:
        j = self._get(f"/markets/{ticker}/orderbook", depth=self.cfg.orderbook_depth)
        return parse_orderbook(t, ticker, j, last_price)
