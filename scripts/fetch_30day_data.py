#!/usr/bin/env python3
"""Download 30 days of BTC 1-second klines and Kalshi KXBTC15M trade history.

Produces two ZIPs in --out:

  BTC_1sec_30days.zip       the daily Binance files BTCUSDT-1s-YYYY-MM-DD.zip,
                            checksum-verified, stored as-is
  Kalshi_BTC15M_30days.zip  markets.csv (one row per contract: ticker, open/close
                            times, strike, settlement) and trades/YYYY-MM-DD.csv
                            (every execution in every contract that opened that
                            UTC day, with YES and NO price, size, taker side)

Standard library only, so it runs on Python 3.11+ without installing anything:

  python scripts/fetch_30day_data.py --start 2026-09-26 --end 2026-10-25 --out data

Re-running resumes: finished downloads are cached in --out/_cache and skipped.
Read-only; uses only public endpoints, no Kalshi API key needed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

BINANCE_URL = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1s/BTCUSDT-1s-{d}.zip"
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXBTC15M"
UA = {"User-Agent": "kalshi-btc-bot-data-fetch/1.0"}

MARKET_COLS = ["ticker", "event_ticker", "open_time", "close_time", "expiration_time",
               "strike", "strike_type", "floor_strike", "cap_strike", "result",
               "settlement_value", "expiration_value", "volume", "status", "source"]
TRADE_COLS = ["ticker", "trade_id", "created_time", "ts_unix", "yes_price", "no_price",
              "count", "taker_side"]


# ----------------------------------------------------------------------- http
_last_call = 0.0


def http_get(url: str, params: dict | None = None, min_interval: float = 0.0,
             tries: int = 6) -> bytes:
    """GET with polite pacing and retry/backoff on 429, 5xx and network errors."""
    global _last_call
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    for attempt in range(tries):
        wait = _last_call + min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            if e.code != 429 and e.code < 500:
                raise RuntimeError(f"HTTP {e.code} for {url}: {e.read()[:300]!r}") from e
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e
        delay = 2 ** attempt
        print(f"  retry {attempt + 1}/{tries} in {delay}s ({err})", file=sys.stderr)
        time.sleep(delay)
    raise RuntimeError(f"giving up on {url}")


def kalshi(path: str, **params) -> dict:
    # Kalshi's basic tier allows ~20 reads/s; stay well under it.
    return json.loads(http_get(KALSHI_API + path, params, min_interval=0.08))


def days(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def iso_to_unix(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def to_unix(dt: date) -> int:
    return int(datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc).timestamp())


# -------------------------------------------------------------------- binance
def fetch_binance(start: date, end: date, out: Path, cache: Path) -> Path:
    cache = cache / "binance"
    cache.mkdir(parents=True, exist_ok=True)
    files = []
    for d in days(start, end):
        name = f"BTCUSDT-1s-{d.isoformat()}.zip"
        dest = cache / name
        url = BINANCE_URL.format(d=d.isoformat())
        if not dest.exists():
            print(f"binance {name}")
            data = http_get(url)
            try:
                expected = http_get(url + ".CHECKSUM").decode().split()[0]
                got = hashlib.sha256(data).hexdigest()
                if got != expected:
                    raise RuntimeError(f"checksum mismatch for {name}")
            except urllib.error.HTTPError:
                print(f"  no checksum published for {name}, skipping verification")
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.rename(dest)
        files.append(dest)
    zpath = out / "BTC_1sec_30days.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:   # already compressed
        for f in files:
            z.write(f, f.name)
    return zpath


# --------------------------------------------------------------------- kalshi
def _price(d: dict, key: str) -> str:
    """Dollar price as a string from `<key>_dollars` or legacy integer-cent `<key>`."""
    v = d.get(f"{key}_dollars")
    if v not in (None, ""):
        return f"{float(v):.4f}"
    v = d.get(key)
    return "" if v in (None, "") else f"{float(v) / 100:.4f}"


def _count(d: dict) -> str:
    v = d.get("count_fp", d.get("count"))
    return "" if v is None else str(v)


def _paged(path: str, key: str, **params):
    cursor = None
    while True:
        j = kalshi(path, limit=1000, cursor=cursor, **params)
        yield from j.get(key) or []
        cursor = j.get("cursor")
        if not cursor:
            return


def list_markets(start: date, end: date) -> list[dict]:
    """Every KXBTC15M contract that opened between start 00:00 and end 24:00 UTC.

    Kalshi moves older settled markets out of /markets into /historical/markets,
    so query both and de-duplicate by ticker.
    """
    lo, hi = to_unix(start), to_unix(end + timedelta(days=1))
    rng = dict(series_ticker=SERIES, min_close_ts=lo + 1, max_close_ts=hi + 15 * 60)
    found: dict[str, dict] = {}
    for path, src in (("/markets", "live"), ("/historical/markets", "historical")):
        try:
            for m in _paged(path, "markets", **rng):
                if lo <= iso_to_unix(m["open_time"]) < hi and m["ticker"] not in found:
                    m["_source"] = src
                    found[m["ticker"]] = m
        except (urllib.error.HTTPError, RuntimeError) as e:
            print(f"  {path} unavailable ({e}); continuing", file=sys.stderr)
    return sorted(found.values(), key=lambda m: m["open_time"])


def market_row(m: dict) -> dict:
    strike = next((m[k] for k in ("floor_strike", "cap_strike", "functional_strike")
                   if m.get(k) not in (None, "")), "")
    return {
        "ticker": m["ticker"], "event_ticker": m.get("event_ticker", ""),
        "open_time": m.get("open_time", ""), "close_time": m.get("close_time", ""),
        "expiration_time": m.get("expiration_time", ""), "strike": strike,
        "strike_type": m.get("strike_type", ""), "floor_strike": m.get("floor_strike", ""),
        "cap_strike": m.get("cap_strike", ""), "result": (m.get("result") or "").upper(),
        "settlement_value": _price(m, "settlement_value"),
        "expiration_value": m.get("expiration_value", ""),
        "volume": m.get("volume_fp", m.get("volume", "")), "status": m.get("status", ""),
        "source": m.get("_source", ""),
    }


def fetch_trades(ticker: str, src: str) -> list[dict]:
    paths = ["/historical/trades", "/markets/trades"] if src == "historical" \
        else ["/markets/trades", "/historical/trades"]
    for path in paths:
        try:
            rows = list(_paged(path, "trades", ticker=ticker))
        except (urllib.error.HTTPError, RuntimeError) as e:
            print(f"  {path} failed for {ticker} ({e})", file=sys.stderr)
            continue
        if rows:
            return rows
    return []


def trade_row(t: dict) -> dict:
    return {
        "ticker": t.get("ticker", ""), "trade_id": t.get("trade_id", ""),
        "created_time": t.get("created_time", ""),
        "ts_unix": f"{iso_to_unix(t['created_time']):.6f}" if t.get("created_time") else "",
        "yes_price": _price(t, "yes_price"), "no_price": _price(t, "no_price"),
        "count": _count(t), "taker_side": t.get("taker_side", ""),
    }


def fetch_kalshi(start: date, end: date, out: Path, cache: Path) -> Path:
    cache = cache / "kalshi"
    (cache / "trades").mkdir(parents=True, exist_ok=True)
    mcache = cache / "markets.json"
    if mcache.exists():
        markets = json.loads(mcache.read_text())
    else:
        print("kalshi: listing contracts")
        markets = list_markets(start, end)
        if not markets:
            raise RuntimeError("Kalshi returned no KXBTC15M contracts for that range")
        mcache.write_text(json.dumps(markets))
    print(f"kalshi: {len(markets)} contracts")

    by_day: dict[str, list[Path]] = {}
    for i, m in enumerate(markets, 1):
        tk = m["ticker"]
        f = cache / "trades" / f"{tk}.json"
        if not f.exists():
            rows = sorted((trade_row(t) for t in fetch_trades(tk, m.get("_source", ""))),
                          key=lambda r: (r["ts_unix"], r["trade_id"]))
            tmp = f.with_suffix(".part")
            tmp.write_text(json.dumps(rows))
            tmp.rename(f)
            print(f"  [{i}/{len(markets)}] {tk}: {len(rows)} trades")
        day = datetime.fromtimestamp(iso_to_unix(m["open_time"]), timezone.utc).date().isoformat()
        by_day.setdefault(day, []).append(f)

    zpath = out / "Kalshi_BTC15M_30days.zip"
    n_trades = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("markets.csv", _csv(MARKET_COLS, [market_row(m) for m in markets]))
        for day, fs in sorted(by_day.items()):
            rows = [r for f in fs for r in json.loads(f.read_text())]
            n_trades += len(rows)
            z.writestr(f"trades/{day}.csv", _csv(TRADE_COLS, rows))
        z.writestr("README.txt", KALSHI_README.format(
            start=start, end=end, n=len(markets), t=n_trades,
            made=datetime.now(timezone.utc).isoformat(timespec="seconds")))
    print(f"kalshi: {n_trades} trades written")
    return zpath


def _csv(cols, rows) -> str:
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


KALSHI_README = """Kalshi KXBTC15M contracts that opened {start} 00:00 UTC through {end} 23:59 UTC.
{n} contracts, {t} trades. Built {made} from Kalshi's public trade API.

markets.csv   one row per contract
  strike            the opening BTC reference price the contract settles against
                    (floor_strike for "above" contracts)
  result            YES or NO
  expiration_value  the settlement BTC index value Kalshi published
  source            live = /markets, historical = /historical/markets

trades/<day>.csv  every individual execution, grouped by the day the contract opened
  yes_price / no_price  in dollars (yes_price + no_price = 1.00)
  count                 contracts traded
  taker_side            which side crossed the spread (yes or no)
Each execution has a YES price, so every row is usable as a YES trade print.
"""


# ----------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2026-09-26", type=date.fromisoformat)
    p.add_argument("--end", default="2026-10-25", type=date.fromisoformat, help="inclusive")
    p.add_argument("--out", default="data", type=Path)
    p.add_argument("--only", choices=["btc", "kalshi"])
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cache = a.out / "_cache"
    if a.only != "kalshi":
        print("wrote", fetch_binance(a.start, a.end, a.out, cache))
    if a.only != "btc":
        print("wrote", fetch_kalshi(a.start, a.end, a.out, cache))


if __name__ == "__main__":
    main()
