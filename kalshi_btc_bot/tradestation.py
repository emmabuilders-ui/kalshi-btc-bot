"""TradeStation v3 market data (read-only).

Pulls bars, quotes, symbol details and option chains straight from TradeStation's
WebAPI so no files have to be downloaded. Nothing here reads positions, orders or
balances, and nothing places orders: the token is requested with the MarketData
and ReadAccount scopes only.

Credentials come from the environment, never from code or chat:

    TS_CLIENT_ID       API key id issued by TradeStation Client Services
    TS_CLIENT_SECRET   its secret
    TS_REFRESH_TOKEN   long-lived token from the one-time browser login
                       (see examples/ts_login.py)

Access tokens last 20 minutes and are refreshed automatically.

Written against TradeStation's published v3 spec; exercised against the live API
only once Michael's key arrived, so check the first run's log before trusting
unusual endpoints (symbol search, option chain snapshots).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Iterator
from zoneinfo import ZoneInfo

import pandas as pd
import requests

log = logging.getLogger("kalshi_btc_bot.tradestation")

API = "https://api.tradestation.com/v3"
SIGNIN = "https://signin.tradestation.com"
SCOPES = "openid profile offline_access MarketData ReadAccount"
ET = ZoneInfo("America/New_York")

# TradeStation caps on one bar request (intraday bars); daily and longer are unlimited.
MAX_INTRADAY_BARSBACK = 57_600
MAX_BARSBACK = 57_600


class TradeStationError(RuntimeError):
    pass


# --------------------------------------------------------------------------- auth
def auth_url(client_id: str, redirect_uri: str, state: str = "kbot") -> str:
    """Browser URL for the one-time login. Market-data scopes only, no Trade."""
    from urllib.parse import urlencode
    q = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "audience": "https://api.tradestation.com",
        "scope": SCOPES,
        "state": state,
    }
    return f"{SIGNIN}/authorize?{urlencode(q)}"


def exchange_code(client_id: str, client_secret: str, code: str, redirect_uri: str,
                  session: requests.Session | None = None) -> dict:
    """Trade the login code for tokens. Returns the token response (has refresh_token)."""
    s = session or requests.Session()
    r = s.post(f"{SIGNIN}/oauth/token", data={
        "grant_type": "authorization_code",
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": redirect_uri,
    }, timeout=30)
    if r.status_code != 200:
        raise TradeStationError(f"token exchange failed {r.status_code}: {r.text[:300]}")
    return r.json()


@dataclass
class Credentials:
    client_id: str
    client_secret: str
    refresh_token: str

    @classmethod
    def from_env(cls) -> "Credentials":
        missing = [k for k in ("TS_CLIENT_ID", "TS_CLIENT_SECRET", "TS_REFRESH_TOKEN") if not os.environ.get(k)]
        if missing:
            raise TradeStationError("missing environment variables: " + ", ".join(missing))
        return cls(os.environ["TS_CLIENT_ID"], os.environ["TS_CLIENT_SECRET"], os.environ["TS_REFRESH_TOKEN"])


# --------------------------------------------------------------------------- client
@dataclass
class TradeStation:
    """Thin client. All timestamps returned as tz-aware pandas Timestamps in ET."""

    creds: Credentials = field(default_factory=Credentials.from_env)
    session: requests.Session = field(default_factory=requests.Session)
    timeout: float = 30.0
    _token: str = ""
    _token_exp: float = 0.0

    # ---- auth
    def _refresh(self) -> None:
        r = self.session.post(f"{SIGNIN}/oauth/token", data={
            "grant_type": "refresh_token",
            "client_id": self.creds.client_id,
            "client_secret": self.creds.client_secret,
            "refresh_token": self.creds.refresh_token,
        }, timeout=self.timeout)
        if r.status_code != 200:
            raise TradeStationError(f"token refresh failed {r.status_code}: {r.text[:300]}")
        j = r.json()
        self._token = j["access_token"]
        self._token_exp = time.time() + int(j.get("expires_in", 1200)) - 60

    def _headers(self) -> dict:
        if not self._token or time.time() >= self._token_exp:
            self._refresh()
        return {"Authorization": f"Bearer {self._token}"}

    def _get(self, path: str, params: dict | None = None, retries: int = 3):
        url = f"{API}{path}"
        for attempt in range(retries + 1):
            r = self.session.get(url, params=params, headers=self._headers(), timeout=self.timeout)
            if r.status_code == 429 and attempt < retries:
                wait = float(r.headers.get("Retry-After", 2 ** attempt))
                log.warning("tradestation 429 on %s, sleeping %.0fs", path, wait)
                time.sleep(wait)
                continue
            if r.status_code == 401 and attempt < retries:
                self._token = ""
                continue
            if r.status_code != 200:
                raise TradeStationError(f"GET {path} -> {r.status_code}: {r.text[:300]}")
            return r.json()
        raise TradeStationError(f"GET {path}: gave up after {retries} retries")

    def _stream(self, path: str, params: dict | None, max_seconds: float) -> Iterator[dict]:
        """Read a TradeStation JSON-lines stream for up to max_seconds, yielding objects."""
        url = f"{API}{path}"
        deadline = time.time() + max_seconds
        with self.session.get(url, params=params, headers=self._headers(), stream=True,
                              timeout=(self.timeout, max_seconds)) as r:
            if r.status_code != 200:
                raise TradeStationError(f"STREAM {path} -> {r.status_code}: {r.text[:300]}")
            try:
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        j = json.loads(line)
                    except ValueError:
                        continue
                    if "Error" in j:
                        raise TradeStationError(f"STREAM {path}: {j}")
                    yield j
                    if time.time() > deadline:
                        return
            except requests.exceptions.ReadTimeout:
                return

    # ---- symbols
    def symbol_details(self, symbols: str | Iterable[str]) -> pd.DataFrame:
        """Asset type, exchange, tick size, currency, underlying etc. Up to 50 symbols."""
        syms = [symbols] if isinstance(symbols, str) else list(symbols)
        j = self._get(f"/marketdata/symbols/{','.join(syms)}")
        for e in j.get("Errors", []) or []:
            log.warning("symbol_details: %s", e)
        return pd.DataFrame(j.get("Symbols", []))

    def search_symbols(self, criteria: str) -> pd.DataFrame:
        """Symbol search, e.g. 'N=ES&C=Future' or 'N=AAPL'. Unverified against live API."""
        j = self._get(f"/data/symbols/search/{criteria}")
        return pd.DataFrame(j if isinstance(j, list) else j.get("Symbols", []))

    def crypto_symbols(self) -> list[str]:
        return self._get("/marketdata/crypto/symbolnames").get("SymbolNames", [])

    # ---- quotes
    def quotes(self, symbols: str | Iterable[str]) -> pd.DataFrame:
        """Snapshot quotes (last, bid, ask, volume, open/high/low, prev close). Up to 100 symbols."""
        syms = [symbols] if isinstance(symbols, str) else list(symbols)
        j = self._get(f"/marketdata/quotes/{','.join(syms)}")
        for e in j.get("Errors", []) or []:
            log.warning("quotes: %s", e)
        df = pd.DataFrame(j.get("Quotes", []))
        if "TradeTime" in df:
            df["TradeTimeET"] = pd.to_datetime(df["TradeTime"], utc=True, errors="coerce").dt.tz_convert(ET)
        return _numeric(df)

    # ---- bars
    def bars(self, symbol: str, interval: int = 1, unit: str = "Minute",
             barsback: int | None = None, start: datetime | str | None = None,
             end: datetime | str | None = None, session_template: str = "Default") -> pd.DataFrame:
        """OHLCV bars, oldest first, index = bar time in ET.

        unit: Minute (interval 1–1440), Daily, Weekly, Monthly.
        Give either barsback (count ending at `end`/now) or start (walks forward in
        chunks of 57,600 bars until `end`). session_template: Default, USEQPre,
        USEQPost, USEQPreAndPost, USEQ24Hour (equities only).
        """
        unit = unit.capitalize()
        if unit not in ("Minute", "Daily", "Weekly", "Monthly"):
            raise ValueError("unit must be Minute, Daily, Weekly or Monthly")
        base = {"interval": str(interval if unit == "Minute" else 1), "unit": unit}
        if session_template != "Default":
            base["sessiontemplate"] = session_template
        frames = []
        if start is None:
            n = barsback or 100
            last = _ts(end)
            while n > 0:
                p = dict(base, barsback=str(min(n, MAX_BARSBACK)))
                if last is not None:
                    p["lastdate"] = _iso(last)
                df = self._bars_page(symbol, p)
                if df.empty:
                    break
                frames.append(df)
                n -= len(df)
                last = df.index[0].tz_convert("UTC").to_pydatetime() - timedelta(seconds=1)
                if len(df) < min(n + len(df), MAX_BARSBACK):
                    break
        else:
            first = _ts(start)
            stop = _ts(end) or datetime.now(timezone.utc)
            while first is not None and first < stop:
                p = dict(base, firstdate=_iso(first), lastdate=_iso(stop))
                df = self._bars_page(symbol, p)
                if df.empty:
                    break
                frames.append(df)
                nxt = df.index[-1].tz_convert("UTC").to_pydatetime() + timedelta(seconds=1)
                if nxt <= first or len(df) < MAX_BARSBACK:
                    break
                first = nxt
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames).sort_index()
        return out[~out.index.duplicated(keep="last")]

    def _bars_page(self, symbol: str, params: dict) -> pd.DataFrame:
        j = self._get(f"/marketdata/barcharts/{symbol}", params)
        rows = j.get("Bars", [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["TimeStamp"], utc=True).dt.tz_convert(ET)
        df.index.name = "time_et"
        df = df.rename(columns={"TotalVolume": "Volume"})
        keep = [c for c in ("Open", "High", "Low", "Close", "Volume", "OpenInterest", "UpVolume",
                            "DownVolume", "TotalTicks", "BarStatus") if c in df]
        return _numeric(df[keep])

    # ---- options
    def option_expirations(self, underlying: str, strike_price: float | None = None) -> pd.DataFrame:
        p = {"strikePrice": str(strike_price)} if strike_price is not None else None
        j = self._get(f"/marketdata/options/expirations/{underlying}", p)
        df = pd.DataFrame(j.get("Expirations", []))
        if "Date" in df:
            df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(ET).dt.date
        return df

    def option_strikes(self, underlying: str, expiration: str | None = None) -> list[float]:
        p = {"expiration": expiration} if expiration else None
        j = self._get(f"/marketdata/options/strikes/{underlying}", p)
        return sorted(float(s[0]) for s in j.get("Strikes", []) if s)

    def option_chain(self, underlying: str, expiration: str | None = None,
                     strike_proximity: int = 5, max_seconds: float = 8.0) -> pd.DataFrame:
        """Snapshot of an option chain (Greeks, IV, bid/ask, volume, OI).

        TradeStation serves chains only as a stream, so this reads the stream until
        every contract has reported once or max_seconds passes. expiration like
        '2026-12-18' (default: nearest). strike_proximity = strikes each side of ATM.
        """
        p = {"strikeProximity": str(strike_proximity)}
        if expiration:
            p["expiration"] = expiration
        rows: dict[str, dict] = {}
        for j in self._stream(f"/marketdata/stream/options/chains/{underlying}", p, max_seconds):
            if "Heartbeat" in j:
                if rows:
                    break
                continue
            legs = j.get("Legs") or []
            key = "|".join(l.get("Symbol", "") for l in legs) or str(len(rows))
            if key in rows:
                break          # stream has started repeating: snapshot complete
            flat = {k: v for k, v in j.items() if k != "Legs"}
            if legs:
                flat.update({"Symbol": legs[0].get("Symbol"), "StrikePrice": legs[0].get("StrikePrice"),
                             "Expiration": legs[0].get("Expiration"), "OptionType": legs[0].get("OptionType")})
            rows[key] = flat
        df = pd.DataFrame(list(rows.values()))
        if "Expiration" in df:
            df["ExpirationET"] = pd.to_datetime(df["Expiration"], utc=True, errors="coerce").dt.tz_convert(ET)
        return _numeric(df)


# --------------------------------------------------------------------------- helpers
def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c]):
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().sum() and conv.notna().sum() >= df[c].notna().sum() * 0.9:
                df[c] = conv
    return df


def _ts(t: datetime | str | None) -> datetime | None:
    if t is None:
        return None
    if isinstance(t, str):
        t = pd.Timestamp(t).to_pydatetime()
    if t.tzinfo is None:
        t = t.replace(tzinfo=ET)       # naive times are ET, per project convention
    return t.astimezone(timezone.utc)


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
