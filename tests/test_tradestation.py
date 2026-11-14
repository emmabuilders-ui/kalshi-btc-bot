import json

import pandas as pd

from kalshi_btc_bot import tradestation as ts


class FakeResp:
    def __init__(self, j, status=200):
        self._j, self.status_code, self.headers, self.text = j, status, {}, json.dumps(j)

    def json(self):
        return self._j


class FakeSession:
    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def post(self, url, data=None, timeout=None):
        return FakeResp({"access_token": "t", "expires_in": 1200})

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params))
        return FakeResp(self.pages.pop(0))


def client(pages):
    c = ts.TradeStation(creds=ts.Credentials("id", "sec", "rt"), session=FakeSession(pages))
    return c


def bar(t, close):
    return {"TimeStamp": t, "Open": "1", "High": "2", "Low": "0.5", "Close": str(close), "TotalVolume": "10",
            "OpenInterest": "0", "BarStatus": "Closed"}


def test_bars_index_in_et_and_numeric():
    c = client([{"Bars": [bar("2026-11-06T14:30:00Z", 100), bar("2026-11-06T14:31:00Z", 101)]}])
    df = c.bars("SPY", 1, "Minute", barsback=2)
    assert str(df.index[0]) == "2026-11-06 09:30:00-05:00"
    assert df["Close"].tolist() == [100.0, 101.0]
    assert df["Volume"].dtype.kind in "if"
    url, params = c.session.calls[0]
    assert url.endswith("/marketdata/barcharts/SPY") and params["barsback"] == "2"


def test_bars_pages_backwards_past_the_cap(monkeypatch):
    monkeypatch.setattr(ts, "MAX_BARSBACK", 2)
    p1 = {"Bars": [bar("2026-11-06T14:32:00Z", 3), bar("2026-11-06T14:33:00Z", 4)]}
    p2 = {"Bars": [bar("2026-11-06T14:30:00Z", 1), bar("2026-11-06T14:31:00Z", 2)]}
    c = client([p1, p2])
    df = c.bars("SPY", barsback=4)
    assert df["Close"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert c.session.calls[1][1]["lastdate"] == "2026-11-06T14:31:59Z"


def test_bars_forward_from_start_stops_on_short_page():
    c = client([{"Bars": [bar("2026-01-02T21:00:00Z", 5)]}])
    df = c.bars("ES", unit="Daily", start="2026-01-01", end="2026-01-05")
    assert len(df) == 1 and len(c.session.calls) == 1
    p = c.session.calls[0][1]
    assert p["firstdate"] == "2026-01-01T05:00:00Z"      # naive start read as ET midnight
    assert p["unit"] == "Daily"


def test_auth_url_requests_market_data_scopes_only():
    u = ts.auth_url("abc", "http://localhost:3001/callback")
    assert "MarketData" in u and "ReadAccount" in u and "Trade" not in u.replace("TradeStation", "")
    assert u.startswith("https://signin.tradestation.com/authorize?")


def test_quotes_frame():
    c = client([{"Quotes": [{"Symbol": "MSFT", "Last": "410.25", "Bid": "410.2", "Ask": "410.3",
                             "TradeTime": "2026-11-06T20:59:59Z"}], "Errors": []}])
    q = c.quotes("MSFT")
    assert q.loc[0, "Last"] == 410.25
    assert str(q.loc[0, "TradeTimeET"]) == "2026-11-06 15:59:59-05:00"


def test_bad_unit():
    import pytest
    with pytest.raises(ValueError):
        client([]).bars("SPY", unit="Hourly")
