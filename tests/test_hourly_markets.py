"""The KXBTC series holds the hourly range markets and, at the 5 PM ET close, Kalshi's daily or weekly
market under the same event ticker. No hourly-band bot may trade that one."""

from kalshi_btc_bot.range_paper import hourly_markets, is_daily_close, is_daily_event, is_hourly_market, market_life


def mk(open_time, close_time, status="active", **kw):
    return {"ticker": "KXBTC-26NOV1317-B68250", "open_time": open_time, "close_time": close_time, "status": status, **kw}


def test_hourly_market_lives_one_hour():
    m = mk("2026-11-13T20:00:00Z", "2026-11-13T21:00:00Z")
    assert market_life(m) == 3600
    assert is_hourly_market(m)


def test_daily_and_weekly_markets_are_not_hourly():
    daily = mk("2026-11-10T21:00:00Z", "2026-11-11T22:00:00Z")      # the Nov 11 5 PM market, listed a day ahead
    weekly = mk("2026-11-06T21:00:00Z", "2026-11-13T22:00:00Z")     # the Friday Nov 13 one, listed a week ahead
    assert not is_hourly_market(daily)
    assert not is_hourly_market(weekly)
    assert hourly_markets([daily, weekly], "KXBTC-26NOV1317") == []


def test_hourly_markets_keeps_open_hourly_only():
    good = mk("2026-11-13T20:00:00Z", "2026-11-13T21:00:00Z")
    closed = mk("2026-11-13T20:00:00Z", "2026-11-13T21:00:00Z", status="finalized")
    daily = mk("2026-11-12T21:00:00Z", "2026-11-13T22:00:00Z")
    assert hourly_markets([good, closed, daily], "KXBTC-26NOV1316") == [good]


def test_missing_times_do_not_block_trading():
    assert is_hourly_market({"ticker": "x", "status": "active"})
    assert hourly_markets([{"ticker": "x", "status": "active"}], "ev") == [{"ticker": "x", "status": "active"}]


def test_daily_event_flags():
    assert is_daily_event("KXBTC-26NOV1317-B68250")
    assert is_daily_event("KXBTC-26NOV1117-B74125")
    assert not is_daily_event("KXBTC-26NOV1316-B66250")
    assert not is_daily_event("KXBTC15M-26NOV141715-T68000")
    assert not is_daily_event(None)
    # 2026-11-13 22:00 UTC is 5 PM Eastern (EST); 21:00 UTC is 4 PM
    assert is_daily_close(1794607200)
    assert not is_daily_close(1794603600)
    assert not is_daily_close(None)
