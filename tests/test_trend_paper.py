from kalshi_btc_bot.trend_paper import (TrendPaper, centre_distance, inside, projection, slope_of, window_for)


def test_slope_and_projection():
    assert abs(slope_of([100 + 0.5 * i for i in range(15)]) - 0.5) < 1e-9
    assert abs(slope_of([100.0] * 15)) < 1e-12
    # 3 minutes left: 150 s to the middle of the settlement window
    assert abs(projection(78474.47, -0.644, 3) - (78474.47 - 0.644 * 150)) < 1e-9


def test_centre_distance_and_inside():
    assert abs(centre_distance(1012.5, 1000, 1024.99) - 0.0) < 0.01
    assert abs(centre_distance(1000, 1000, 1024.99) - 1.0) < 0.01
    assert abs(centre_distance(1006.25, 1000, 1024.99) - 0.5) < 0.01
    assert inside(1024.995, 1000, 1024.99) and not inside(1025.01, 1000, 1024.99) and not inside(999.9, 1000, 1024.99)


def test_windows_to_the_cent():
    assert window_for("tight", 0.70, 0.40) == (0.70, 0.84) and window_for("tight", 0.84, 0.0) == (0.70, 0.84)
    assert window_for("tight", 0.69, 0.1) is None and window_for("tight", 0.85, 0.1) is None
    assert window_for("tight", 0.75, 0.41) is None and window_for("tight", None, 0.0) is None
    assert window_for("broad", 0.61, 0.80) == (0.61, 0.80) and window_for("broad", 0.80, 0.79) == (0.61, 0.80)
    assert window_for("broad", 0.75, 0.81) is None
    assert window_for("broad", 0.81, 0.40) == (0.81, 0.90) and window_for("broad", 0.90, 0.1) == (0.81, 0.90)
    assert window_for("broad", 0.85, 0.41) is None and window_for("broad", 0.91, 0.0) is None


class FakeKalshi:
    key = "k"

    def __init__(self, markets, books, result="yes"):
        self.markets, self.books, self.res = markets, books, result
        self.calls = []

    def get(self, path, **params):
        return {"markets": self.markets}

    def orderbook(self, ticker):
        self.calls.append(ticker)
        return self.books.get(ticker, {"yes_bids": [], "no_bids": []})

    def result(self, ticker):
        return self.res


def _mk(tk, lo, hi, bid, ask):
    return {"ticker": tk, "status": "active", "floor_strike": lo, "cap_strike": hi,
            "yes_bid_dollars": bid, "yes_ask_dollars": ask}


def _bot(tmp_path, markets, books, rule="tight", slope=0.0):
    bot = TrendPaper(tmp_path, None, None, rule=rule)
    bot.k = FakeKalshi(markets, books)
    close = 3600 * 1000
    for s in range(close - 3600, close):
        bot.brti[s] = 1010.0 + slope * (s - (close - 3600))   # a straight line
    return bot, close


def test_tight_rule_buys_once_per_hour_when_price_and_projection_stay_inside(tmp_path):
    # price 1010 in band B (1000-1024.99, centre 1012.5, 20% from centre); A and C are neighbours
    markets = [_mk("A", 975, 999.99, 0.05, 0.08), _mk("B", 1000, 1024.99, 0.75, 0.78), _mk("C", 1025, 1049.99, 0.10, 0.12)]
    books = {"B": {"no_bids": [[0.22, 5.0], [0.20, 100.0], [0.10, 100.0]]}}   # YES asks 0.78 x5, 0.80 x100, 0.90
    bot, close = _bot(tmp_path, markets, books)
    bot.check(close - 21 * 60)                   # 21 minutes left: too early
    assert not bot.held and not bot.k.calls
    bot.check(close - 10 * 60)                   # 10 minutes left: B qualifies
    assert list(bot.held) == ["B"] and bot.k.calls == ["B"]
    tr = bot.held["B"]
    assert tr["side"] == "YES" and tr["contracts"] == 20 and abs(tr["avg_price"] - (5 * 0.78 + 15 * 0.80) / 20) < 1e-6
    bot.check(close - 9 * 60)                    # one trade per hour
    assert bot.k.calls == ["B"]
    bot.settle(close + 61)
    rows = (tmp_path / "trades.csv").read_text().splitlines()
    assert len(rows) == 2 and ",yes,1," in rows[1]
    sig = (tmp_path / "signals.csv").read_text().splitlines()
    assert sig[0].startswith("time_utc,ticker,side,mins_left,floor,cap,brti,slope_per_s,projection,centre_dist_pct")
    assert "B,YES,10,1000,1024.99,1010.0,0.0,1010.0,20.0,,0.78" in sig[1]   # model blank: flat test prices give no volatility


def test_skips_when_trend_leaves_the_band_or_price_is_too_far_from_centre(tmp_path):
    markets = [_mk("B", 1000, 1024.99, 0.75, 0.78)]
    books = {"B": {"no_bids": [[0.22, 100.0]]}}
    # falling $0.05/s: from 1010 with 10 min left the projection is 1010 - 0.05*570 = 981, outside the band
    bot, close = _bot(tmp_path, markets, books, slope=-0.05)
    bot.brti = {s: 1010.0 - 0.05 * (s - (close - 600)) for s in range(close - 3600, close)}
    bot.check(close - 10 * 60)
    assert not bot.held and not bot.k.calls
    # flat, but price at 1005 is 60% from the centre: tight rule skips, broad rule (<=80%) buys
    bot, close = _bot(tmp_path, markets, books)
    bot.brti = {s: 1005.0 for s in range(close - 3600, close)}
    bot.check(close - 10 * 60)
    assert not bot.held
    bot2, close = _bot(tmp_path / "b", markets, books, rule="broad")
    bot2.brti = {s: 1005.0 for s in range(close - 3600, close)}
    bot2.check(close - 10 * 60)
    assert list(bot2.held) == ["B"]


def test_skips_wide_spread_and_last_minutes(tmp_path):
    markets = [_mk("B", 1000, 1024.99, 0.70, 0.78)]                # 8c spread
    bot, close = _bot(tmp_path, markets, {"B": {"no_bids": [[0.22, 100.0]]}})
    bot.check(close - 10 * 60)
    assert not bot.held
    markets = [_mk("B", 1000, 1024.99, 0.75, 0.78)]
    bot, close = _bot(tmp_path / "c", markets, {"B": {"no_bids": [[0.22, 100.0]]}})
    bot.check(close - 3 * 60)                    # 3 minutes left: after the window
    assert not bot.held


def test_dashboard_trend_events(tmp_path):
    from kalshi_btc_bot.dashboard import favorite_events, history_events
    d = tmp_path / "trend-tight"
    d.mkdir()
    (d / "signals.csv").write_text(
        "time_utc,ticker,side,mins_left,floor,cap,brti,slope_per_s,projection,centre_dist_pct,ask_snapshot,book_best,limit,filled,avg_price,book\n"
        "2026-11-04 15:50:00,A,YES,10,975,999.99,990,0.1,993,10,0.78,0.78,0.84,20,0.78,[]\n")
    (d / "trades.csv").write_text(
        "time_utc,ticker,side,mins_left,contracts,avg_price,fee,cost,result,won,pnl\n"
        "2026-11-04 15:50:00,A,YES,10,20,0.78,0.24,15.84,yes,1,4.16\n")
    ev = favorite_events(d, "Trend tight")
    assert [(e["bot"], e["status"], e["lo"], e["hi"], e["pnl"]) for e in ev] == [("Trend tight", "won", 975, 999.99, 4.16)]
    for k in ("range", "breakdown", "fastpush"):
        (tmp_path / k).mkdir()
    h = history_events({"range": tmp_path / "range", "breakdown": tmp_path / "breakdown",
                        "fastpush": tmp_path / "fastpush", "trend_tight": d})
    assert [e["bot"] for e in h["range"]] == ["Trend tight"]
