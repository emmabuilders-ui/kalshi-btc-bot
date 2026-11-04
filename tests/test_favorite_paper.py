from kalshi_btc_bot.favorite_paper import HIGH, FavoritePaper, favorite_ok


def test_price_window_to_the_cent():
    assert favorite_ok(0.75) and favorite_ok(0.84) and favorite_ok(0.8049)
    assert not favorite_ok(0.74) and not favorite_ok(0.85) and not favorite_ok(0.744) and not favorite_ok(None)


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


def _mk(tk, lo, hi, ask):
    return {"ticker": tk, "status": "active", "floor_strike": lo, "cap_strike": hi, "yes_ask_dollars": ask}


def _bot(tmp_path, markets, books):
    bot = FavoritePaper(tmp_path, None, None)
    bot.k = FakeKalshi(markets, books)
    close = 3600 * 1000
    for s in range(close - 3600, close):
        bot.brti[s] = 1000.0
    return bot, close


def test_buys_yes_in_window_first_minute_only_once(tmp_path):
    markets = [_mk("A", 975, 999.99, 0.80), _mk("B", 1000, 1024.99, 0.70), _mk("C", 1025, 1049.99, 0.85)]
    books = {"A": {"no_bids": [[0.20, 5.0], [0.17, 100.0], [0.10, 100.0]]}}   # YES asks 0.80 x5, 0.83 x100, 0.90
    bot, close = _bot(tmp_path, markets, books)
    bot.check(close - 240)                       # 4 minutes left: too early
    assert not bot.held and not bot.k.calls
    bot.check(close - 180)                       # 3 minutes left: A qualifies, B (70c) and C (85c) do not
    assert list(bot.held) == ["A"] and bot.k.calls == ["A"]
    tr = bot.held["A"]
    assert tr["side"] == "YES" and tr["contracts"] == 20 and abs(tr["avg_price"] - (5 * 0.80 + 15 * 0.83) / 20) < 1e-6
    bot.check(close - 120)                       # already held: no second buy
    assert bot.k.calls == ["A"]
    bot.settle(close + 61)
    rows = (tmp_path / "trades.csv").read_text().splitlines()
    assert len(rows) == 2 and ",yes,1," in rows[1]
    assert "A,YES,3,975,999.99" in (tmp_path / "signals.csv").read_text()


def test_no_fill_above_limit_is_logged_not_held(tmp_path):
    markets = [_mk("A", 975, 999.99, 0.84)]
    books = {"A": {"no_bids": [[0.15, 100.0]]}}  # best YES ask 0.85 > limit
    bot, close = _bot(tmp_path, markets, books)
    bot.check(close - 60)
    assert not bot.held and HIGH == 0.84
    assert ",0,," in (tmp_path / "signals.csv").read_text().splitlines()[1]


def test_dashboard_favorite_events(tmp_path):
    from kalshi_btc_bot.dashboard import favorite_events, history_events
    d = tmp_path / "favorite"
    d.mkdir()
    (d / "signals.csv").write_text(
        "time_utc,ticker,side,mins_left,floor,cap,brti,ask_snapshot,book_best,limit,filled,avg_price,book\n"
        "2026-11-04 15:57:00,A,YES,3,975,999.99,990,0.8,0.8,0.84,20,0.8,[]\n"
        "2026-11-04 16:58:00,B,YES,2,1000,1024.99,1010,0.82,0.82,0.84,20,0.82,[]\n")
    (d / "trades.csv").write_text(
        "time_utc,ticker,side,mins_left,contracts,avg_price,fee,cost,result,won,pnl\n"
        "2026-11-04 15:57:00,A,YES,3,20,0.8,0.23,16.23,yes,1,3.77\n")
    (d / "open_positions.json").write_text(
        '[{"time_utc": "2026-11-04 16:58:00", "ticker": "B", "side": "YES", "mins_left": 2, "contracts": 20, '
        '"avg_price": 0.82, "fee": 0.21, "cost": 16.61, "close_ts": 1793811600}]')
    ev = favorite_events(d)
    assert [(e["bot"], e["status"], e["lo"], e["hi"]) for e in ev] == [("Favorite", "won", 975, 999.99),
                                                                       ("Favorite", "open", 1000, 1024.99)]
    assert ev[0]["pnl"] == 3.77 and "ticker" not in ev[0] and ev[1]["close"] == 1793811600
    assert favorite_events(None) == [] and favorite_events(tmp_path / "missing") == []
    for k in ("range", "breakdown", "fastpush"):
        (tmp_path / k).mkdir()
    h = history_events({"range": tmp_path / "range", "breakdown": tmp_path / "breakdown",
                        "fastpush": tmp_path / "fastpush", "favorite": d})
    assert [e["bot"] for e in h["range"]] == ["Favorite", "Favorite"]
