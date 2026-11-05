from kalshi_btc_bot.quiet_paper import QuietPaper, cushion, is_quiet, price_ok


def test_helpers():
    assert cushion(1010, 1000, 1024.99) == 10 and abs(cushion(1020, 1000, 1024.99) - 5) < 1e-9
    assert cushion(999, 1000, 1024.99) is None and cushion(1025.5, 1000, 1024.99) is None
    assert is_quiet(2.4, 10) and not is_quiet(2.5, 10) and is_quiet(-2.4, 10) and not is_quiet(0, 0)
    assert price_ok(0.61) and price_ok(0.95) and not price_ok(0.60) and not price_ok(0.96) and not price_ok(None)


class FakeKalshi:
    key = "k"

    def __init__(self, markets, books, result="yes"):
        self.markets, self.books, self.res, self.calls = markets, books, result, []

    def get(self, path, **params):
        return {"markets": self.markets}

    def orderbook(self, ticker):
        self.calls.append(ticker)
        return self.books.get(ticker, {"yes_bids": [], "no_bids": []})

    def result(self, ticker):
        return self.res


def _mk(tk, lo, hi, bid, ask):
    return {"ticker": tk, "status": "active", "floor_strike": lo, "cap_strike": hi, "yes_bid_dollars": bid, "yes_ask_dollars": ask}


def _bot(tmp_path, markets, books, path):
    bot = QuietPaper(tmp_path, None, None)
    bot.k = FakeKalshi(markets, books)
    close = 3600 * 1000
    for s in range(close - 3600, close):
        bot.brti[s] = path(s - close)
    return bot, close


def test_buys_when_quiet_once_per_hour_and_settles(tmp_path):
    markets = [_mk("A", 975, 999.99, 0.05, 0.08), _mk("B", 1000, 1024.99, 0.75, 0.78)]
    books = {"B": {"no_bids": [[0.22, 5.0], [0.20, 100.0]]}}
    bot, close = _bot(tmp_path, markets, books, lambda s: 1010.0)      # flat at 1010, cushion 10
    bot.check(close - 19 * 60)                  # too early
    assert not bot.held
    bot.check(close - 10 * 60)
    assert list(bot.held) == ["B"] and abs(bot.held["B"]["avg_price"] - (5 * 0.78 + 15 * 0.80) / 20) < 1e-6
    bot.check(close - 9 * 60)
    assert bot.k.calls == ["B"]                 # one per hour
    bot.settle(close + 61)
    assert ",yes,1," in (tmp_path / "trades.csv").read_text().splitlines()[1]


def test_skips_when_btc_moved_too_much_or_price_outside_window(tmp_path):
    markets = [_mk("B", 1000, 1024.99, 0.75, 0.78)]
    books = {"B": {"no_bids": [[0.22, 100.0]]}}
    # moved $3 in the last 10 minutes with a $10 cushion: 0.3x, not quiet
    bot, close = _bot(tmp_path, markets, books, lambda s: 1010.0 + 3.0 * (s + 600) / 600 if s > -600 else 1010.0)
    bot.brti[close - 600] = 1007.0
    bot.check(close - 10 * 60)
    assert not bot.held
    bot, close = _bot(tmp_path / "b", [_mk("B", 1000, 1024.99, 0.94, 0.97)], books, lambda s: 1010.0)
    bot.check(close - 10 * 60)
    assert not bot.held
