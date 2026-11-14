import numpy as np

from kalshi_btc_bot.fivepm_paper import FivePmPaper, daily_markets, is_five_pm_close, market_kind

# 2026-11-16 22:00 UTC = 5 PM New York (EST): the daily market's slot. 21:00 UTC is the 4 PM hourly slot.
FIVE_PM = 1794002400
FOUR_PM = FIVE_PM - 3600


def _iso(ts):
    import datetime as dt
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mk(tk, lo, hi, ask, close=FIVE_PM, life=25 * 3600, bid="tight"):
    m = {"ticker": tk, "status": "active", "floor_strike": lo, "cap_strike": hi, "yes_ask_dollars": ask,
         "strike_type": "between", "open_time": _iso(close - life), "close_time": _iso(close)}
    if bid == "tight":
        bid = round(ask - 0.02, 2)
    if bid is not None:
        m["yes_bid_dollars"] = bid
    return m


def test_market_kind_from_life():
    assert market_kind(_mk("A", 1, 2, 0.5, life=3600)) is None            # hourly
    assert market_kind(_mk("A", 1, 2, 0.5, life=25 * 3600)) == "daily"
    assert market_kind(_mk("A", 1, 2, 0.5, life=169 * 3600)) == "weekly"
    assert market_kind({"ticker": "A"}) is None
    assert is_five_pm_close(FIVE_PM) and not is_five_pm_close(FOUR_PM)


def test_daily_markets_drops_hourly_tails_and_closed():
    ms = [_mk("D", 1, 2, 0.5), _mk("H", 1, 2, 0.5, life=3600), dict(_mk("C", 1, 2, 0.5), status="settled")]
    tail = _mk("T", None, 2, 0.5)
    assert [m["ticker"] for m in daily_markets(ms + [tail])] == ["D"]


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


def _bot(tmp_path, markets, books, close=FIVE_PM):
    bot = FivePmPaper(tmp_path, None, None)
    bot.k = FakeKalshi(markets, books)
    rng = np.random.default_rng(3)
    p = 80000.0
    for s in range(close - 2 * 3600, close):      # a quiet random walk so the model is well above 90c for the home band
        if s % 60 == 0:
            p *= float(np.exp(rng.normal(0, 0.00005)))
        bot.brti[s] = p
    return bot, close


def test_buys_yes_favorite_only_in_the_five_pm_hour(tmp_path):
    books = {"HOME": {"no_bids": [[0.08, 5.0], [0.07, 100.0]]}}             # YES asks 0.92 x5, 0.93 x100
    # the 4 PM hourly slot: same markets but an hourly life -> nothing happens
    bot, close = _bot(tmp_path, [], books, close=FOUR_PM)
    lo = round(bot.price_at(close - 600) / 250) * 250 - 125          # the $250 band around the price
    bot.k.markets = [_mk("HOME", lo, lo + 249.99, 0.92, close=FOUR_PM, life=3600)]
    bot.check(close - 600)
    assert not bot.held and not bot.k.calls
    # the 5 PM slot with the daily market: the home band at 92c qualifies (model >= 94c), the far band at 5c does not
    bot, close = _bot(tmp_path, [], books)
    lo = round(bot.price_at(close - 600) / 250) * 250 - 125
    bot.k.markets = [_mk("HOME", lo, lo + 249.99, 0.92), _mk("FAR", lo + 1500, lo + 1749.99, 0.05)]
    bot.check(close - 31 * 60)                    # 31 min left: before the window
    assert not bot.held
    bot.check(close - 600)
    assert list(bot.held) == ["HOME"] and bot.k.calls == ["HOME"]
    tr = bot.held["HOME"]
    assert tr["side"] == "YES" and tr["market"] == "daily" and tr["contracts"] == 20
    assert abs(tr["avg_price"] - (5 * 0.92 + 15 * 0.93) / 20) < 1e-6
    bot.check(close - 540)                        # already held: no second buy
    assert bot.k.calls == ["HOME"]
    bot.settle(close + 61)
    rows = (tmp_path / "trades.csv").read_text().splitlines()
    assert len(rows) == 2 and ",yes,1," in rows[1]
    assert f"HOME,YES,daily,10,{lo},{lo + 249.99}" in (tmp_path / "signals.csv").read_text()


def test_no_side_is_never_traded(tmp_path):
    # BRTI far outside the band: the model says ~0 for YES, so NO at 3c would qualify for the range bot. Not here.
    far = [_mk("FAR", 85000, 85249.99, 0.03)]
    bot, close = _bot(tmp_path, far, {"FAR": {"yes_bids": [[0.02, 500.0]]}})
    bot.check(close - 600)
    assert not bot.held and not bot.k.calls
