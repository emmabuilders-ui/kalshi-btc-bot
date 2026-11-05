import numpy as np

from kalshi_btc_bot.range_paper import RangeModel, event_ticker, fill, price_limit, qualifies


def test_rule_thresholds():
    assert qualifies(0.86, 0.75) and not qualifies(0.849, 0.75)   # 75-89c needs +10
    assert qualifies(0.92, 0.90) and not qualifies(0.919, 0.90)   # 90c+ needs +2
    assert not qualifies(0.99, 0.74)                              # below the 75c floor
    assert not qualifies(1.0, None)


def test_price_limit_and_fill():
    assert price_limit(0.97) == 0.95
    assert price_limit(0.88) == 0.78
    # buying YES takes NO bids: NO bid 0.07 -> YES at 0.93
    n, avg = fill([[0.07, 5], [0.06, 100], [0.03, 50]], 20, 0.95)
    assert n == 20 and abs(avg - (5 * 0.93 + 15 * 0.94) / 20) < 1e-9
    assert fill([[0.02, 50]], 20, 0.95) == (0, None)


def test_event_ticker_uses_new_york_hour():
    # 2026-11-01 20:00 UTC = 15:00 New York (EST)
    assert event_ticker(1793563200) == "KXBTC-26NOV0115"


def test_model_is_a_probability_and_sensible():
    m = RangeModel()
    closes = 80000 * np.exp(np.cumsum(np.r_[0, np.random.default_rng(1).normal(0, 0.0005, 100)]))
    s100, vb = m.vol(closes)
    mid = m.p_inside(80050, 80000, 80099.99, 5, s100, vb)
    far = m.p_inside(80050, 81000, 81099.99, 5, s100, vb)
    assert 0 <= far < 0.05 < mid <= 1
    assert m.p_inside(80050, 80000, 80099.99, 25, s100, vb) < mid  # more time, less sure
    total = sum(m.p_inside(80050, lo, lo + 99.99, 10, s100, vb) for lo in range(78000, 82000, 100))
    assert abs(total - 1) < 0.02


def test_spread_filter_only_below_90c():
    from kalshi_btc_bot.range_paper import spread_ok
    assert spread_ok(0.80, 0.78) and spread_ok(0.80, 0.79) and spread_ok(0.80, 0.8)
    assert not spread_ok(0.80, 0.77) and not spread_ok(0.80, 0.75) and not spread_ok(0.80, None)
    assert spread_ok(0.90, 0.85) and spread_ok(0.95, None)        # 90c+ is not filtered
    assert not spread_ok(None, 0.5)


def test_check_skips_wide_spread_below_90c_only(tmp_path):
    import json

    from kalshi_btc_bot.range_paper import RangePaper

    class FakeKalshi:
        key = True

        def __init__(self, markets, books):
            self.markets, self.books, self.calls = markets, books, []

        def get(self, path, **kw):
            return {"markets": self.markets}

        def orderbook(self, tk):
            self.calls.append(tk)
            return self.books[tk]

    bot = RangePaper(tmp_path, None, None)
    close = 1793563200
    t = close - 2 * 60
    bot.price_at = lambda u: 80050.0
    bot.minute_closes = lambda u: 80050 * np.exp(np.cumsum(np.r_[0, np.random.default_rng(1).normal(0, 0.0001, 100)]))
    # band B: the in-band YES is near certain with 2 min left; its ask is set so each case lands where we want
    def band(tk, lo, hi, bid, ask):
        return {"ticker": tk, "status": "active", "floor_strike": lo, "cap_strike": hi, "strike_type": "between",
                "yes_bid_dollars": bid, "yes_ask_dollars": ask}
    tight = band("T", 80000, 80099.99, 0.78, 0.80)      # 75-89c, spread 2c: trades
    wide = band("W", 80000, 80099.99, 0.75, 0.80)       # 75-89c, spread 5c: skipped
    high = band("H", 80000, 80099.99, 0.85, 0.90)       # 90c+, spread 5c: still trades
    books = {tk: {"no_bids": [[0.20, 500]], "yes_bids": [[0.78, 500]]} for tk in "TWH"}
    bot.k = FakeKalshi([tight, wide, high], books)
    bot.check(t)
    assert sorted(bot.held) == ["H", "T"] and "W" not in bot.k.calls
    skipped = (tmp_path / "skipped.csv").read_text().splitlines()
    assert len(skipped) == 2 and skipped[1].startswith("2026-11-01 19:58:00,W,YES,2,0.8,0.75,")
    assert json.loads((tmp_path / "open_positions.json").read_text())[0]["ticker"] in ("T", "H")
