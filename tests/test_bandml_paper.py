import math

import numpy as np

from kalshi_btc_bot.bandml_paper import BOT_CONTRACTS, BandMLPaper, evaluate, features, quote_ok
from kalshi_btc_bot.model_gap_paper import DirectModel


def stub_model(p: float) -> DirectModel:
    """A model with no trees: always returns p."""
    return DirectModel({"features": ["mid"], "baseline": math.log(p / (1 - p)), "trees": [], "name": "stub"})


class FakeKalshi:
    key = "k"

    def __init__(self, markets, books):
        self.markets, self.books = markets, books
        self.calls = []

    def get(self, path, **params):
        return {"markets": self.markets}

    def orderbook(self, ticker):
        self.calls.append(ticker)
        return self.books.get(ticker, {"yes_bids": [], "no_bids": []})

    def result(self, ticker):
        return "yes"


def _mk(tk, lo, hi, bid, ask, kind="between"):
    return {"ticker": tk, "status": "active", "floor_strike": lo, "cap_strike": hi, "strike_type": kind,
            "yes_bid_dollars": bid, "yes_ask_dollars": ask}


def _bot(tmp_path, markets, books, p=0.80):
    bot = BandMLPaper(tmp_path, None, None, model=stub_model(p))
    bot.k = FakeKalshi(markets, books)
    close = 3600 * 1000
    for s in range(close - 2 * 3600, close):
        bot.brti[s] = 1000.0
    return bot, close


def test_default_is_ten_contracts():
    assert BOT_CONTRACTS == 10


def test_quote_ok():
    assert quote_ok(0.50, 0.55) and quote_ok(0.50, 0.60)
    assert not quote_ok(0.0, 0.55) and not quote_ok(None, 0.55) and not quote_ok(0.50, None)
    assert not quote_ok(0.40, 0.55) and not quote_ok(0.60, 0.60) and not quote_ok(0.5, 1.2)


def test_evaluate_needs_inside_and_gap():
    f = {"inside": 1.0}
    assert evaluate(0.80, f, 0.60)["buy"]
    assert not evaluate(0.80, {"inside": 0.0}, 0.60)["buy"]
    assert not evaluate(0.64, f, 0.60)["buy"]                 # 4 points minus the fee is not 5


def test_buys_inside_band_in_window_one_per_band(tmp_path):
    markets = [_mk("A", 975, 1024.99, 0.58, 0.60), _mk("B", 1100, 1149.99, 0.58, 0.60)]
    books = {"A": {"no_bids": [[0.40, 4.0], [0.38, 100.0], [0.10, 100.0]]}}   # YES asks 0.60 x4, 0.62 x100, 0.90
    bot, close = _bot(tmp_path, markets, books)
    bot.check(close - 25 * 60)
    bot.check(close - 6 * 60)
    assert not bot.held and not bot.k.calls                    # outside the 24..7 minute window
    bot.check(close - 10 * 60)
    assert list(bot.held) == ["A"]                              # B is outside the band: skipped, logged
    tr = bot.held["A"]
    assert tr["side"] == "YES" and tr["contracts"] == 10
    assert abs(tr["avg_price"] - (4 * 0.60 + 6 * 0.62) / 10) < 1e-6
    assert "BRTI outside the band" in (tmp_path / "skips.csv").read_text()
    bot.check(close - 9 * 60)
    assert bot.k.calls == ["A"]                                 # no second buy on the same band
    bot.settle(close + 61)
    rows = (tmp_path / "trades.csv").read_text().splitlines()
    assert len(rows) == 2 and ",yes,1," in rows[1]


def test_skips_open_ended_wide_spread_and_low_gap(tmp_path):
    markets = [_mk("A", 1100, None, 0.58, 0.60, "greater"), _mk("B", 975, 1024.99, 0.40, 0.60),
               _mk("C", 975, 1024.99, 0.58, 0.60)]
    bot, close = _bot(tmp_path, markets, {}, p=0.62)           # gap = 0.62 - 0.60 - fee < 5 points
    bot.check(close - 10 * 60)
    assert not bot.held and not bot.k.calls


def test_no_fill_when_book_is_above_the_gap_limit(tmp_path):
    markets = [_mk("A", 975, 1024.99, 0.58, 0.60)]
    books = {"A": {"no_bids": [[0.20, 50.0]]}}                  # best YES ask 0.80 > the 0.7x limit
    bot, close = _bot(tmp_path, markets, books)
    bot.check(close - 10 * 60)
    assert not bot.held
    row = (tmp_path / "signals.csv").read_text().splitlines()
    assert len(row) == 2 and not (tmp_path / "trades.csv").exists()   # the signal is logged, nothing is held


def test_features_flat_market():
    sec = np.full(3601, 1000.0)
    f = features(sec, 3600 * 1000 - 600, 3600 * 1000, 975.0, 1024.99, 0.58, 0.60)
    assert f["inside"] == 1.0 and f["secs_left"] == 600.0 and abs(f["mid"] - 0.59) < 1e-12
    assert f["ret_3600"] == 0.0 and abs(f["spread"] - 0.02) < 1e-12
    out = features(sec, 3600 * 1000 - 600, 3600 * 1000, 1100.0, 1149.99, 0.58, 0.60)
    assert out["inside"] == 0.0 and out["dist_lo"] < 0
