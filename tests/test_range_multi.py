import json

import numpy as np

from kalshi_btc_bot.range_multi import ACCOUNTS, LO, MID, HI, RangeMulti, next_stake, rebuilt_model, rule_limit
from kalshi_btc_bot.range_paper import RangeModel


def test_rule_limits():
    yes9099 = [(MID, HI, 0.02)]
    assert rule_limit(yes9099, 0.95, 0.93) == 0.93
    assert rule_limit(yes9099, 0.949, 0.93) is None          # needs +2c
    assert rule_limit(yes9099, 0.99, 0.89) is None           # below 90c
    assert rule_limit(yes9099, 1.0, 0.97) == 0.98            # limit = model - 2c
    bot_no = [(LO, MID, 0.10), (MID, HI, 0.02)]
    assert rule_limit(bot_no, 0.90, 0.80) == 0.80
    assert rule_limit(bot_no, 0.899, 0.80) is None           # 75-89c needs +10
    assert rule_limit(bot_no, 0.99, 0.88) == 0.89            # stays inside 75-89c group
    assert rule_limit([(LO, MID, 0.0)], 0.80, 0.80) == 0.80  # account 3: model >= ask
    assert rule_limit(yes9099, 0.99, None) is None


def test_doubling_stake():
    s = 7.0
    for won, want in [(True, 14), (True, 28), (True, 56), (True, 112), (True, 224), (True, 224), (False, 7)]:
        s = next_stake(s, won, 7.0, 224.0)
        assert s == want


def test_rebuilt_table_loads_and_differs():
    a, b = RangeModel(), rebuilt_model()
    closes = 80000 * np.exp(np.cumsum(np.r_[0, np.random.default_rng(2).normal(0, 0.0005, 100)]))
    s100, vb = a.vol(closes)
    pa = a.p_inside(80050, 80000, 80099.99, 5, s100, vb)
    pb = b.p_inside(80050, 80000, 80099.99, 5, s100, vb)
    assert 0 < pa < 1 and 0 < pb < 1 and pa != pb


class FakeKalshi:
    key = True

    def __init__(self, markets, books):
        self.markets, self.books = markets, books

    def get(self, path, **kw):
        return {"markets": self.markets}

    def orderbook(self, tk):
        return self.books[tk]

    def result(self, tk):
        return "yes"


def test_one_trade_per_hour_per_account_and_scaling(tmp_path):
    bot = RangeMulti(tmp_path, None, None, False, scale=1 / 3)
    assert [a.get("contracts") for a in bot.accts] == [27, 13, 19, 4, None]
    assert bot.accts[4]["base"] == 2.33 and bot.accts[4]["cap"] == 74.67
    close = 1793563200
    t = close - 2 * 60
    bot.price_at = lambda u: 80050.0
    bot.minute_closes = lambda u: 80050 * np.exp(np.cumsum(np.r_[0, np.random.default_rng(1).normal(0, 0.0001, 100)]))
    band = {"ticker": "KXBTC-26NOV0115-B80050", "status": "active", "floor_strike": 80000, "cap_strike": 80099.99,
            "strike_type": "between", "yes_bid_dollars": 0.89, "yes_ask_dollars": 0.90}
    wide = dict(band, ticker="KXBTC-26NOV0115-B80150", floor_strike=80100, cap_strike=80199.99,
                yes_bid_dollars=0.80, yes_ask_dollars=0.90)   # 10c spread: skipped
    bot.k = FakeKalshi([band, wide], {band["ticker"]: {"no_bids": [[0.10, 500]], "yes_bids": [[0.89, 500]]},
                                      wide["ticker"]: {"no_bids": [[0.10, 500]], "yes_bids": [[0.80, 500]]}})
    p = bot.models["rebuilt"].p_inside(80050, 80000, 80099.99, 2, *bot.model.vol(bot.minute_closes(t)))
    assert p >= 0.92                       # the in-band YES at 90c qualifies for accounts 1 and 5
    bot.check(t)
    held = {(v["account"], v["ticker"]) for v in bot.held.values()}
    assert (1, band["ticker"]) in held and (5, band["ticker"]) in held
    assert all(tk != wide["ticker"] for _, tk in held)
    n1 = len(bot.held)
    bot.check(t + 60)                      # same hour: no second trade for those accounts
    assert len(bot.held) == n1
    bot.settle(close + 120)
    st = json.loads((tmp_path / "status.json").read_text())
    a5 = st["accounts"]["5 YES 90-99c rebuilt +2c doubling"]
    assert a5["trades"] == 1 and a5["next_stake_dollars"] == 4.66
    assert len(ACCOUNTS) == 5


def test_range5_chart_data(tmp_path):
    from kalshi_btc_bot.dashboard_range5 import range5_data
    (tmp_path / "signals.csv").write_text("time_utc,account,ticker,floor,cap,filled\n2026-11-03 14:55:00,1,T1,80000,80099.99,27\n")
    (tmp_path / "trades.csv").write_text("time_utc,account,ticker,side,contracts,avg_price,cost,model,won,pnl\n"
                                         "2026-11-03 14:55:00,1,T1,YES,27,0.92,25.2,0.95,1,1.8\n")
    (tmp_path / "open_positions.json").write_text("[]")
    d = range5_data(tmp_path, [[1, 80000.0]])
    assert d["trades"][0]["account"] == 1 and d["trades"][0]["lo"] == 80000 and d["trades"][0]["won"]
    assert d["signals"] == 1 and d["unfilled"] == 0 and d["open"] == []
