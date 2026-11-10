import json
import math

import numpy as np

from kalshi_btc_bot.longshot_paper import (MIN_EDGE, PX_MAX, LongshotPaper, book_asks, edge, evaluate, fair, inputs,
                                           limit_price, volatility)

S = 1_800_000_000          # multiple of 900: a 15-minute market opens here


def test_volatility_and_inputs():
    sec = np.full(901, 1000.0)
    sec[-301:] = np.linspace(1000.0, 999.0, 301)            # the last 5 minutes slide $1
    f = inputs(sec, S + 60, S + 900, 1000.0)
    assert f["brti"] == 999.0 and f["strike"] == 1000.0
    assert abs(f["dist"] - math.log(999 / 1000)) < 1e-12
    assert f["secs_left"] == 840 and f["sig15"] > 0
    assert abs(f["scale"] - f["sig15"] * math.sqrt(800)) < 1e-12   # 840 s left less the 40 s settlement window
    assert abs(f["move_1sd"] - 999.0 * f["scale"]) < 1e-9
    flat = inputs(np.full(901, 1000.0), S + 60, S + 900, 1000.0)
    assert flat["sig15"] == 0.0 and volatility(np.full(901, 1000.0)) == 0.0


def test_fair_value_is_the_normal_chance_of_travelling_the_distance():
    f = {"dist": math.log(999 / 1000), "scale": 0.01, "brti": 999.0, "strike": 1000.0}
    z = -math.log(999 / 1000) / 0.01                         # YES must travel up
    assert abs(fair("yes", f) - 0.5 * math.erfc(z / math.sqrt(2))) < 1e-12
    assert abs(fair("yes", f) + fair("no", f) - 1.0) < 1e-12  # the two sides share the same curve
    assert fair("no", f) > 0.5 > fair("yes", f)               # BTC is below the strike, so NO is ahead
    at = {"dist": 0.0, "scale": 0.01, "brti": 1000.0, "strike": 1000.0}
    assert abs(fair("yes", at) - 0.5) < 1e-12
    far = {"dist": math.log(900 / 1000), "scale": 0.01, "brti": 900.0, "strike": 1000.0}
    assert fair("yes", far) < 1e-6                            # $100 away with a $10 move: hopeless


def test_edge_subtracts_the_taker_fee():
    assert abs(edge(0.25, 0.10) - (0.25 - 0.10 - 0.07 * 0.10 * 0.90)) < 1e-12
    assert edge(0.10, 0.10) < 0                               # a fair price still loses the fee


def test_evaluate_takes_only_the_cheap_side_with_a_real_chance():
    f = {"dist": -0.0005, "scale": 0.02, "brti": 999.5, "strike": 1000.0}   # BTC just under the strike
    asks = {"yes": [[0.12, 100]], "no": [[0.90, 100]]}
    ev = evaluate(f, asks)
    assert ev["look"] == "yes" and ev["yes_ask"] == 0.12 and ev["no_ask"] == 0.90
    assert abs(ev["fair"] - fair("yes", f)) < 1e-12 and ev["fair"] > 0.4    # nearly a coin flip
    assert abs(ev["zgap"] - 0.0005 / 0.02) < 1e-9
    assert ev["yes_edge"] > MIN_EDGE and ev["no_edge"] < 0
    rich = evaluate(f, {"yes": [[0.20, 100]], "no": [[0.82, 100]]})         # 20c is not a longshot
    assert rich["look"] is None and rich["yes_ask"] == 0.20
    far = {"dist": math.log(900 / 1000), "scale": 0.02, "brti": 900.0, "strike": 1000.0}
    assert evaluate(far, asks)["look"] is None                             # cheap but no chance
    thin = {"dist": -0.02, "scale": 0.02, "brti": 980.0, "strike": 1000.0}  # fair about 16%, ask 12c
    assert 0 < evaluate(thin, asks)["yes_edge"] < MIN_EDGE and evaluate(thin, asks)["look"] is None
    assert evaluate(f, {"yes": [], "no": []})["look"] is None               # empty book


def test_book_asks_and_limit():
    a = book_asks({"yes_bids": [[0.88, 5], [0.87, 50]], "no_bids": [[0.89, 7], [0.88, 40]]})
    assert a["yes"] == [[0.11, 7], [0.12, 40]] and a["no"] == [[0.12, 5], [0.13, 50]]
    assert limit_price(0.25, a["yes"]) == 0.12                 # both levels keep 8 points
    assert limit_price(0.20, a["yes"]) == 0.11                 # 0.12 would leave under 8 points
    assert limit_price(0.15, a["yes"]) == 0.0
    deep = [[0.14, 5], [0.16, 50]]
    assert limit_price(0.60, deep) == 0.14                     # 16c is no longer a longshot
    assert PX_MAX == 0.15


class FakeKalshi:
    key = None

    def __init__(self, yes_bids, no_bids, result="yes"):
        self.books = {"yes_bids": yes_bids, "no_bids": no_bids}
        self.res, self.calls = result, 0

    def open_market(self, now):
        o = int(now) // 900 * 900
        return {"ticker": "KXBTC15M-TEST-00", "strike": 1000.0, "open_ts": o, "close_ts": o + 900}

    def orderbook(self, ticker):
        self.calls += 1
        return self.books

    def result(self, ticker):
        return self.res


def _run(bot, end, price_after_open=999.5):
    for t in range(S - 2 * 3600, S):
        bot.series.brti[t] = 1000.0 + 5.0 * math.sin(t / 7)       # a lively hour: a real volatility estimate
    for t in range(S, end + 1):
        bot.series.brti[t] = price_after_open
    for t in range(S, end + 1):
        bot.wall = lambda t=t: t + 1
        bot.process(t, "brti")


def test_bot_buys_the_longshot_once_and_settles(tmp_path):
    bot = LongshotPaper(tmp_path)
    bot.k = FakeKalshi([[0.10, 500.0]], [[0.88, 500.0]], result="yes")      # YES ask 0.12, NO ask 0.90
    _run(bot, S + 300)
    rows = (tmp_path / "signals.csv").read_text().splitlines()
    assert len(rows) == 2 and "BUY YES" in rows[1]                          # the first check is the one look
    assert bot.k.calls == 1                                                  # later minutes are not checked again
    tr = bot.unsettled[0]
    assert tr["side"] == "YES" and tr["contracts"] == 20 and abs(tr["avg_price"] - 0.12) < 1e-9
    assert tr["secs_left"] == 840 and tr["model_prob"] > 0.4 and tr["gap_pts"] > 100 * MIN_EDGE
    assert json.loads((tmp_path / "state.json").read_text())["looked"] == ["KXBTC15M-TEST-00"]
    bot.settle(S + 1000)
    out = (tmp_path / "trades.csv").read_text().splitlines()
    assert len(out) == 2 and ",yes,1," in out[1]


def test_bot_keeps_checking_while_nothing_qualifies(tmp_path):
    bot = LongshotPaper(tmp_path)
    bot.k = FakeKalshi([[0.10, 500.0]], [[0.88, 500.0]])
    _run(bot, S + 200, price_after_open=900.0)                               # $100 under the strike: no chance
    rows = (tmp_path / "signals.csv").read_text().splitlines()[1:]
    assert len(rows) == 3 and all("no look" in r for r in rows)
    assert not bot.unsettled and bot.k.calls == 3


def test_bot_skips_when_the_price_is_not_a_longshot(tmp_path):
    bot = LongshotPaper(tmp_path)
    bot.k = FakeKalshi([[0.40, 500.0]], [[0.40, 500.0]])                     # both asks 60c
    _run(bot, S + 130)
    rows = (tmp_path / "signals.csv").read_text().splitlines()[1:]
    assert rows and all("no look" in r for r in rows) and not bot.unsettled


def test_bot_walks_the_book_only_while_the_edge_holds(tmp_path):
    bot = LongshotPaper(tmp_path)
    bot.k = FakeKalshi([[0.10, 500.0]], [[0.88, 5.0], [0.86, 500.0]])        # 5 at 12c, then 14c
    _run(bot, S + 130)
    tr = bot.unsettled[0]
    assert tr["contracts"] == 20 and 0.12 < tr["avg_price"] < 0.14            # fair is high, so 14c still qualifies
    bot2 = LongshotPaper(tmp_path / "b")
    bot2.k = FakeKalshi([[0.10, 500.0]], [[0.88, 5.0], [0.80, 500.0]])       # the next level is 20c: not a longshot
    _run(bot2, S + 130)
    assert bot2.unsettled[0]["contracts"] == 5


def test_restart_keeps_the_one_look(tmp_path):
    bot = LongshotPaper(tmp_path)
    bot.k = FakeKalshi([[0.10, 500.0]], [[0.88, 500.0]])
    _run(bot, S + 61)
    again = LongshotPaper(tmp_path)
    again.k = FakeKalshi([[0.10, 500.0]], [[0.88, 500.0]])
    _run(again, S + 200)
    assert again.k.calls == 0 and len(again.unsettled) == 1                  # open trade reloaded, no second buy


def test_missing_brti_history_is_logged_not_traded(tmp_path):
    bot = LongshotPaper(tmp_path)
    bot.k = FakeKalshi([[0.10, 500.0]], [[0.88, 500.0]])
    for t in range(S - 400, S + 130):                                        # only part of the volatility window
        bot.series.brti[t] = 999.5
    for t in range(S, S + 130):
        bot.wall = lambda t=t: t + 1
        bot.process(t, "brti")
    rows = (tmp_path / "signals.csv").read_text().splitlines()[1:]
    assert rows and all("BRTI history missing" in r or "no volatility" in r for r in rows)
    assert not bot.unsettled and bot.k.calls == 0
