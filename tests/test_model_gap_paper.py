import json
import math

import numpy as np

from kalshi_btc_bot.model_gap_paper import (CUSHION_MIN, DirectModel, ModelGapPaper, book_asks, evaluate, features, gap,
                                            limit_price)

S = 1_800_000_000          # multiple of 900: a 15-minute market opens here
F0 = {"dist": -0.002, "z": -1.5, "z60": -1.3, "secs_left": 540.0, "sig1": 6e-5, "sig5": 6e-5, "sig15": 6e-5, "sig60": 7e-5,
      "vol_ratio": 0.86, "vol_ratio_short": 1.0, "ret_10": 0.0, "ret_30": -1e-4, "ret_60": -2e-4, "ret_300": -6e-4,
      "ret_900": -1.5e-3, "dist_hi": -2.5e-3, "dist_lo": 0.0, "hour_utc": 14.0, "dow": 2.0}


def test_frozen_model_matches_reference_values():
    m = DirectModel.load()
    assert len(m.trees) == 250 and len(m.features) == 19
    assert abs(m.predict(F0) - 0.07542951664412502) < 1e-12              # below the strike: NO favoured
    up = dict(F0, dist=0.002, z=1.5, z60=1.3)
    assert abs(m.predict(up) - 0.9162749859822148) < 1e-12               # above the strike: YES favoured


def test_features_from_seconds():
    sec = np.full(3601, 1000.0)
    sec[-301:] = np.linspace(1000.0, 990.0, 301)                          # last 5 minutes slide $10
    t = S + 360
    f = features(sec, t, S, S + 900, 1000.0)
    assert abs(f["dist"] - math.log(990 / 1000)) < 1e-12
    assert abs(f["ret_300"] - math.log(990 / 1000)) < 1e-12 and f["ret_900"] == f["ret_300"]
    assert f["secs_left"] == 540 and f["sig60"] > 0 and f["z"] < 0
    assert abs(f["dist_lo"]) < 1e-12 and f["dist_hi"] < 0                 # at the low since the open
    assert f["hour_utc"] == float((S + 360) // 3600 % 24)


class Stub:
    features = []

    def __init__(self, p):
        self.p = p

    def predict(self, f):
        return self.p


def test_evaluate_gap_cushion_and_breakout():
    asks = {"yes": [[0.20, 100]], "no": [[0.80, 100]]}
    f = {"z": -1.5, "brti": 990.0}
    ev = evaluate(f, 0.10, asks, pre_hi=1010.0, pre_lo=995.0)
    assert ev["look"] == "no" and abs(ev["no_gap"] - gap(0.90, 0.80)) < 1e-12
    assert ev["cushion"] == 1.5 and ev["breakout"] and ev["condition"]
    ev = evaluate({"z": -0.5, "brti": 999.0}, 0.10, asks, 1010.0, 995.0)  # small cushion, inside the pre-open range
    assert ev["look"] == "no" and not ev["condition"]
    ev = evaluate({"z": -0.5, "brti": 990.0}, 0.10, asks, 1010.0, 995.0)  # breakout only
    assert ev["condition"] and ev["cushion"] < CUSHION_MIN
    assert evaluate(f, 0.17, asks, 1010.0, 995.0)["look"] is None          # gap under 3 points
    assert evaluate(f, 0.10, {"yes": [[0.35, 1]], "no": [[0.65, 1]]}, 1010.0, 995.0)["look"] is None  # ask under 70c


def test_book_asks_and_limit():
    a = book_asks({"yes_bids": [[0.18, 5], [0.17, 50]], "no_bids": [[0.80, 7]]})
    assert a["no"] == [[0.82, 5], [0.83, 50]] and a["yes"] == [[0.2, 7]]
    assert limit_price(0.90, a["no"]) == 0.83                           # 0.83 still leaves 3 points
    assert limit_price(0.865, a["no"]) == 0.82
    assert limit_price(0.85, a["no"]) == 0.0                            # no level leaves 3 points


class FakeKalshi:
    key = None

    def __init__(self, yes_bids, no_bids, result="no"):
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


def _run(bot, end, price_after_open=990.0):
    for t in range(S - 2 * 3600, S):
        bot.series.brti[t] = 1000.0 + 0.5 * math.sin(t / 7)
    for t in range(S, end + 1):
        bot.series.brti[t] = price_after_open
    for t in range(S, end + 1):
        bot.wall = lambda t=t: t + 1
        bot.process(t, "brti")


def test_bot_buys_no_once_and_settles(tmp_path):
    bot = ModelGapPaper(tmp_path, model=Stub(0.10))
    bot.k = FakeKalshi([[0.18, 500.0]], [[0.80, 500.0]], result="no")    # NO ask 0.82, YES ask 0.20
    _run(bot, S + 300)
    rows = (tmp_path / "signals.csv").read_text().splitlines()
    assert len(rows) == 2 and "BUY NO" in rows[1]                         # the first check (14 min left) is the one look
    assert bot.k.calls == 1                                               # later minutes are not checked again
    tr = bot.unsettled[0]
    assert tr["side"] == "NO" and tr["contracts"] == 20 and abs(tr["avg_price"] - 0.82) < 1e-9 and tr["secs_left"] == 840
    assert json.loads((tmp_path / "state.json").read_text())["looked"] == ["KXBTC15M-TEST-00"]
    bot.settle(S + 1000)
    out = (tmp_path / "trades.csv").read_text().splitlines()
    assert len(out) == 2 and ",no,1," in out[1]


def test_yes_look_is_logged_but_not_traded(tmp_path):
    bot = ModelGapPaper(tmp_path, model=Stub(0.95))
    bot.k = FakeKalshi([[0.80, 500.0]], [[0.18, 500.0]])                 # YES ask 0.82
    _run(bot, S + 200, price_after_open=1010.0)
    rows = (tmp_path / "signals.csv").read_text().splitlines()
    assert len(rows) == 2 and "YES side is not in the rule" in rows[1] and not bot.unsettled


def test_no_look_keeps_checking_each_minute(tmp_path):
    bot = ModelGapPaper(tmp_path, model=Stub(0.15))                      # NO 0.85 vs ask 0.82: gap under 3
    bot.k = FakeKalshi([[0.18, 500.0]], [[0.80, 500.0]])
    _run(bot, S + 200)
    rows = (tmp_path / "signals.csv").read_text().splitlines()[1:]
    assert len(rows) == 3 and all("no look" in r for r in rows)          # checks at 1, 2 and 3 minutes after the open
    assert not bot.unsettled and bot.k.calls == 3


def test_restart_keeps_the_one_look(tmp_path):
    bot = ModelGapPaper(tmp_path, model=Stub(0.10))
    bot.k = FakeKalshi([[0.18, 500.0]], [[0.80, 500.0]])
    _run(bot, S + 61)
    again = ModelGapPaper(tmp_path, model=Stub(0.10))
    again.k = FakeKalshi([[0.18, 500.0]], [[0.80, 500.0]])
    _run(again, S + 200)
    assert again.k.calls == 0 and len(again.unsettled) == 1               # open trade reloaded, no second buy


# ----------------------------------------------------------------------------- correction-model bot
from kalshi_btc_bot.model_gap_paper import CorrectionGapPaper, kalshi_inputs  # noqa: E402


def test_kalshi_inputs_from_book():
    k = kalshi_inputs({"yes_bid": 0.18, "yes_ask": 0.20}, 1000.0)
    assert abs(k["x"] - math.log(0.19 / 0.81)) < 1e-12 and abs(k["spread"] - 0.02) < 1e-12
    assert abs(k["log_cumvol"] - math.log1p(1000)) < 1e-12
    assert kalshi_inputs({"yes_bid": None, "yes_ask": 0.2}, 5) is None


class FakeKalshiVol(FakeKalshi):
    def orderbook(self, ticker):
        self.calls += 1
        yb, nb = self.books["yes_bids"], self.books["no_bids"]
        return {**self.books, "yes_bid": yb[0][0], "yes_ask": round(1 - nb[0][0], 4)}

    def volume(self, ticker):
        return 5000.0


class SeesKalshi(Stub):
    def predict(self, f):
        assert "x" in f and "spread" in f and "log_cumvol" in f
        return self.p


def test_correction_bot_buys_no_on_gap_alone(tmp_path):
    bot = CorrectionGapPaper(tmp_path, model=SeesKalshi(0.12))        # NO 0.88 vs ask 0.82: gap about 4.97 points
    bot.k = FakeKalshiVol([[0.18, 500.0]], [[0.80, 500.0]])
    _run(bot, S + 120, price_after_open=1000.0)                         # BTC at the strike: no cushion, no breakout
    rows = (tmp_path / "signals.csv").read_text().splitlines()
    assert "BUY NO" in rows[1] and bot.unsettled[0]["level"] == "gap only"
    assert bot.unsettled[0]["secs_left"] == 840


def test_correction_bot_needs_its_gap(tmp_path):
    bot = CorrectionGapPaper(tmp_path, model=SeesKalshi(0.10))        # NO 0.90 vs ask 0.82: about 6.9 points
    bot.k = FakeKalshiVol([[0.18, 500.0]], [[0.80, 500.0]])
    _run(bot, S + 120, price_after_open=1000.0)
    assert "BUY NO" in (tmp_path / "signals.csv").read_text().splitlines()[1]
    bot2 = CorrectionGapPaper(tmp_path / "b", model=SeesKalshi(0.17))  # NO 0.83 vs 0.82: under any gap
    bot2.k = FakeKalshiVol([[0.18, 500.0]], [[0.80, 500.0]])
    _run(bot2, S + 120, price_after_open=1000.0)
    assert not bot2.unsettled


def test_frozen_correction_model_matches_reference():
    m = DirectModel.load("model_gap_correction.json")
    assert len(m.trees) == 250 and m.features[-3:] == ["x", "spread", "log_cumvol"]
    f = dict(F0, x=-2.0, spread=0.01, log_cumvol=12.0)
    assert abs(m.predict(f) - 0.09095584151255706) < 1e-12
