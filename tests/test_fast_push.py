import numpy as np

from kalshi_btc_bot.fast_push import (FastPushPaper, FPSession, all_levels, minute_vol, price_ok,
                                      push_reading)


def test_price_bands():
    assert price_ok(0.30) == (True, 0.4999)
    assert price_ok(0.50)[0] is False and price_ok(0.60)[0] is False and price_ok(0.74)[0] is False
    assert price_ok(0.75) == (True, 0.96) and price_ok(0.96) == (True, 0.96)
    assert price_ok(0.97)[0] is False and price_ok(None)[0] is False


def test_push_reading_sign():
    v = 0.001  # 0.1% per minute -> normal 15-s move 0.05%
    assert abs(push_reading(1001.0, 1000.0, v, "yes") - np.log(1.001) / 0.0005) < 1e-9
    assert push_reading(1001.0, 1000.0, v, "no") < 0
    assert minute_vol(np.full(50, 100.0)) is None


def test_levels_sides_and_gap():
    o = 10 * 86400 + 3600  # 01:00 UTC
    n = 2 * 86400
    b = np.full(n, 1000.0)
    b[-1200] = 980.0   # inside prev 60m, outside prev 15m
    b[-600] = 1003.0   # prev15 high only $3 above open -> no level
    lv = all_levels(b, o)
    assert lv["prev60_low"] == ("no", 980.0)
    assert "prev15_high" not in lv
    assert "round1000_high" not in lv and "round500_low" not in lv  # 1000 is the open itself
    for side, p in lv.values():
        assert (side == "no" and p < 990) or (side == "yes" and p > 1010)


class FakeKalshi:
    key = None

    def __init__(self, yes_bids, no_bids, result="no"):
        self.books = {"yes_bids": yes_bids, "no_bids": no_bids}
        self.res = result

    def orderbook(self, ticker):
        return self.books

    def result(self, ticker):
        return self.res


def _bot(tmp_path, yes_bids, no_bids, result, noise=0.5):
    bot = FastPushPaper(tmp_path, max_lag=None)
    bot.k = FakeKalshi(yes_bids, no_bids, result)
    o = 1_000_200  # multiple of 60 is not needed
    rng = np.random.default_rng(1)
    for t in range(o - 7000, o + 700):  # calm market: noise of about $0.5 around 1000
        bot.series.brti[t] = 1000.0 + rng.normal(0, noise)
    return bot, o


def test_fast_break_buys_no_and_settles(tmp_path):
    bot, o = _bot(tmp_path, [[0.12, 500.0]], [[0.80, 500.0]], "no")  # NO ask 0.88
    bot.session = FPSession("T1", 1000.0, o, o + 900, 1000.0, {"sup": ("no", 980.0)})
    base = bot.series.brti[o + 699]
    # sharp drop through 980 at 780 s (120 s left)
    for k, sec in enumerate(range(700, 781)):
        bot.series.brti[o + sec] = base + (975.0 - base) * (k / 80) ** 4
    for sec in range(700, 781):
        bot.step(o + sec, bot.series.brti[o + sec])
    assert len(bot.unsettled) == 1
    tr = bot.unsettled[0]
    assert tr["side"] == "NO" and tr["contracts"] == 20 and tr["avg_price"] == 0.88
    bot.settle(o + 1000)
    assert ",no,1," in (tmp_path / "trades.csv").read_text().splitlines()[1]


def test_slow_break_or_bad_price_skips(tmp_path):
    # slow drift through the level in a busy market: no fast push
    bot, o = _bot(tmp_path, [[0.12, 500.0]], [], "no", noise=10.0)
    bot.session = FPSession("T1", 1000.0, o, o + 900, 1000.0, {"sup": ("no", 980.0)})
    p = bot.series.brti[o + 699]
    for sec in range(700, 781):
        p = p - (p - 974.0) / max(1, 781 - sec)  # straight line to 974
        bot.series.brti[o + sec] = p
        bot.step(o + sec, p)
    assert not bot.unsettled
    assert "skip: push" in (tmp_path / "signals.csv").read_text()
    # fast break but NO ask 0.62 (between 50c and 75c)
    bot2, o2 = _bot(tmp_path / "b", [[0.38, 500.0]], [], "no")
    bot2.session = FPSession("T2", 1000.0, o2, o2 + 900, 1000.0, {"sup": ("no", 980.0)})
    base = bot2.series.brti[o2 + 699]
    for k, sec in enumerate(range(700, 781)):
        bot2.series.brti[o2 + sec] = base + (975.0 - base) * (k / 80) ** 4
        bot2.step(o2 + sec, bot2.series.brti[o2 + sec])
    assert not bot2.unsettled
    assert "not under 50c or 75-96c" in (tmp_path / "b" / "signals.csv").read_text()


def test_break_outside_window_not_traded(tmp_path):
    bot, o = _bot(tmp_path, [[0.12, 500.0]], [], "no")
    bot.session = FPSession("T1", 1000.0, o, o + 900, 1000.0, {"sup": ("no", 980.0)})
    base = bot.series.brti[o + 399]
    for k, sec in enumerate(range(400, 481)):  # break at ~480 s: 420 s left
        bot.series.brti[o + sec] = base + (975.0 - base) * (k / 80) ** 4
        bot.step(o + sec, bot.series.brti[o + sec])
    assert not bot.unsettled
    assert "outside 1-3 min window" in (tmp_path / "breaks.csv").read_text()
