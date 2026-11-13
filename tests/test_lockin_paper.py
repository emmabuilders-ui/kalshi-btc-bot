import json
import math

import numpy as np

from kalshi_btc_bot.lockin_paper import (CHECK_SECS, MAX_LAG, MIN_EDGE, LockinPaper, book_sides, edge, evaluate,
                                         inputs, limit_price, spread_ok)
from kalshi_btc_bot.settlement import settle_prob, sigma_1s

S = 1_800_000_000          # multiple of 900: a 15-minute market opens here
T = S + 900                # and closes here


def _sec(end, n=901, base=1000.0, last=None):
    """n seconds of lively prices ending at `end` (inclusive); the last 60 can be pinned to `last`."""
    t = np.arange(end - n + 1, end + 1)
    sec = base + 5.0 * np.sin(t / 7)
    if last is not None:
        sec[-61:] = last
    return sec


def test_inputs_lock_in_the_known_prints():
    sec = _sec(T - 30)
    sec[-31:] = 1002.0                                         # the 31 prints already in the average
    f = inputs(sec, T - 30, T, 1000.0)
    assert f["secs_left"] == 30 and f["known_n"] == 31 and f["known_avg"] == 1002.0
    assert f["brti"] == 1002.0 and f["sigma"] > 0
    assert abs(f["p_yes"] - settle_prob(1002.0, 1000.0, 30, sigma_1s(sec), 1002.0, 31)) < 1e-12
    assert f["p_yes"] > 0.95                                   # $2 above with half the prints locked: nearly sure
    # at 70 s left nothing is locked yet
    g = inputs(_sec(T - 70), T - 70, T, 1000.0)
    assert g["known_n"] == 0 and g["known_avg"] is None and g["settle_mean"] == g["brti"]
    flat = inputs(np.full(901, 1000.0), T - 30, T, 1000.0)
    assert flat["sigma"] == 0.0 and flat["p_yes"] is None


def test_locked_prints_can_outvote_the_spot():
    """BTC is below the strike now, but the 50 prints already made were well above it: YES still favoured."""
    sec = _sec(T - 10)
    sec[-51:] = 1010.0
    sec[-1] = 999.0
    f = inputs(sec, T - 10, T, 1000.0)
    assert f["known_n"] == 51 and f["brti"] == 999.0
    assert f["settle_mean"] > 1000.0 and f["p_yes"] > 0.9


def test_edge_subtracts_the_fee_and_evaluate_picks_the_side():
    assert abs(edge(0.90, 0.80) - (0.90 - 0.80 - 0.07 * 0.80 * 0.20)) < 1e-12
    f = {"p_yes": 0.92}
    sides = book_sides({"yes_bids": [[0.84, 100]], "no_bids": [[0.15, 100]]})     # YES ask 85c, NO ask 16c
    assert sides["yes"]["ask"] == 0.85 and sides["yes"]["bid"] == 0.84 and sides["no"]["ask"] == 0.16
    ev = evaluate(f, sides)
    assert ev["look"] == "yes" and ev["ask"] == 0.85 and abs(ev["prob"] - 0.92) < 1e-12
    assert ev["yes_edge"] > MIN_EDGE and ev["no_edge"] < 0
    rich = evaluate(f, book_sides({"yes_bids": [[0.89, 100]], "no_bids": [[0.10, 100]]}))   # YES ask 90c: 2 pts short
    assert rich["look"] is None and 0 < rich["yes_edge"] < MIN_EDGE
    no_side = evaluate({"p_yes": 0.05}, book_sides({"yes_bids": [[0.12, 50]], "no_bids": [[0.85, 50]]}))
    assert no_side["look"] == "no" and no_side["ask"] == 0.88
    # the cheap side is never bought, however big its edge looks: YES 40% vs a 25c ask is 15 points but not favoured
    cheap = evaluate({"p_yes": 0.40}, book_sides({"yes_bids": [[0.24, 50]], "no_bids": [[0.75, 50]]}))
    assert cheap["look"] is None and cheap["yes_edge"] > MIN_EDGE
    assert evaluate({"p_yes": None}, sides)["look"] is None
    assert evaluate(f, book_sides({"yes_bids": [], "no_bids": []}))["look"] is None


def test_limit_price_and_spread():
    asks = [[0.85, 5], [0.86, 10], [0.88, 100]]
    assert limit_price(0.95, asks) == 0.88                     # 0.95 - 0.88 - fee(0.88) = 6.3 pts: still fine
    assert limit_price(0.92, asks) == 0.86                     # 0.88 would leave only 3.3 pts
    assert limit_price(0.80, asks) == 0.0
    assert spread_ok(0.85, 0.84) and spread_ok(0.85, 0.83)
    assert not spread_ok(0.85, 0.82) and not spread_ok(0.85, None) and not spread_ok(None, 0.84)


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


def _run(bot, start, end, price=1004.0, lag=1, proxy_from=None):
    for t in range(S - 3600, start):
        bot.series.brti[t] = 1000.0 + 5.0 * math.sin(t / 7)       # a lively hour: a real volatility estimate
    for t in range(start, end + 1):
        if proxy_from is not None and t >= proxy_from:
            bot.series.proxy[t] = price
        else:
            bot.series.brti[t] = price
    for t in range(start, end + 1):
        bot.wall = lambda t=t: t + lag
        bot.process(t, "brti" if t in bot.series.brti else "proxy")


def test_bot_buys_at_the_first_qualifying_check_and_settles(tmp_path):
    bot = LockinPaper(tmp_path)
    bot.k = FakeKalshi([[0.84, 500.0]], [[0.15, 500.0]], result="yes")      # YES ask 85c, bid 84c
    _run(bot, T - 120, T - 1)
    rows = (tmp_path / "signals.csv").read_text().splitlines()
    assert len(rows) == 2 and "BUY YES" in rows[1] and rows[1].split(",")[2] == "50"   # first check, 50 s left
    assert bot.k.calls == 1                                                  # later checks are not made
    tr = bot.unsettled[0]
    assert tr["side"] == "YES" and tr["contracts"] == 20 and abs(tr["avg_price"] - 0.85) < 1e-9
    assert tr["secs_left"] == 50 and tr["model_prob"] > 0.9 and tr["gap_pts"] >= 100 * MIN_EDGE
    assert tr["level"] == "lock-in at 50 s" and tr["locked_avg"] == 1004.0
    assert json.loads((tmp_path / "state.json").read_text())["looked"] == ["KXBTC15M-TEST-00"]
    bot.settle(T + 100)
    out = (tmp_path / "trades.csv").read_text().splitlines()
    assert len(out) == 2 and ",yes,1," in out[1]
    assert not (tmp_path / "skips.csv").exists()


def test_bot_keeps_checking_while_nothing_qualifies(tmp_path):
    bot = LockinPaper(tmp_path)
    bot.k = FakeKalshi([[0.94, 500.0]], [[0.04, 500.0]])                     # YES ask 95c: no 5-point edge
    _run(bot, T - 120, T - 1)
    rows = (tmp_path / "signals.csv").read_text().splitlines()[1:]
    assert len(rows) == len(CHECK_SECS) and all("no trade" in r for r in rows)
    assert [int(r.split(",")[2]) for r in rows] == list(CHECK_SECS)
    assert not bot.unsettled and bot.k.calls == len(CHECK_SECS)


def test_wide_spread_is_skipped_logged_and_does_not_use_the_look(tmp_path):
    bot = LockinPaper(tmp_path)
    bot.k = FakeKalshi([[0.80, 500.0]], [[0.15, 500.0]])                     # YES ask 85c, bid 80c: 5c spread
    _run(bot, T - 120, T - 1)
    rows = (tmp_path / "signals.csv").read_text().splitlines()[1:]
    assert len(rows) == len(CHECK_SECS) and all("spread 5c" in r for r in rows)
    skips = (tmp_path / "skips.csv").read_text().splitlines()
    assert len(skips) == 1 + len(CHECK_SECS) and ",20,0.85" in skips[1]      # would have filled 20 at 85c
    assert not bot.unsettled and not bot.looked


def test_late_or_proxy_checks_are_skipped(tmp_path):
    bot = LockinPaper(tmp_path)
    bot.k = FakeKalshi([[0.84, 500.0]], [[0.15, 500.0]])
    _run(bot, T - 120, T - 1, lag=MAX_LAG + 2)                               # the feed is far behind
    rows = (tmp_path / "signals.csv").read_text().splitlines()[1:]
    assert len(rows) == len(CHECK_SECS) and all("late" in r for r in rows) and bot.k.calls == 0
    bot2 = LockinPaper(tmp_path / "b")
    bot2.k = FakeKalshi([[0.84, 500.0]], [[0.15, 500.0]])
    _run(bot2, T - 120, T - 1, proxy_from=T - 100)                            # only exchange prices in the last minute
    rows = (tmp_path / "b" / "signals.csv").read_text().splitlines()[1:]
    assert len(rows) == len(CHECK_SECS) and all("proxy" in r for r in rows) and bot2.k.calls == 0


def test_bot_walks_the_book_only_while_the_edge_holds(tmp_path):
    bot = LockinPaper(tmp_path)
    bot.k = FakeKalshi([[0.84, 500.0]], [[0.15, 5.0], [0.14, 500.0]])        # 5 at 85c, then 86c
    _run(bot, T - 60, T - 1)
    tr = bot.unsettled[0]
    assert tr["contracts"] == 20 and 0.85 < tr["avg_price"] < 0.86
    bot2 = LockinPaper(tmp_path / "b")
    bot2.k = FakeKalshi([[0.84, 500.0]], [[0.15, 5.0], [0.02, 500.0]])       # the next level is 98c: no edge
    _run(bot2, T - 60, T - 1)
    assert bot2.unsettled[0]["contracts"] == 5


def test_restart_keeps_the_one_trade(tmp_path):
    bot = LockinPaper(tmp_path)
    bot.k = FakeKalshi([[0.84, 500.0]], [[0.15, 500.0]])
    _run(bot, T - 60, T - 50)
    assert len(bot.unsettled) == 1
    again = LockinPaper(tmp_path)
    again.k = FakeKalshi([[0.84, 500.0]], [[0.15, 500.0]])
    _run(again, T - 49, T - 1)
    assert again.k.calls == 0 and len(again.unsettled) == 1                  # open trade reloaded, no second buy


def test_missing_brti_history_is_logged_not_traded(tmp_path):
    bot = LockinPaper(tmp_path)
    bot.k = FakeKalshi([[0.84, 500.0]], [[0.15, 500.0]])
    for t in range(T - 400, T):                                              # only part of the volatility window
        bot.series.brti[t] = 1004.0
    for t in range(T - 60, T):
        bot.wall = lambda t=t: t + 1
        bot.process(t, "brti")
    rows = (tmp_path / "signals.csv").read_text().splitlines()[1:]
    assert rows and all("BRTI history missing" in r or "no volatility" in r for r in rows)
    assert not bot.unsettled and bot.k.calls == 0
