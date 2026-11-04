import numpy as np

from kalshi_btc_bot.late_breakdown import (Series, fill_no, hard_levels, kalshi_fee, ma_stack,
                                           session_levels)


def _walk(points, n_per_leg=600):
    out = []
    for a, b in zip(points, points[1:]):
        out.extend(np.linspace(a, b, n_per_leg, endpoint=False))
    return np.array(out)


def test_hard_bottom_needs_two_turns_in_40_band():
    # two dips to ~100 and ~120 (within $40), one to 300, swings of $200 between
    b = _walk([500, 100, 400, 120, 450, 300, 600, 600])
    lv = hard_levels(b, len(b))
    assert len(lv) == 1 and abs(lv[0] - 110) < 1


def test_session_levels_need_10_gap():
    b = np.r_[np.full(2 * 86400 - 900, 1000.0), np.linspace(1000, 995, 900)]
    assert "15min" not in session_levels(b, len(b))  # prev low only $5 below open
    b[-500] = 980
    assert abs(session_levels(b, len(b))["15min"] - 980) < 1e-9


def test_ma_stack():
    down = np.linspace(2000, 1000, 90000)
    assert ma_stack(down, len(down) - 1) == "down"
    assert ma_stack(down[::-1].copy(), len(down) - 1) == "up"


def test_fee_and_fill():
    assert kalshi_fee(10, 0.90) == 0.07  # 0.063 rounds up
    n, avg = fill_no([[0.10, 5], [0.08, 100]], 20, 0.96)  # NO at 0.90 x5, then 0.92
    assert n == 22 and abs(avg - (5 * 0.90 + 17 * 0.92) / 22) < 1e-9
    assert fill_no([[0.02, 50]], 20, 0.96)[0] == 0  # NO at 98c is above the cap


def test_series_prefers_brti_and_corrects_proxy():
    s = Series()
    s.proxy = {1: 105.0, 2: 106.0, 3: 107.0}
    s.brti = {1: 100.0, 2: 101.0}
    s.recalibrate()
    assert s.bias == 5.0 and s.value(1) == 100.0 and s.value(3) == 102.0
    assert list(s.array(1, 5)) == [100.0, 101.0, 102.0, 102.0]


def test_step_buys_no_on_late_break_and_settles(tmp_path):
    from kalshi_btc_bot.late_breakdown import LateBreakdownPaper, Session

    class FakeKalshi:
        key = None
        def orderbook(self, ticker):
            return {"yes_bids": [[0.10, 500.0], [0.09, 500.0]]}
        def result(self, ticker):
            return "no"

    bot = LateBreakdownPaper(tmp_path)
    bot.k = FakeKalshi()
    o = 1_000_000
    for t in range(o - 90000, o):  # rising day, so the MA stack is not pointing down
        bot.series.brti[t] = 900.0 + 100.0 * (t - o + 90000) / 90000
    bot.session = Session("T1", strike=1000.0, open_ts=o, close_ts=o + 900, s0=1000.0,
                          levels={"hard": 990.0})
    for sec in range(0, 790):
        price = 1000.0 if sec < 778 else 984.0  # breaks $5 under 990 at 778 s -> entry at 780 s
        bot.series.brti[o + sec] = price
        bot.step(o + sec, price)
    assert len(bot.unsettled) == 1
    tr = bot.unsettled[0]
    assert tr["contracts"] == 22 and tr["avg_price"] == 0.9 and tr["secs_left"] == 120
    bot.settle(o + 1000)
    rows = (tmp_path / "trades.csv").read_text().splitlines()
    assert len(rows) == 2 and ",no,1," in rows[1]


def test_step_buys_yes_on_late_breakout(tmp_path):
    from kalshi_btc_bot.late_breakdown import LateBreakdownPaper, Session

    class FakeKalshi:
        key = None
        def orderbook(self, ticker):
            return {"yes_bids": [[0.70, 500.0]], "no_bids": [[0.20, 500.0]]}  # YES ask 0.80
        def result(self, ticker):
            return "yes"

    bot = LateBreakdownPaper(tmp_path)
    bot.k = FakeKalshi()
    o = 1_000_000
    for t in range(o - 90000, o):  # falling day, so the MA stack is not pointing up
        bot.series.brti[t] = 1100.0 - 100.0 * (t - o + 90000) / 90000
    bot.session = Session("T2", strike=1000.0, open_ts=o, close_ts=o + 900, s0=1000.0,
                          levels={"hard_top": 1010.0})
    for sec in range(0, 790):
        price = 1000.0 if sec < 778 else 1016.0
        bot.series.brti[o + sec] = price
        bot.step(o + sec, price)
    assert len(bot.unsettled) == 1 and bot.unsettled[0]["side"] == "YES"
    assert bot.unsettled[0]["avg_price"] == 0.8 and bot.unsettled[0]["contracts"] == 25
    bot.settle(o + 1000)
    assert ",yes,1," in (tmp_path / "trades.csv").read_text().splitlines()[1]


def test_run_records_crash_in_log(tmp_path):
    import pytest
    from kalshi_btc_bot.late_breakdown import LateBreakdownPaper

    bot = LateBreakdownPaper(tmp_path)
    def boom(seconds=None):
        raise RuntimeError("feed died")
    bot._loop = boom
    with pytest.raises(RuntimeError):
        bot.run(1)
    log = (tmp_path / "crash.log").read_text()
    assert ": start" in log and "crashed:" in log and "RuntimeError: feed died" in log


def test_open_trade_survives_restart(tmp_path):
    from kalshi_btc_bot.late_breakdown import LateBreakdownPaper

    bot = LateBreakdownPaper(tmp_path)
    bot.unsettled.append({"time_utc": "x", "ticker": "T9", "side": "NO", "level": "hard", "secs_left": 120,
                          "contracts": 22, "avg_price": 0.9, "fee": 0.14, "cost": 19.94, "close_ts": 100,
                          "result": "", "won": "", "pnl": ""})
    bot._save_open()

    class FakeKalshi:
        key = None
        def result(self, ticker):
            return "no"

    again = LateBreakdownPaper(tmp_path)  # restarted process
    again.k = FakeKalshi()
    assert len(again.unsettled) == 1
    again.settle(1000)
    assert again.unsettled == [] and ",no,1," in (tmp_path / "trades.csv").read_text()
    assert LateBreakdownPaper(tmp_path).unsettled == []
