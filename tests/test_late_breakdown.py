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
