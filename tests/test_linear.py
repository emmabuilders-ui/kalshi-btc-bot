from datetime import datetime

import numpy as np
import pytest

from kalshi_btc_bot.config import Config, ExitMethod
from kalshi_btc_bot.indicators import ExitBandSnapshot, TrailingStop
from kalshi_btc_bot.linear import (ET, LinearBacktester, LinearConfig, Segment, load_price_segments,
                                   resolve_instrument, rth_session)
from kalshi_btc_bot.synthetic import generate_linear


def _dip_then_rally(base=20_000.0, gap=10, rally=40.0, after=None):
    p = [base + (0.5 if i % 2 else -0.5) for i in range(600)]
    p += [base - 40] * 5 + [base - 10] * gap + [base + rally] * 200
    p += after if after is not None else [base + rally] * 900
    return np.array(p)


def _run(prices, t0=1_798_761_600, symbol="MNQ", lcfg=None, spread=0.25):
    seg = Segment(t0, prices, prices - spread / 2, prices + spread / 2)
    return LinearBacktester(Config(), lcfg or LinearConfig(), resolve_instrument(symbol)).run([seg])


def test_long_entry_pnl_uses_multiplier_spread_and_commission():
    bt = _run(_dip_then_rally())
    hold = [t for t in bt.trades if t.method == "hold_to_settlement"]
    assert len(hold) == 1
    t = hold[0]
    assert t.side == "long" and t.exit_reason == "time_stop"
    assert t.exit_t - t.entry_t == pytest.approx(900, abs=1)
    # bought at ask, sold at bid: P&L = (bid - ask) * $2/pt, commission 2 x 0.62
    assert t.gross_pnl == pytest.approx((t.exit_price - t.entry_price) * 2.0)
    assert t.exit_price - t.entry_price == pytest.approx(-0.25)
    assert t.commission == pytest.approx(1.24)
    assert bt.entries_identical()


def test_hard_stop_fires_on_collapse():
    after = [20_040.0] * 30 + [19_500.0] * 900
    bt = _run(_dip_then_rally(after=after))
    hold = next(t for t in bt.trades if t.method == "hold_to_settlement")
    assert hold.exit_reason == "hard_stop"
    assert hold.gross_pnl < 0


def test_fixed_point_stop_and_quantity():
    after = [20_040.0] * 30 + [20_030.0] * 900
    bt = _run(_dip_then_rally(after=after), lcfg=LinearConfig(hard_stop_points=5.0, quantity=3))
    hold = next(t for t in bt.trades if t.method == "hold_to_settlement")
    assert hold.exit_reason == "hard_stop" and hold.qty == 3


def test_lockout_keeps_entries_identical():
    p = np.concatenate([_dip_then_rally(after=[20_040.0] * 50), _dip_then_rally()[100:]])
    bt = _run(p)
    assert bt.entries_identical()
    assert bt.rejected.get("lockout", 0) > 0
    n = {sum(t.method == m.name for t in bt.trades) for m in Config().exit_methods}
    assert len(n) == 1


def test_short_side_mirrors_long():
    up = 2 * 20_000.0 - _dip_then_rally()       # mirror image: spike, then death cross
    assert not _run(up).trades                     # shorts are off by default
    bt = _run(up, lcfg=LinearConfig(allow_short=True))
    assert bt.trades and all(t.side == "short" for t in bt.trades)


def test_mirrored_trailing_stop():
    m = ExitMethod("x", candle_seconds=5, lookback=20, confirm_seconds=60)
    ts = TrailingStop(m, "short")
    snap = lambda t, p: ExitBandSnapshot(t, p, 110.0, 100.0, 90.0, True, p)
    assert not ts.update(snap(0, 89)) and ts.activated
    assert [ts.update(snap(5 * (i + 1), 101)) for i in range(12)] == [False] * 11 + [True]


def test_rth_session_and_etf_flatten():
    o = int(datetime(2026, 10, 27, 9, 30, tzinfo=ET).timestamp())    # a Tuesday
    assert rth_session(o) == (o, o + 6.5 * 3600)
    assert rth_session(o - 1) is None
    assert rth_session(int(datetime(2026, 10, 31, 12, 0, tzinfo=ET).timestamp())) is None  # Saturday
    close = o + int(6.5 * 3600)
    # The golden cross lands at index 661. Put it 16 minutes before the close: allowed,
    # and flattened 60 s before the close (before the 900 s time stop).
    p = _dip_then_rally(base=80.0, rally=0.5)
    p[600:605] = 79.5
    start = close - 16 * 60 - 661
    bt = _run(p, t0=start, symbol="TQQQ", spread=0.01)
    hold = next(t for t in bt.trades if t.method == "hold_to_settlement")
    assert hold.exit_reason == "time_stop" and hold.exit_t == close - 60 + 1
    # Same signal with 10 minutes left: rejected.
    bt2 = _run(p, t0=close - 10 * 60 - 661, symbol="TQQQ", spread=0.01)
    assert not bt2.trades and bt2.rejected == {"too_close_to_market_close": 1}


def test_gaps_split_segments(tmp_path):
    f = tmp_path / "price.csv"
    rows = ["timestamp,price"] + [f"{t},{100 + t}" for t in range(0, 100)]
    rows += [f"{t},{100 + t}" for t in range(103, 110)] + [f"{t},1" for t in range(5000, 5010)]
    f.write_text("\n".join(rows))
    segs = load_price_segments(f, max_ffill_s=5)
    assert [len(s.price) for s in segs] == [110, 10]
    assert segs[0].price[101] == 199          # short gap forward-filled


def test_synthetic_linear_end_to_end(tmp_path):
    generate_linear(tmp_path, "MNQ", 21_000, 0.25, hours=8, seed=2)
    bt = LinearBacktester(Config(), LinearConfig(), resolve_instrument("MNQ")).run(
        load_price_segments(tmp_path / "price.csv"))
    assert bt.trades and bt.entries_identical()
    report = bt.write_report(tmp_path / "out", "synthetic")
    assert "MNQ exit method comparison" in report.read_text()
