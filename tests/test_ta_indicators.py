import numpy as np
import pandas as pd

from kalshi_btc_bot.ta_indicators import (make_bars, relative_vigor_index,
                                          relative_volatility_index,
                                          directional_volatility_strength)


def _bars(n=60, seed=1):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    o = np.r_[100, c[:-1]]
    h = np.maximum(o, c) + rng.random(n)
    l = np.minimum(o, c) - rng.random(n)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c},
                        index=pd.date_range("2026-01-01", periods=n, freq="1min"))


def test_rvi_bounded_and_matches_manual():
    b = _bars()
    out = relative_vigor_index(b, 10)
    assert out["rvi"].dropna().abs().max() <= 1.0 + 1e-9
    # manual last value
    co, hl = (b.close - b.open).to_numpy(), (b.high - b.low).to_numpy()
    sw = lambda x, i: (x[i] + 2 * x[i-1] + 2 * x[i-2] + x[i-3]) / 6
    i = len(b) - 1
    num = sum(sw(co, i - k) for k in range(10)) / 10
    den = sum(sw(hl, i - k) for k in range(10)) / 10
    assert abs(out["rvi"].iloc[-1] - num / den) < 1e-12


def test_rvi_all_up_bars_is_one():
    n = 30
    o = np.arange(n, dtype=float)
    b = pd.DataFrame({"open": o, "close": o + 1, "high": o + 1, "low": o})
    assert abs(relative_vigor_index(b, 10)["rvi"].iloc[-1] - 1.0) < 1e-12


def test_no_lookahead():
    b = _bars()
    full = relative_vigor_index(b)["signal"]
    part = relative_vigor_index(b.iloc[:40])["signal"]
    assert np.allclose(full.iloc[:40].dropna(), part.dropna())
    c = b.close
    assert np.allclose(relative_volatility_index(c).iloc[:40].dropna(),
                       relative_volatility_index(c.iloc[:40]).dropna())


def test_rvol_range_and_direction():
    c = _bars().close
    r = relative_volatility_index(c).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    up = pd.Series(np.arange(60, dtype=float) ** 1.5)  # rising, widening stdev
    assert relative_volatility_index(up).iloc[-1] == 100.0
    assert relative_volatility_index(-up).iloc[-1] == 0.0


def test_make_bars_right_labelled():
    idx = pd.date_range("2026-01-01 00:00:00", periods=180, freq="1s")
    p = pd.Series(np.arange(180, dtype=float), index=idx)
    b = make_bars(p, "1m")
    assert b.index[0] == pd.Timestamp("2026-01-01 00:01:00")
    assert b.iloc[0].tolist() == [0, 59, 0, 59]


def test_dvs_blend_and_range():
    b = _bars(80)
    d = directional_volatility_strength(b).dropna()
    assert len(d) > 0 and d["dvs"].abs().max() <= 1.0 + 1e-9
    assert np.allclose(d["dvs"], (d["dorsey"] + d["vigor"]) / 2)
