import math

import numpy as np
import pytest

from kalshi_btc_bot.touch_prob import touch_probability, vol_state


def closes(sig_old=0.002, sig_new=0.002, n=200, seed=1):
    rng = np.random.default_rng(seed)
    r = np.concatenate([rng.normal(0, sig_old, n - 10), rng.normal(0, sig_new, 10)])
    return 80000 * np.exp(np.cumsum(r))


def test_needs_enough_bars():
    with pytest.raises(ValueError):
        vol_state(closes()[:50])


def test_farther_targets_are_less_likely():
    vs = vol_state(closes())
    ps = [touch_probability("1h", vs, vs.price * (1 + p / 100), 4).prob for p in (0.2, 0.5, 1, 2)]
    assert all(a > b for a, b in zip(ps, ps[1:]))


def test_longer_window_is_more_likely():
    vs = vol_state(closes())
    tp = vs.price * 1.01
    assert touch_probability("1h", vs, tp, 12).prob > touch_probability("1h", vs, tp, 2).prob


def test_expanding_volatility_raises_odds():
    calm = vol_state(closes(sig_new=0.001))
    hot = vol_state(closes(sig_new=0.006))
    # same target in units of 100-bar volatility
    t_calm = calm.price * math.exp(1.0 * calm.sigma * 2)
    t_hot = hot.price * math.exp(1.0 * hot.sigma * 2)
    assert touch_probability("15m", hot, t_hot, 4).prob > touch_probability("15m", calm, t_calm, 4).prob


def test_matches_fitted_scale():
    # 1 normal move away within 1 bar was touched about 20-25% of the time in the data
    vs = vol_state(closes())
    vs.ratio, vs.last_dir = 1.0, 1
    p = touch_probability("1h", vs, vs.price * math.exp(vs.sigma), 1).prob
    assert 0.15 < p < 0.35


def test_down_target_and_flags():
    vs = vol_state(closes())
    r = touch_probability("4h", vs, vs.price * 0.9, 1)
    assert r.direction == -1 and r.outside_fit and r.prob < 0.05


@pytest.mark.parametrize("tf", ["1m", "5m"])
def test_fast_timeframes(tf):
    vs = vol_state(closes())
    vs.ratio, vs.last_dir = 1.0, 1
    ps = [touch_probability(tf, vs, vs.price * math.exp(k * vs.sigma * 2), 4).prob for k in (0.5, 1, 2)]
    assert all(a > b for a, b in zip(ps, ps[1:])) and 0.05 < ps[1] < 0.6
