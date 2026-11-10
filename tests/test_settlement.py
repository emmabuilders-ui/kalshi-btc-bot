"""Settlement-aware probabilities: Kalshi settles on the 60-second BRTI average, not the last print."""
import math

import numpy as np

from kalshi_btc_bot.favorite_paper import MIN_CUSHION_SD, cushion_sd
from kalshi_btc_bot.range_multi import rebuilt_model
from kalshi_btc_bot.range_paper import RangeModel
from kalshi_btc_bot.settlement import (band_prob, effective_seconds, settle_mean, settle_prob,
                                       sigma_1s, worth_it)


def test_effective_seconds_is_40s_shrink_then_locked_in():
    assert effective_seconds(900) == 860 and effective_seconds(60) == 20
    # under a minute the random part shrinks fast: 30 s left is worth about 2.4 "seconds" of variance
    assert 2.0 < effective_seconds(30) < 3.0
    assert effective_seconds(10) < 0.2 and effective_seconds(1) > 0


def test_locked_in_average_can_make_a_spot_cross_meaningless():
    # 20 s left: 41 prints known, all $3 above the strike; spot has just dipped $1 under it
    p = settle_prob(spot=999.0, strike=1000.0, secs_left=20, sigma=1e-5, known_avg=1003.0, known_n=41)
    assert p > 0.99                                           # the average cannot fall back below the strike
    assert settle_mean(999.0, 20, 1003.0, 41) == (41 * 1003 + 19 * 999) / 60
    # the same spot with 120 s left and no locked-in part is a near coin flip
    q = settle_prob(999.0, 1000.0, 120, 3e-4)   # one sd of the average is about $2.70
    assert 0.3 < q < 0.5


def test_band_prob_and_sides():
    sigma = sigma_1s(1000 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 1e-4, 900))))
    assert 5e-5 < sigma < 2e-4
    inside = band_prob(1000.0, 975.0, 1024.99, 600, sigma)
    far = band_prob(1000.0, 1100.0, 1124.99, 600, sigma)
    assert 0.9 < inside <= 1 and far < 0.01
    assert abs(band_prob(1000.0, None, 1024.99, 600, sigma) + band_prob(1000.0, 1025.0, None, 600, sigma) - 1) < 1e-9
    assert worth_it(0.95, 0.90) and not worth_it(0.90, 0.90) and not worth_it(0.99, None)


def test_range_model_uses_40s_shrink_and_rebuilt_keeps_30s():
    m = RangeModel()
    assert abs(m.shrink - 40 / 60) < 1e-12 and abs(rebuilt_model().shrink - 0.5) < 1e-12
    closes = 80000 * np.exp(np.cumsum(np.r_[0, np.random.default_rng(1).normal(0, 0.0005, 100)]))
    s100, vb = m.vol(closes)
    # with 1 minute left the 40-s version is surer about a band the price sits in the middle of
    old = RangeModel(shrink_minutes=0.5)
    assert m.p_inside(80050, 80000, 80099.99, 1, s100, vb) > old.p_inside(80050, 80000, 80099.99, 1, s100, vb)
    assert math.isclose(m.p_inside(80050, 80000, 80099.99, 30, s100, vb),
                        old.p_inside(80050, 80000, 80099.99, 30, s100, vb), rel_tol=0.01)


def test_cushion_in_settlement_sds():
    assert cushion_sd(1000.0, 975.0, 1024.99, 10.0) == 2.5
    assert cushion_sd(1030.0, 975.0, 1024.99, 10.0) < 0           # outside the band
    assert cushion_sd(1000.0, 975.0, 1024.99, None) is None       # no volatility reading: no gate
    assert MIN_CUSHION_SD == 0.5
