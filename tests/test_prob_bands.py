"""Probability bands: monotone quantiles, ladder CDF, strike-fitted width."""
import math

from kalshi_btc_bot import prob_bands as pb
from kalshi_btc_bot.range_paper import RangeModel

M = RangeModel()
V = (0.0004, 1)


def test_model_fan_is_ordered_and_widens():
    f = pb.fan_from_model(M, 80000.0, 12, V)
    qs = ["0.025", "0.1", "0.25", "0.75", "0.9", "0.975"]
    for i in range(len(f["h"])):
        vals = [f["q"][q][i] for q in qs]
        assert vals == sorted(vals)
    assert f["q"]["0.975"][-1] - f["q"]["0.025"][-1] > f["q"]["0.975"][0] - f["q"]["0.025"][0]


def _ladder(centre, spread=0.0):
    out = []
    for i in range(14):
        lo = 79300 + 100 * i
        p = max(0.01, 0.30 - abs(lo + 50 - centre) / 100 * 0.05)
        out.append({"lo": lo, "hi": lo + 99.99, "yes_bid": p - spread, "yes_ask": p})
    return out


def test_ladder_quantiles_centre_and_order():
    q = pb.ladder_quantiles(_ladder(80000))
    assert abs(q[0.25] + q[0.75] - 160000) < 150 and q[0.1] < q[0.25] < q[0.75] < q[0.9]


def test_ladder_with_gap_or_short_is_unusable():
    lad = _ladder(80000)
    assert pb.ladder_quantiles(lad[:4]) is None
    del lad[6]
    assert pb.ladder_quantiles(lad) is None


def test_width_mult_reproduces_price():
    price, strike, left = 80000.0, 80050.0, 10.0
    f = pb.implied_width_mult(M, price, strike, 0.40, left, V)
    assert f and f > 1                       # Kalshi's 40c above a strike above the price: wider than the model's ~31c
    te = left - M.shrink
    x_k = math.log(strike / price) / (V[0] * math.sqrt(te))
    import numpy as np
    c = M.cdf[(M.bucket(left), V[1])]
    assert abs((1 - np.interp(x_k / f, c, M.qs)) - 0.40) < 0.01


def test_width_mult_none_near_fifty():
    assert pb.implied_width_mult(M, 80000.0, 80001.0, 0.50, 10, V) is None


def test_window_bands_both_kinds():
    h = pb.window_bands(M, 80000.0, 20, V, hourly_bands=_ladder(80000))
    assert h["kalshi_how"] == "ladder" and h["kalshi"]["h"] == h["model"]["h"]
    m = pb.window_bands(M, 80000.0, 8, V, strike=80100.0, yes_bid=0.30, yes_ask=0.32)
    assert m["kalshi_how"] == "strike" and "sd_to_strike" in m and m["sd_to_strike"] > 0
    assert pb.window_bands(M, 80000.0, 8, V)["kalshi"] is None
