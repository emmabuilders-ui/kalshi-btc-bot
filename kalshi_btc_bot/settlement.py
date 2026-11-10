"""Settlement-aware probabilities for Kalshi's Bitcoin contracts.

Kalshi settles on the CF Benchmarks BRTI averaged over the last 60 seconds before the close (the
mean of the 60 one-second prints T-60 .. T-1), not on the last print. Treating BRTI as a random
walk with a per-second log volatility `sigma`, and with `tau` seconds left:

* tau >= 60: the average's expected value is today's spot and its variance is
  sigma^2 * (tau - 60) for reaching the start of the settlement minute plus sigma^2 * 60/3 for the
  average itself, i.e. sigma^2 * (tau - 40). The right "time shrink" is 40 seconds.
* tau < 60: 61 - tau prints are already known. settlement = (known_n * known_avg + r * (spot +
  drift)) / 60 with r = 60 - known_n prints still to come, and the variance of the mean of r steps
  of a random walk is sigma^2 * (r + 1)(2r + 1) / (6r), scaled by (r / 60)^2.

Checked 2026-11-10 on real BRTI (Dec 2025 - Oct 2026): the 40-second version has the best Brier
score at every minute from 1 to 30, and under 60 seconds the locked-in part is what decides.
"""

from __future__ import annotations

import math

import numpy as np

SETTLE_SECONDS = 60      # prints in the settlement average
SHRINK_SECONDS = 40      # variance of the average = sigma^2 * (secs_left - 40) when secs_left >= 60
MIN_SIGMA = 1e-9


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def sigma_1s(prices) -> float:
    """Standard deviation of the one-second log returns of `prices` (gaps already filled)."""
    p = np.asarray(prices, dtype=float)
    p = p[~np.isnan(p)]
    if len(p) < 3:
        return 0.0
    lr = np.diff(np.log(p))
    m = float(lr.mean())
    return math.sqrt(max(float((lr * lr).mean()) - m * m, 0.0))


def effective_seconds(secs_left: float) -> float:
    """Variance of the settlement average, in 'seconds of random walk' (multiply by sigma^2)."""
    tau = max(float(secs_left), 0.0)
    if tau >= SETTLE_SECONDS:
        return tau - SHRINK_SECONDS
    r = max(tau - 1.0, 0.0)                      # prints still to come: t+1 .. T-1
    if r <= 0:
        return 1e-6
    return (r / SETTLE_SECONDS) ** 2 * (r + 1) * (2 * r + 1) / (6 * r)


def settle_mean(spot: float, secs_left: float, known_avg: float | None = None, known_n: int = 0) -> float:
    """Expected settlement value: spot, blended with the prints already made when under 60 s are left."""
    if secs_left >= SETTLE_SECONDS or not known_n or known_avg is None:
        return spot
    r = SETTLE_SECONDS - known_n
    return (known_n * known_avg + r * spot) / SETTLE_SECONDS


def settle_prob(spot: float, strike: float, secs_left: float, sigma: float,
                known_avg: float | None = None, known_n: int = 0) -> float:
    """Chance the settlement average ends at or above `strike`."""
    sd = spot * max(sigma, MIN_SIGMA) * math.sqrt(effective_seconds(secs_left))
    mean = settle_mean(spot, secs_left, known_avg, known_n)
    return norm_cdf((mean - strike) / max(sd, 1e-9))


def band_prob(spot: float, lo: float | None, hi: float | None, secs_left: float, sigma: float,
              known_avg: float | None = None, known_n: int = 0) -> float:
    """Chance the settlement average lands inside an hourly range band [lo, hi] (hi is the cap strike,
    e.g. 86399.99, so the band really runs to hi + 0.01). lo or hi None = open-ended band."""
    above_lo = 1.0 if lo is None else settle_prob(spot, lo, secs_left, sigma, known_avg, known_n)
    above_hi = 0.0 if hi is None else settle_prob(spot, hi + 0.01, secs_left, sigma, known_avg, known_n)
    return max(0.0, above_lo - above_hi)


def side_prob(p_yes: float, side: str) -> float:
    return p_yes if side.lower() == "yes" else 1.0 - p_yes


def taker_fee(price: float) -> float:
    return 0.07 * price * (1.0 - price)


def worth_it(p_side: float, ask: float | None) -> bool:
    """A side is worth buying only if its settlement-aware chance beats the ask plus the taker fee."""
    return ask is not None and p_side >= ask + taker_fee(ask)
