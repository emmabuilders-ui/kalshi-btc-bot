"""Probability bands around the BTC price for the 15-minute and 1-hour Kalshi windows (chart indicator, display only).

Two sets of bands, each at the central 50 / 80 / 95 % (quantiles 25/75, 10/90, 2.5/97.5):

* model  : the range model's table (s100 x sqrt(time left) -> empirical quantiles, frozen fit, no retraining).
           The last point of each fan uses the settlement-aware time (40 s off the time left, BRTI 60-s average).
* Kalshi : the same quantiles backed out of Kalshi's own prices.
    - 1 hour : the CDF of the band ladder. Each band's price is the bid/ask middle when the spread is 3c or less and the
               ask otherwise (a band with no bid and an ask of 3c or less counts half), then normalised over the ladder.
               A quantile that falls in an end band of the ladder is left out (None).
    - 15 min : one price at one strike only fixes ONE point of the CDF, so the model's shape is kept and its width is
               scaled until it reproduces Kalshi's price at the strike. No scale when the price is near 50c (strike next
               to BTC): the strike then says nothing about the width.

Fans: price at each horizon h up to the close. Model quantiles use the table at horizon h; the Kalshi hourly offsets are
scaled by sqrt(h / time left). Nothing here places orders or changes a bot.
"""

from __future__ import annotations

import math

import numpy as np

QS = (0.025, 0.10, 0.25, 0.75, 0.90, 0.975)
LEVELS = ((50, 0.25, 0.75), (80, 0.10, 0.90), (95, 0.025, 0.975))
FAN_STEPS = 12
MIN_TAIL = 0.08          # |price - 50c| needed before a 15-min price is used to size the width


def model_offsets(model, minutes: float, v: tuple[float, int], qs=QS, settle: bool = True, mult: float = 1.0) -> dict[float, float]:
    """log-return quantiles of the price `minutes` ahead; settle=True takes the settlement-average shrink off the time."""
    te = max(minutes - (model.shrink if settle else 0.0), 1 / 60)
    scale = mult * v[0] * math.sqrt(te)
    c = model.cdf[(model.bucket(minutes), v[1])]
    return {q: float(np.interp(q, model.qs, c)) * scale for q in qs}


def fan_from_model(model, price: float, left: float, v, mult: float = 1.0, steps: int = FAN_STEPS) -> dict:
    """{'h': [minutes ahead], 'q': {quantile: [prices]}} from now to the close."""
    hs = [left * k / steps for k in range(1, steps + 1)]
    out = {q: [] for q in QS}
    for k, h in enumerate(hs):
        off = model_offsets(model, h, v, settle=(k == steps - 1), mult=mult)
        for q in QS:
            out[q].append(price * math.exp(off[q]))
    return {"h": hs, "q": {f"{q:g}": [round(x, 2) for x in a] for q, a in out.items()}}


def implied_width_mult(model, price: float, strike: float, p_yes: float, left: float, v) -> float | None:
    """Factor on the model's width that reproduces Kalshi's chance of finishing at or above `strike`."""
    if not (MIN_TAIL <= abs(p_yes - 0.5) and 0.02 <= p_yes <= 0.98) or strike <= 0 or price <= 0:
        return None
    te = max(left - model.shrink, 1 / 60)
    c = model.cdf[(model.bucket(left), v[1])]
    x_star = float(np.interp(1 - p_yes, model.qs, c))      # standardised move that Kalshi's price puts at the strike
    x_k = math.log(strike / price) / (v[0] * math.sqrt(te))
    if x_star == 0 or x_k / x_star <= 0:
        return None
    f = x_k / x_star
    return f if 0.2 <= f <= 5 else None


def ladder_quantiles(bands: list[dict], qs=QS) -> dict[float, float | None] | None:
    """Quantiles of the CDF implied by contiguous range bands [{'lo','hi','yes_bid','yes_ask'}], or None if unusable."""
    rows = sorted((b for b in bands if b.get("lo") is not None and b.get("hi") is not None and b.get("yes_ask")),
                  key=lambda b: b["lo"])
    if len(rows) < 6:
        return None
    lo = np.array([b["lo"] for b in rows], float)
    hi = np.array([b["hi"] + 0.01 for b in rows], float)
    if (lo[1:] - hi[:-1] > 1.5).any():
        return None                                         # a gap in the ladder
    p = []
    for b in rows:
        bid, ask = b.get("yes_bid") or 0.0, b["yes_ask"]
        p.append((ask + bid) / 2 if bid > 0 and ask - bid <= 0.0301 else (ask / 2 if bid <= 0 and ask <= 0.03 else ask))
    p = np.array(p)
    if p.sum() <= 0:
        return None
    cum = np.append(0, np.cumsum(p)) / p.sum()
    edges = np.append(lo, hi[-1])
    out: dict[float, float | None] = {}
    for q in qs:
        out[q] = None if q <= cum[1] or q >= cum[-2] else float(np.interp(q, cum, edges))
    return out


def fan_from_ladder(price: float, left: float, qv: dict, steps: int = FAN_STEPS) -> dict:
    hs = [left * k / steps for k in range(1, steps + 1)]
    q = {}
    for k in QS:
        v = qv.get(k)
        q[f"{k:g}"] = [None if v is None else round(price + (v - price) * math.sqrt(h / left), 2) for h in hs]
    return {"h": hs, "q": q}


def window_bands(model, price: float, left: float, v, *, hourly_bands: list[dict] | None = None,
                 strike: float | None = None, yes_bid: float | None = None, yes_ask: float | None = None) -> dict:
    """Everything the chart and the table need for one window (`left` in minutes)."""
    left = max(left, 0.25)
    res = {"left": round(left, 2), "price": round(price, 2), "model": fan_from_model(model, price, left, v)}
    mq = {q: price * math.exp(o) for q, o in model_offsets(model, left, v).items()}
    res["model_end"] = {f"{q:g}": round(x, 2) for q, x in mq.items()}
    kal, how = None, None
    if hourly_bands is not None:
        qv = ladder_quantiles(hourly_bands)
        if qv:
            kal, how = fan_from_ladder(price, left, qv), "ladder"
            res["kalshi_end"] = {f"{q:g}": None if x is None else round(x, 2) for q, x in qv.items()}
    elif strike and yes_ask is not None:
        p = (yes_ask + yes_bid) / 2 if yes_bid is not None and yes_ask - yes_bid <= 0.05 else yes_ask
        f = implied_width_mult(model, price, strike, p, left, v)
        if f:
            kal, how = fan_from_model(model, price, left, v, mult=f), "strike"
            res["kalshi_end"] = {f"{q:g}": round(price * math.exp(o), 2) for q, o in model_offsets(model, left, v, mult=f).items()}
            res["width_mult"] = round(f, 3)
    if strike:
        res["strike"] = strike
        res["sd_to_strike"] = round(math.log(strike / price) / (v[0] * math.sqrt(max(left - model.shrink, 1 / 60))), 2)
    res["kalshi"], res["kalshi_how"] = kal, how
    return res
