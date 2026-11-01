"""Chance that price touches a target (up or down) within N bars.

Model: logistic regression on how far the target is in units of current
volatility, the volatility state (last 10 bars vs last 100), the window length,
the target's direction and whether the last bar moved toward it. Fitted on
Binance 1-minute BTC/ETH/SOL data, 2026-05-01 to 2026-10-31. Earlier versions of
the same design were calibrated within about 1 point on untouched 2026 data.

Inputs are price bars only (open, high, low, close), so any instrument with bar
history can be scored, though other markets should be refitted first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Fitted 2026-05-01..2026-10-31 (6 months), BTC+ETH+SOL. Feature order:
# ln_z, z, vol_ln, vol_ln*ln_z, ln_bars, ln_bars*ln_z, ln_bars*vol_ln, direction, trend
MODELS = {
    "15m": (-0.405510884786265, [-1.3652225356552687, -0.6091906342698061, 1.1295673839449634,
                                 0.1366582038825023, 0.05122973887307682, 0.040132138076163164,
                                 -0.16974742075434332, -0.04250940832875883, 0.00785685685504035]),
    "1h": (-0.40784352192671375, [-1.221687848268137, -0.6513080005521087, 0.8180665239418126,
                                  0.14077243739148393, 0.06449202948913679, -0.02204525201871758,
                                  -0.10880749907644598, -0.06098658541988178, 0.02158794577715944]),
    "4h": (-0.46454444994788036, [-1.2911982102737014, -0.5797576699124184, 0.7949958942493944,
                                  0.42830555236379053, 0.03765100580531538, -0.047167183124767954,
                                  -0.15891746244145294, -0.07172265112060684, 0.03891189956514083]),
}
FIT_PERIOD = "2026-05-01 to 2026-10-31"
Z_RANGE = (0.5, 3.0)       # target distances the model was fitted on (volatility units)
BARS_RANGE = (1, 24)
TF_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400}


@dataclass
class VolState:
    price: float
    sigma: float          # std of the last 100 bars' log returns
    ratio: float          # last-10-bar volatility / last-100-bar volatility
    last_dir: int         # +1 last bar up, -1 down, 0 flat


@dataclass
class TouchResult:
    direction: int
    target_price: float
    pct: float
    z: float
    prob: float
    outside_fit: bool


def vol_state(closes) -> VolState:
    """Volatility state from completed bar closes, oldest first (needs 101+)."""
    c = np.asarray(closes, dtype=float)
    if len(c) < 101:
        raise ValueError(f"need at least 101 completed bars, got {len(c)}")
    r = np.diff(np.log(c))
    s100 = float(np.std(r[-100:], ddof=1))
    s10 = float(np.std(r[-10:], ddof=1))
    if s100 <= 0:
        raise ValueError("price has not moved in the last 100 bars")
    return VolState(price=float(c[-1]), sigma=s100, ratio=s10 / s100, last_dir=int(np.sign(r[-1])))


def touch_probability(tf: str, vs: VolState, target_price: float, bars: int) -> TouchResult:
    if tf not in MODELS:
        raise ValueError(f"timeframe must be one of {sorted(MODELS)}")
    if not BARS_RANGE[0] <= bars <= BARS_RANGE[1]:
        raise ValueError(f"bars must be {BARS_RANGE[0]}-{BARS_RANGE[1]}")
    if target_price <= 0 or target_price == vs.price:
        raise ValueError("target must be a positive price different from the current price")
    direction = 1 if target_price > vs.price else -1
    dist = abs(math.log(target_price / vs.price))
    z = dist / (vs.sigma * math.sqrt(bars))
    zc = min(max(z, 0.25), 8.0)
    lz, lw = math.log(zc), math.log(bars)
    v = max(min(math.log(max(vs.ratio, 1e-9)), 3.0), -3.0)
    trend = 1.0 if vs.last_dir == direction else 0.0
    x = [lz, zc, v, v * lz, lw, lw * lz, lw * v, direction, trend]
    b0, w = MODELS[tf]
    p = 1.0 / (1.0 + math.exp(-(b0 + sum(a * b for a, b in zip(w, x)))))
    if z < 0.25:  # very close targets: blend toward certain instead of flattening
        p = 1.0 - (1.0 - p) * (z / 0.25)
    return TouchResult(direction, target_price, 100 * (target_price / vs.price - 1), z, p,
                       outside_fit=not Z_RANGE[0] <= z <= Z_RANGE[1])


# ---------- live bars (Coinbase public candles, no key) ----------

COINBASE = "https://api.exchange.coinbase.com/products/{product}/candles"


def fetch_closes(coin: str, tf: str, session=None) -> list[float]:
    """Completed bar closes, oldest first. 4h bars are built from 1h candles."""
    import time

    import requests

    s = session or requests.Session()
    gran = 3600 if tf == "4h" else TF_SECONDS[tf]
    need = 440 if tf == "4h" else 110
    rows: dict[int, float] = {}
    end = int(time.time()) // gran * gran          # drop the bar still forming
    while len(rows) < need:
        start = end - 300 * gran
        r = s.get(COINBASE.format(product=f"{coin.upper()}-USD"),
                  params={"granularity": gran, "start": start, "end": end - 1}, timeout=10)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        for t, _lo, _hi, _op, close, _vol in batch:
            if t + gran <= end:
                rows[int(t)] = float(close)
        end = start
    ts = sorted(rows)
    if tf == "4h":                                   # close of each completed 4h bar (UTC 00,04,...)
        ts = [t for t in ts if (t + 3600) % 14400 == 0]
    return [rows[t] for t in ts]


def run_cli(coin: str, tf: str, bars: int, target: str | None) -> None:
    vs = vol_state(fetch_closes(coin, tf))
    state = "expanding" if vs.ratio > 1.15 else "calming" if vs.ratio < 0.85 else "normal"
    print(f"{coin.upper()} {tf}: price ${vs.price:,.2f}, volatility state {vs.ratio:.2f} ({state}), "
          f"normal {tf} move {100 * vs.sigma:.2f}%, last bar {'up' if vs.last_dir > 0 else 'down'}")
    print(f"Chance of touching within {bars} bar(s) ({bars * TF_SECONDS[tf] / 3600:g} h); model fitted {FIT_PERIOD}\n")
    if target:
        t = target.strip()
        if t.endswith("%"):
            pct = float(t[:-1])
            targets = [vs.price * (1 + pct / 100)]
        else:
            targets = [float(t.replace("$", "").replace(",", ""))]
    else:
        targets = [vs.price * (1 + s * p / 100) for p in (0.25, 0.5, 1, 1.5, 2, 3) for s in (1, -1)]
    print(f"{'target':>12} {'move':>8} {'chance':>7}")
    for tp in targets:
        r = touch_probability(tf, vs, tp, bars)
        flag = "  (outside fitted range, rough)" if r.outside_fit else ""
        print(f"{tp:>12,.2f} {r.pct:>+7.2f}% {100 * r.prob:>6.1f}%{flag}")
