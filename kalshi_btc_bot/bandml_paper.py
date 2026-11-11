"""Hourly range-band YES "Band ML" PAPER trader (read-only, no orders are sent).

Michael asked for it on 2026-11-11 ("build it ... only do 10 contracts") after the ML test on the hourly Kalshi
markets (research/hourly_ml/, strategy/hourly_ml/ in the project files). Rule:

* Model: gradient-boosted "Correction" model on the hourly range bands (KXBTC). Inputs: BRTI history (distance to the
  band edges in volatility units, the settlement-aware probability with the 40-s shrink, returns, volatility over
  1/5/15/60 minutes, distance from the 15-minute and 1-hour high and low, time left, UTC hour, weekday) plus Kalshi's
  own YES mid (as log-odds) and spread. Fit once on Dec 2025 - Aug 2026 and frozen (data/band_ml_correction.json).
* Checks: at each minute mark with 24 down to 7 minutes left before the hourly close, on the 14 bands nearest BRTI
  30 minutes before the close (the same ladder the range bot watches), "between" bands only.
* Buy YES when (a) BRTI is inside the band at that second, (b) the model's chance minus the YES ask minus the taker
  fee 0.07 x p x (1-p) is at least 5 points, and (c) the quote is two-sided with a spread of 10c or less.
* First qualifying minute per band, one position per band, held to settlement, 10 contracts, walking the book only
  while the gap stays at least 5 points. YES only; NO and outside-the-band looks are logged to skips.csv.

Backtest (frozen Dec-Aug fit for Sep-Oct, monthly walk-forward for Mar-Aug; one trade per band; win % vs average
price after fee), research/hourly_ml/: see docs/paper_test_decision_rule.md for the numbers and the retire rule.
The 7-24 minute window and the inside-band condition were chosen after seeing the data, so the paper test is the
only clean check. Backtest prices were one-minute candle asks; the paper bot uses the live order book.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone

import numpy as np

from .late_breakdown import kalshi_fee, log
from .model_gap_paper import DirectModel, book_asks, gap, limit_price
from .range_paper import CONTRACTS, N_BANDS, RangePaper, _f, _iso, band_mid, event_ticker, fill

MODEL_FILE = "band_ml_correction.json"
MAX_LEFT, MIN_LEFT = 24, 7      # minutes left, inclusive
MIN_GAP = 0.05
MAX_SPREAD = 0.10
HISTORY = 3601                  # seconds of BRTI the features need, ending at the second before the check
SETTLE_SHRINK = 40.0
BOT_CONTRACTS = 10


def _ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _logit(p: float) -> float:
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def features(sec: np.ndarray, t: int, close: int, lo: float, hi: float, yes_bid: float, yes_ask: float) -> dict:
    """Model inputs at minute mark t. `sec` holds 1-second BRTI for [t-3601, t-1] (3601 values, gaps filled): the
    spot is the print of the previous whole second, as in the research replay (research/hourly_ml/scripts/feat.py)."""
    sec = np.asarray(sec, float)
    lb = np.log(sec)
    lr = np.diff(lb)
    spot = float(sec[-1])

    def sd(w):
        r = lr[-w:]
        m = float(r.mean())
        return math.sqrt(max(float((r * r).mean()) - m * m, 1e-14))

    v = {w: sd(w) for w in (60, 300, 900, 3600)}
    secs = float(close - t)
    teff = max(secs - SETTLE_SHRINK, 1.0)
    sd_dollars = spot * v[900] * math.sqrt(teff)
    hi2 = hi + 0.01
    f = {"secs_left": secs, "vol_60": v[60], "vol_300": v[300], "vol_900": v[900], "vol_3600": v[3600]}
    f["dist_lo"] = math.log(spot / lo)
    f["z_lo"] = (spot - lo) / max(sd_dollars, 1e-9)
    f["dist_hi"] = math.log(spot / hi2)
    f["z_hi"] = (spot - hi2) / max(sd_dollars, 1e-9)
    p_sa = min(max(_ncdf(f["z_lo"]) - _ncdf(f["z_hi"]), 0.0), 1.0)
    f["p_sa"] = p_sa
    f["plogit"] = _logit(p_sa)
    f["inside"] = 1.0 if lo <= spot <= hi2 else 0.0
    f["abs_dist_edge"] = min(abs(f["dist_lo"]), abs(f["dist_hi"]))
    f["z_edge"] = f["abs_dist_edge"] / max(v[900] * math.sqrt(teff), 1e-9)
    for lag in (10, 30, 60, 300, 900, 1800, 3600):
        f[f"ret_{lag}"] = float(lb[-1] - lb[-1 - lag])
    for w in (900, 3600):
        f[f"from_hi_{w}"] = float(lb[-1] - lb[-w:].max())
        f[f"from_lo_{w}"] = float(lb[-1] - lb[-w:].min())
    f["volratio_60_900"] = v[60] / v[900]
    f["volratio_300_3600"] = v[300] / v[3600]
    f["prior_hr_vol_pct"] = v[3600] * math.sqrt(3600) * 100
    dt = datetime.fromtimestamp(t, timezone.utc)
    f["hour"], f["dow"] = float(dt.hour), float(dt.weekday())
    mid = (yes_bid + yes_ask) / 2
    f["mid"], f["klogit"], f["spread"] = mid, _logit(mid), yes_ask - yes_bid
    f["spot"] = spot
    return f


def evaluate(p_yes: float, f: dict, ask: float) -> dict:
    """Decide one band at one minute: inside the band, gap of at least MIN_GAP, YES only."""
    g = gap(p_yes, ask)
    inside = f["inside"] == 1.0
    return {"gap": g, "inside": inside, "buy": inside and g >= MIN_GAP}


def quote_ok(yb: float | None, ya: float | None) -> bool:
    return yb is not None and ya is not None and yb > 0 and ya <= 1 and ya > yb and ya - yb <= MAX_SPREAD + 1e-9


class BandMLPaper(RangePaper):
    def __init__(self, out, key_id=None, key_path=None, push_logs: bool = False, contracts: int = BOT_CONTRACTS,
                 model: DirectModel | None = None):
        super().__init__(out, key_id, key_path, push_logs, contracts)
        self.band_model = model or DirectModel.load(MODEL_FILE)

    def sec_history(self, t: int) -> np.ndarray | None:
        """BRTI for seconds t-3601 .. t-1, gaps carried forward; None when more than 2% of it is missing or the
        start is unknown."""
        out = np.empty(HISTORY)
        last = None
        for u in range(t - 1, t - 1 - 30, -1):          # the second before the check may still be in flight
            if u in self.brti:
                last = u
                break
        if last is None:
            return None
        miss = 0
        prev = None
        for i, u in enumerate(range(t - HISTORY, t)):
            v = self.brti.get(u)
            if v is None:
                miss += 1
                v = prev
            if v is None:
                v = self.price_at(u)
            if v is None:
                return None
            out[i] = prev = v
        return out if miss <= 0.02 * HISTORY else None

    def check(self, t: int):
        close = (t // 3600 + 1) * 3600
        left = (close - t) // 60
        if left > MAX_LEFT or left < MIN_LEFT:
            return
        sec = self.sec_history(t)
        if sec is None:
            log.warning("%s: BRTI history incomplete", _iso(t))
            return
        price = float(sec[-1])
        anchor = self.anchor.setdefault(close, self.price_at(close - 1800) or price)
        ev = event_ticker(close)
        markets = self.k.get("/markets", event_ticker=ev, limit=1000).get("markets", [])
        markets = [m for m in markets if m.get("status") in ("active", "open")]
        if not markets:
            log.warning("no open markets for %s", ev)
            return
        for m in sorted(markets, key=lambda mm: abs(band_mid(mm) - anchor))[:N_BANDS]:
            tk = m["ticker"]
            if tk in self.held:
                continue
            lo, hi = _f(m.get("floor_strike")), _f(m.get("cap_strike"))
            yb, ya = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
            if m.get("strike_type") != "between" or lo is None or hi is None or not quote_ok(yb, ya):
                continue
            f = features(sec, t, close, lo, hi, yb, ya)
            p_yes = self.band_model.predict(f)
            e = evaluate(p_yes, f, ya)
            if e["gap"] >= MIN_GAP and not e["inside"]:
                self._append("skips.csv", {"time_utc": _iso(t), "ticker": tk, "mins_left": left, "ask": ya,
                                           "model": round(p_yes, 4), "gap_pts": round(100 * e["gap"], 2),
                                           "brti": round(price, 2), "floor": lo, "cap": hi,
                                           "reason": "BRTI outside the band"})
                continue
            if e["buy"]:
                self.enter(t, close, left, m, ya, price, p_yes, e["gap"])

    def enter(self, t, close, left, m, ask, price, p_yes=None, g=None):   # noqa: D102
        tk = m["ticker"]
        try:
            book = self.k.orderbook(tk)
        except Exception as e:
            log.warning("orderbook %s failed: %s", tk, e)
            book = {}
        other = book.get("no_bids") or []
        limit = limit_price(p_yes, book_asks(book)["yes"], MIN_GAP) if other else 0.0
        n, avg = fill(other, self.contracts, limit) if limit else (0, None)
        row = {"time_utc": _iso(t), "ticker": tk, "side": "YES", "mins_left": left,
               "floor": m.get("floor_strike"), "cap": m.get("cap_strike"), "brti": round(price, 2),
               "model": round(p_yes, 4), "gap_pts": round(100 * g, 2), "ask_snapshot": ask,
               "book_best": round(1 - other[0][0], 4) if other else "", "limit": limit,
               "filled": n, "avg_price": round(avg, 4) if avg else "", "book": json.dumps(other[:6])}
        self._append("signals.csv", row)
        if n == 0:
            log.info("%s YES %s: ask %.2f but no fill within the gap (limit %.2f)", _iso(t), tk, ask, limit)
            return
        fee = kalshi_fee(n, avg)
        self.held[tk] = {"time_utc": _iso(t), "ticker": tk, "side": "YES", "mins_left": left,
                         "contracts": n, "avg_price": round(avg, 4), "fee": fee,
                         "cost": round(n * avg + fee, 2), "model": round(p_yes, 4),
                         "gap_pts": round(100 * gap(p_yes, avg), 2), "close_ts": close}
        self._save_open()
        log.info("PAPER BUY %d YES %s @ %.4f (model %.3f, gap %.1f pts, %d min left)", n, tk, avg, p_yes,
                 100 * gap(p_yes, avg), left)


def from_env(out: str, push_logs: bool, contracts: int = BOT_CONTRACTS) -> BandMLPaper:
    return BandMLPaper(out, os.environ.get("KALSHI_KEY_ID"), os.environ.get("KALSHI_KEY_FILE"), push_logs, contracts)
