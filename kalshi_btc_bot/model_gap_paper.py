"""Model-gap PAPER trader for Kalshi KXBTC15M (read-only, no orders are sent).

Michael asked for it on 2026-11-10 ("make that a test for the webpage. but i am going to add more.") after the
leak-free 15-minute ML replay (research/ml_replay/). The rule, exactly as tested there:

* Model: "Direct ML", a gradient-boosted classifier that sees only BRTI history (distance to strike, volatility over
  1/5/15/60 minutes, returns over 10 s to 15 min, high/low since the market opened, time left, UTC hour, weekday).
  It does NOT see Kalshi's price. Fit once on Dec 10 2025 - Aug 31 2026 and frozen (data/model_gap_direct.json);
  that frozen fit is the one scored on Sep-Oct.
* Checks: at each minute mark 1-9 minutes after the open (14 down to 6 minutes left), with BRTI up to that second.
* Gap: the model's probability for a side minus (that side's ask + taker fee 0.07 x p x (1-p)). A side qualifies when
  its ask is 70-99c and the gap is at least 3 points.
* First qualifying minute only: the first minute a side qualifies is the contract's one look (later minutes are not
  checked again), as in the backtest's one trade per contract.
* Trade only the NO side, and only if either
    - cushion: (log(BRTI/strike) / (15-min 1-s volatility x sqrt(seconds left - 40))) on the NO side is above 1.043
      (the top third of Mar-Aug trades), or
    - breakout: BRTI is below the low of the 60 minutes before the contract opened.
  A YES-side or no-condition look is logged as a skip, so other conditions can be scored later from signals.csv.
* 20 contracts at the ask (walking the book only while the gap stays at least 3 points), held to settlement.

Backtest (one trade per contract, first qualifying minute; win % vs average price, after fee):
  Mar-Aug (model out-of-fold): 1,204 trades, 86.8% vs 83.5c, +2.37c per contract, 5 of 6 months positive
  Sep-Oct (frozen model):        487 trades, 86.7% vs 81.0c, +4.56c per contract, 2 of 2 months positive
The rule was chosen after scanning many cells on the same data, so this paper test is the real check.
Backtest prices came from Kalshi trade prints; the paper bot uses the live order book.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from importlib import resources

import numpy as np

from .late_breakdown import kalshi_fee, log
from .range_paper import CONTRACTS, _iso, fill
from .squeeze_paper import SqueezePaper, _read_json

CHECK_KS = range(1, 10)          # minute marks after the open: 14 .. 6 minutes left
PX_MIN, PX_MAX = 0.70, 0.99
MIN_GAP = 0.03
CUSHION_MIN = 1.0427010403045358
BREAK_WINDOW = 3600              # seconds before the open that set the breakout levels
TRADE_SIDES = ("no",)
MAX_LAG = 20                     # seconds between the check second and acting on it


# ----------------------------------------------------------------------------- the model
class DirectModel:
    """Gradient-boosted trees exported from scikit-learn's HistGradientBoostingClassifier (pure Python predict)."""

    def __init__(self, spec: dict):
        self.features = spec["features"]
        self.base = spec["baseline"]
        self.trees = spec["trees"]
        self.name = spec.get("name", "")

    @classmethod
    def load(cls) -> "DirectModel":
        txt = resources.files("kalshi_btc_bot").joinpath("data/model_gap_direct.json").read_text()
        return cls(json.loads(txt))

    def predict(self, feats: dict) -> float:
        x = [feats[f] for f in self.features]
        raw = self.base
        for t in self.trees:
            i = 0
            while not t["leaf"][i]:
                v = x[t["f"][i]]
                left = bool(t["ml"][i]) if v is None or v != v else v <= t["th"][i]
                i = t["l"][i] if left else t["r"][i]
            raw += t["v"][i]
        return 1.0 / (1.0 + math.exp(-raw))


def features(sec: np.ndarray, t: int, open_ts: int, close_ts: int, strike: float) -> dict:
    """Model inputs at second t. `sec` holds 1-second BRTI for [t-3600, t] inclusive (3601 values, gaps filled)."""
    sec = np.asarray(sec, float)
    b = float(sec[-1])
    lr = np.diff(np.log(sec))                      # lr[-1] is the return into second t

    def sd(w):
        r = lr[-w:]
        m = r.mean()
        return float(math.sqrt(max((r * r).mean() - m * m, 0.0)))

    s1, s5, s15, s60 = sd(60), sd(300), sd(900), sd(3600)
    secs_left = close_ts - t
    teff = max(secs_left - 40, 1)
    dist = math.log(b / strike)
    since_open = sec[len(sec) - 1 - (t - open_ts):]
    dt = datetime.fromtimestamp(t, timezone.utc)
    f = {"dist": dist, "z": dist / (max(s15, 1e-7) * math.sqrt(teff)), "z60": dist / (max(s60, 1e-7) * math.sqrt(teff)),
         "secs_left": float(secs_left), "sig1": s1, "sig5": s5, "sig15": s15, "sig60": s60,
         "vol_ratio": s15 / max(s60, 1e-9), "vol_ratio_short": s1 / max(s15, 1e-9),
         "dist_hi": math.log(b / since_open.max()), "dist_lo": math.log(b / since_open.min()),
         "hour_utc": float(dt.hour), "dow": float(dt.weekday()), "brti": b}
    for w in (10, 30, 60, 300, 900):
        f[f"ret_{w}"] = math.log(b / sec[-1 - w])
    return f


def gap(prob_side: float, price: float) -> float:
    return prob_side - price - 0.07 * price * (1 - price)


def book_asks(book: dict) -> dict:
    """Ask ladders for YES and NO from the bid ladders (YES ask = 1 - NO bid)."""
    out = {}
    for side, other in (("yes", "no_bids"), ("no", "yes_bids")):
        lv = book.get(other) or []
        out[side] = [[round(1 - p, 4), q] for p, q in lv]
    return out


def evaluate(f: dict, p_yes: float, asks: dict, pre_hi: float, pre_lo: float) -> dict:
    """Decide one minute check. Returns the look (or None) and every number behind it."""
    out = {"p_yes": p_yes, "look": None}
    for side in ("yes", "no"):
        lv = asks.get(side) or []
        ask = lv[0][0] if lv else None
        prob = p_yes if side == "yes" else 1 - p_yes
        g = gap(prob, ask) if ask is not None else None
        out[f"{side}_ask"], out[f"{side}_gap"] = ask, g
        if ask is not None and PX_MIN <= ask <= PX_MAX and g >= MIN_GAP:
            out["look"] = side
    side = out["look"]
    sgn = 1 if side == "yes" else -1
    out["cushion"] = sgn * f["z"] if side else None
    out["breakout"] = (f["brti"] > pre_hi if side == "yes" else f["brti"] < pre_lo) if side else None
    out["condition"] = bool(side) and (out["cushion"] > CUSHION_MIN or out["breakout"])
    return out


def limit_price(prob_side: float, levels: list[list[float]]) -> float:
    """Highest book price that still leaves a gap of at least MIN_GAP (capped at PX_MAX)."""
    lim = 0.0
    for p, _ in levels:
        if p <= PX_MAX and gap(prob_side, p) >= MIN_GAP:
            lim = max(lim, p)
    return lim


# ----------------------------------------------------------------------------- the bot
class ModelGapPaper(SqueezePaper):
    BOT = "Model gap"

    def __init__(self, out, key_id=None, key_path=None, push_logs: bool = False, contracts: int = CONTRACTS,
                 model: DirectModel | None = None):
        self.model = model or DirectModel.load()
        self.looked: set[str] = set()          # tickers whose one look has been used
        self._pre: dict[str, tuple[float, float]] = {}
        super().__init__(out, key_id=key_id, key_path=key_path, push_logs=push_logs, contracts=contracts, max_lag=MAX_LAG)

    def _load_state(self):
        j = _read_json(self.out / "state.json", {})
        self.looked = set(j.get("looked", []))
        self.traded = set(j.get("traded", []))
        self.unsettled = _read_json(self.out / "open_trades.json", [])

    def _save_state(self):
        (self.out / "state.json").write_text(json.dumps(
            {"looked": sorted(self.looked)[-200:], "traded": sorted(self.traded)[-50:]}))

    def on_second(self, t: int, price: float):
        m = self._market(self.wall())
        if not m or not m["open_ts"] <= t < m["close_ts"] or m["ticker"] in self.looked:
            return
        k, r = divmod(t - m["open_ts"], 60)
        if r or k not in CHECK_KS:
            return
        if self.wall() - t > MAX_LAG:
            self._row(t, m, None, None, None, f"skip: check is {self.wall() - t}s late")
            return
        self.check(t, m)

    def _pre_levels(self, m: dict) -> tuple[float, float]:
        if m["ticker"] not in self._pre:
            a = self.series.array(m["open_ts"] - BREAK_WINDOW, m["open_ts"])
            self._pre = {m["ticker"]: (float(np.nanmax(a)), float(np.nanmin(a)))}
        return self._pre[m["ticker"]]

    def check(self, t: int, m: dict):
        sec = self.series.array(t - 3600, t + 1)
        if np.isnan(sec).any():
            self._row(t, m, None, None, None, "skip: BRTI history missing")
            return
        f = features(sec, t, m["open_ts"], m["close_ts"], m["strike"])
        p_yes = self.model.predict(f)
        try:
            book = self.k.orderbook(m["ticker"])
        except Exception as e:
            log.warning("orderbook failed: %s", e)
            self._row(t, m, f, p_yes, None, "skip: order book unavailable")
            return
        asks = book_asks(book)
        hi, lo = self._pre_levels(m)
        ev = evaluate(f, p_yes, asks, hi, lo)
        ev["pre_hi"], ev["pre_lo"] = hi, lo
        side = ev["look"]
        if side is None:
            self._row(t, m, f, p_yes, ev, "no look (no side 70-99c with a 3-point gap)")
            return
        self.looked.add(m["ticker"])
        self._save_state()
        if side not in TRADE_SIDES:
            self._row(t, m, f, p_yes, ev, f"skip: {side.upper()} side is not in the rule")
            return
        if not ev["condition"]:
            self._row(t, m, f, p_yes, ev, "skip: no cushion and no breakout")
            return
        prob = p_yes if side == "yes" else 1 - p_yes
        other = book.get("no_bids" if side == "yes" else "yes_bids") or []
        n, avg = fill(other, self.contracts, limit_price(prob, asks[side]))
        if not n:
            self._row(t, m, f, p_yes, ev, "skip: nothing to buy with a 3-point gap")
            return
        fee = kalshi_fee(n, avg)
        self.traded.add(m["close_ts"])
        self._save_state()
        self._row(t, m, f, p_yes, ev, f"BUY {side.upper()}", contracts=n, avg_price=round(avg, 4))
        self.unsettled.append({"time_utc": _iso(t), "ticker": m["ticker"], "side": side.upper(),
                               "level": "cushion" if ev["cushion"] > CUSHION_MIN else "breakout",
                               "secs_left": m["close_ts"] - t, "model_prob": round(prob, 4),
                               "gap_pts": round(100 * gap(prob, avg), 2), "contracts": n, "avg_price": round(avg, 4),
                               "fee": fee, "cost": round(n * avg + fee, 2), "close_ts": m["close_ts"],
                               "result": "", "won": "", "pnl": ""})
        self._save_open()
        log.info("PAPER BUY %d %s %s @ %.2f (model %.3f, fee %.2f)", n, side.upper(), m["ticker"], avg, prob, fee)

    def _row(self, t, m, f, p_yes, ev, decision, **extra):
        ev = ev or {}
        r = lambda v, d=4: "" if v is None else round(float(v), d)
        row = {"time_utc": _iso(t), "ticker": m["ticker"], "min_left": (m["close_ts"] - t) // 60,
               "secs_left": m["close_ts"] - t, "brti": r(f["brti"] if f else None, 2), "strike": round(m["strike"], 2),
               "model_p_yes": r(p_yes), "yes_ask": r(ev.get("yes_ask")), "no_ask": r(ev.get("no_ask")),
               "yes_gap_pts": "" if ev.get("yes_gap") is None else round(100 * ev["yes_gap"], 2),
               "no_gap_pts": "" if ev.get("no_gap") is None else round(100 * ev["no_gap"], 2),
               "side": (ev.get("look") or "").upper(), "cushion_z": r(ev.get("cushion"), 3),
               "breakout": "" if ev.get("breakout") is None else int(bool(ev["breakout"])),
               "pre_open_hi": r(ev.get("pre_hi"), 2), "pre_open_lo": r(ev.get("pre_lo"), 2),
               "z": r(f["z"] if f else None, 3), "sig15": r(f["sig15"] if f else None, 8),
               "ret_300": r(f["ret_300"] if f else None, 6),
               "decision": decision, "contracts": "", "avg_price": ""}
        row.update(extra)
        self._append("signals.csv", row)
        log.info("%s %s -> %s", m["ticker"], row["side"] or "-", decision)


def from_env(out: str, push_logs: bool, contracts: int = CONTRACTS) -> ModelGapPaper:
    return ModelGapPaper(out, os.environ.get("KALSHI_KEY_ID"), os.environ.get("KALSHI_KEY_FILE"), push_logs, contracts)
