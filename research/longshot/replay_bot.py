"""Replay the shipped bot's own decision code (kalshi_btc_bot.longshot_paper) over Dec 2025 - Oct 2026.

It uses the bot's `inputs`, `evaluate` and `limit_price` on the leak-free snapshots from research/ml_replay
(work/replay.parquet, built by build_brti.py, build_snaps.py, audit.py, replay.py) with a one-level book at the
snapshot's ask, and takes the first qualifying minute per contract, exactly as the bot does. The numbers it prints
should match the backtest in research/longshot/README.md; this is the check that the bot codes the tested rule.

    python replay_bot.py [path/to/ml_replay/work]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_btc_bot.longshot_paper import MIN_EDGE, PX_MAX, edge, evaluate, inputs, limit_price

T0 = 1764460800          # 2025-11-30 00:00 UTC, the first second of brti.npy
VOL_WINDOW = 900
CONTRACTS = 20


def run(work: Path) -> pd.DataFrame:
    brti = pd.Series(np.load(work / "brti.npy")).ffill().to_numpy()
    S = pd.read_parquet(work / "replay.parquet").sort_values(["ticker", "min_left"], ascending=[True, False])
    rows = []
    looked: set[str] = set()
    for r in S.itertuples():
        if r.ticker in looked:
            continue
        e = int(r.snap_ts) - T0
        sec = brti[e - VOL_WINDOW:e + 1]
        if len(sec) < VOL_WINDOW + 1 or np.isnan(sec).any():
            continue
        f = inputs(sec, int(r.snap_ts), int(r.close_ts), float(r.strike))
        if f["sig15"] <= 0:
            continue
        asks = {"yes": [[round(float(r.ask), 4), CONTRACTS]], "no": [[round(1 - float(r.bid), 4), CONTRACTS]]}
        ev = evaluate(f, asks, MIN_EDGE)
        side = ev["look"]
        if side is None:
            continue
        looked.add(r.ticker)
        price = limit_price(ev["fair"], asks[side], MIN_EDGE)
        if not price:
            continue
        won = int(r.res) if side == "yes" else 1 - int(r.res)
        rows.append({"month": str(r.month), "period": r.period, "ticker": r.ticker, "side": side,
                     "min_left": int(r.min_left), "price": price, "fair": ev["fair"], "zgap": ev["zgap"],
                     "distance_usd": abs(f["brti"] - f["strike"]), "one_sd_usd": f["move_1sd"], "won": won,
                     "pnl": won - price - 0.07 * price * (1 - price)})
    return pd.DataFrame(rows)


def report(t: pd.DataFrame) -> pd.DataFrame:
    def one(g):
        n = len(g)
        return pd.Series({"trades": n, "avg_price_c": round(100 * g.price.mean(), 2),
                          "win_pct": round(100 * g.won.mean(), 2),
                          "breakeven_pct": round(100 * (g.price + 0.07 * g.price * (1 - g.price)).mean(), 2),
                          "win_minus_price": round(100 * (g.won.mean() - g.price.mean()), 2),
                          "ev_c": round(100 * g.pnl.mean(), 2),
                          "se_c": round(100 * g.pnl.std() / np.sqrt(max(n, 1)), 2)})
    return t.groupby("period").apply(one)


if __name__ == "__main__":
    w = Path(sys.argv[1] if len(sys.argv) > 1 else "../ml_replay/work")
    t = run(w)
    pd.set_option("display.width", 200)
    print(f"rule: side at {100 * PX_MAX:g}c or less, fair - ask - fee >= {100 * MIN_EDGE:g} points, "
          f"first qualifying minute per contract\n")
    print(report(t).to_string())
    print("\nby month")
    print(t.groupby("month").apply(lambda g: pd.Series(
        {"trades": len(g), "avg_price_c": round(100 * g.price.mean(), 2), "win_pct": round(100 * g.won.mean(), 2),
         "ev_c": round(100 * g.pnl.mean(), 2)})).to_string())
    print("\nby minutes left")
    print(t.groupby("min_left").apply(lambda g: pd.Series(
        {"trades": len(g), "avg_price_c": round(100 * g.price.mean(), 2), "win_pct": round(100 * g.won.mean(), 2),
         "ev_c": round(100 * g.pnl.mean(), 2)})).to_string())
    print(f"\nmedian distance to the strike ${t.distance_usd.median():.0f} against a typical move of "
          f"${t.one_sd_usd.median():.0f}; {len(t) / t.ticker.nunique():.2f} trades per contract traded")
    print(f"at 20 contracts: {100 * 20 * t.pnl.sum() / 100:.0f} dollars total, "
          f"{t.groupby(t.month).pnl.mean().gt(0).sum()} of {t.month.nunique()} months positive")
