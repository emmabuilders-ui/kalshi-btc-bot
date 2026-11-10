"""Status of a model-gap paper bot against the fixed decision rule (docs/paper_test_decision_rule.md).

Usage: python research/ml_replay/paper_check.py /opt/kalshi-logs-model-gap/model-gap/trades.csv direct
       python research/ml_replay/paper_check.py /opt/kalshi-logs-corr-gap/corr-gap/trades.csv correction
Reads only the bot's own settled paper trades. It never edits anything.
"""
import math
import sys

import pandas as pd

CHECKS = {"direct": [300, 800], "correction": [300, 800]}
RETIRE_BELOW = 1.0      # cents after fee
GO_AT = 2.4             # cents after fee at 800 trades (t of 2)


def kendall_tau(x, y):
    try:
        from scipy.stats import kendalltau
        return float(kendalltau(x, y)[0])
    except Exception:
        return float("nan")


def main(path, bot):
    d = pd.read_csv(path)
    d = d[d["won"].astype(str).isin(["0", "1", "0.0", "1.0"])].copy()
    n = len(d)
    if not n:
        print("no settled trades yet")
        return
    won = d["won"].astype(float)
    px = d["avg_price"].astype(float)
    fee = d["fee"].astype(float) / d["contracts"].astype(float)
    pnl = won - px - fee
    pnl_slip = won - (px + 0.01) - fee
    se = pnl.std() / math.sqrt(n) if n > 1 else float("nan")
    edge = 100 * pnl.mean()
    print(f"{bot}: {n} settled trades")
    print(f"  win {100 * won.mean():.1f}% vs avg price {100 * px.mean():.1f}c, win minus price {100 * (won.mean() - px.mean()):+.1f} pts")
    print(f"  edge after fee {edge:+.2f}c (error {100 * se:.2f}c, t {pnl.mean() / se if se else float('nan'):.2f})")
    print(f"  edge if every fill were 1c worse {100 * pnl_slip.mean():+.2f}c")
    if "gap_pts" in d:
        print(f"  Kendall tau, model gap vs trade result: {kendall_tau(d['gap_pts'].astype(float), pnl):+.3f}")
    first, last = CHECKS[bot]
    if n < first:
        print(f"  next check at {first} trades ({first - n} to go): no decision yet")
    elif n < last:
        print(f"  check 1 ({first} trades) passed: " + ("RETIRE, edge under +1.0c" if edge < RETIRE_BELOW else "keep running") +
              f"; next check at {last} trades")
    else:
        if edge < RETIRE_BELOW:
            v = "RETIRE"
        elif edge >= GO_AT and pnl.mean() / se >= 2 and pnl_slip.mean() > 0:
            v = "eligible for Michael to consider small real size"
        else:
            v = "in between: run to 1,300 trades"
        print(f"  check 2 ({last} trades): {v}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] not in CHECKS:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
