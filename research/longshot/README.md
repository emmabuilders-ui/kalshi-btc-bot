# Longshot scan (contracts at 15c or less), 2026-11-10

Question: is there any condition under which a Kalshi BTC contract bought at 15c or lower (YES or NO) wins more often than
its price plus the taker fee (0.07 x price x (1-price))?

Data: 15-min KXBTC15M snapshots from the leak-free replay (`research/ml_replay/`: `build_brti.py`, `build_snaps.py`,
`audit.py`, `replay.py` -> `work/replay.parquet`), hourly bands from
`data-download/kalshi-hourly-range/` (1-min candles, deduplicated), BRTI 1-second index. Dec 2025 - Oct 25 2026.
Rules chosen on Dec-Aug; Sep-Oct looked at once afterwards.

Scripts (run from a scratch folder; outputs go to the working directory):
- `base.py`: every 1c level x minute left, no conditions (from `strategy/inefficiency_map_rows.parquet`).
- `hb_build.py`, `hb_scan.py`: hourly-band longshot rows with distance to band, moves toward band, volatility state, spread,
  ET hour, weekday; 1,119 cells scanned. Nothing positive on Dec-Aug.
- `m15_build.py`, `m15_scan.py`: 15-min longshot rows with the same kind of conditions; 1,241 cells scanned.
  One cell passed on Dec-Aug (1-5c, 1-3 min left, BTC close to the strike in volatility units) and held on Sep-Oct.
- `fair.py`: the clean version of that lead. fair = normal CDF of -(distance to strike)/(15-min vol x sqrt(seconds left - 40));
  buy the <=15c side when fair - price - fee >= margin, one trade per contract.

Result (15-min, margin 8c, chosen on Dec-Aug as the first margin with t > 2.5):

| Period | Trades | Avg price | Win % | Break-even | Win - price | After fee |
|---|---|---|---|---|---|---|
| Dec-Aug | 1,454 | 9.55c | 12.38% | 10.15% | +2.83 | +2.23c (t 2.6, 8/9 months) |
| Sep-Oct | 174 | 8.97c | 11.49% | 9.53% | +2.52 | +1.96c (t 0.8, 2/2 months) |

About 5-6 trades/day; $648 + $68 at 20 contracts; max drawdown -$211. Hourly bands: same rule is negative at every margin
with real counts. Outputs: `strategy/longshot_scan_15c.xlsx`, `strategy/longshot_rule_15min_pnl.png`.
Not built as a bot. Needs forward confirmation (about 300 trades) before anything else.

## Empirical table version (`emp.py`, 2026-11-10, Michael asked)
Replace the normal curve with the actual Dec-Aug win rate per (distance/SD bin x minutes left) cell, cross-fitted by month
for Dec-Aug and frozen for Sep-Oct. The table is well calibrated out of sample (normal curve says 17% where reality is
12.5%; the table says 12%). It values the normal rule's trades at 12.7% vs 9.55c paid, matching the 12.4% actual win rate.
Trading straight off the table (table - price - fee >= 2c) is weaker: Dec-Aug 3,797 trades 6.2c paid, 6.8% won, +0.16c;
Sep-Oct 612 trades +2.15c. The extra trades it adds (cheap 1-6c contracts, 6-10 min left) have no edge on Dec-Aug (-0.6c).
