# CME BTC futures order flow vs Kalshi (2026-11-14)

Run in order; `work/` is a scratch folder (symlink). Data: /mnt/project-files/tradestation/parquet/ (flow), data-download/brti/ (settlement),
kalshi-daily ZIPs / quotes_5s (15-min quotes), kalshi-hourly-range/btcd_quotes_1min (hourly quotes).

1. `build_brti.py`, `build_snaps.py`, `audit.py`, `replay.py`: copied from research/ml_replay (15-min contract table + fresh minute snapshots).
2. `build_hourly.py`: KXBTCD snapshots, 1-30 min left, BRTI settlement recomputed.
3. `flow.py`: minute grid of up/down volume and trade counts (@BTC, @MBT, combined in BTC units), CME-closed mask, signals at snapshot times.
4. `flow_sec.py`: second-level combined flow for the last-minute test.
5. `analyze.py [15m|hourly]`: tau/AUC, residual logistic, Brier vs Kalshi with contract bootstrap, deciles, trade scan.
6. `absorbed.py`, `seconds_test.py`, `workbook.py`.

Result: fail on every pre-set pass-bar line; see /mnt/project-files/strategy/futures_flow/futures-flow-results.md.
