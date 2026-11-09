# Leak-free replay and baselines, Kalshi KXBTC15M (handoff steps 1 to 4)

Run in order (outputs go to `work/`):

1. `build_brti.py`: one 1-second BRTI array, 2025-11-30 to 2026-10-25.
2. `build_snaps.py`: contract table plus minute snapshots (open + 1..14 min). YES ask = last yes-taker trade price,
   YES bid = last no-taker trade price, each as of the snapshot second; May to Jul from `quotes_5s`.
3. `audit.py`: results recomputed from the BRTI 60-second average vs Kalshi; missing strikes rebuilt from the
   BRTI 60-second average before open.
4. `replay.py`: keeps quotes under 20 s old, adds BRTI distance and trailing 15/60 min volatility, all at or before the snapshot second.
5. `baselines.py`: Kalshi mid as is; static Platt calibration per minute left (fit Dec-Aug); rolling 30-day Platt refit daily on
   contracts closed before that day; normal-CDF vol model with one scale fit on Dec-Aug.
6. `score.py`: Brier, AUC, calibration, one trade per contract at 5/7/10 pt edges after the 0.07*p*(1-p) fee, sample-size math.
7. `drift.py`: weekly Kalshi win rate minus price per 10c bucket with PELT change-point detection.

Held-out period: Sep 1 to Oct 25 2026. No machine-learning model is trained here.
