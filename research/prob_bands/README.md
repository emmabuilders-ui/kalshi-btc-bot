# Probability bands indicator

Chart indicator (display only, no bot uses it): 50 / 80 / 95 % bands around the BTC price for the 15-minute and 1-hour Kalshi
windows, from (1) the range model and (2) Kalshi's own prices. Code: `kalshi_btc_bot/prob_bands.py`, wired into
`dashboard.py` (button "Show probability bands", card "Probability bands now").

- `calibrate.py`  : model bands (hourly + 15-min, every window Dec 2025-Oct 2026) and Kalshi hourly-ladder bands, hit rates by minutes left and period.
- `calibrate15.py`: Kalshi-implied 15-min bands (width fitted to the strike price) and model 15-min bands on Aug-Oct, using each contract's last trade price as the quote and Kalshi's own settlement value.
- Output summary used by the page: `kalshi_btc_bot/data/band_calibration.json` ({kind|source: {period: {minutes_left: [hit50, hit80, hit95, n]}}}).

Inputs: 1-s BRTI array (`research/hourly_ml/build_brti.py`), `kalshi-hourly-range/*.parquet`, `kalshi-daily/Kalshi_BTC15M_*.zip`.
