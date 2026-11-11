# ML models on the hourly Kalshi markets (2026-11-11)

Scripts behind the Band ML paper bot (`kalshi-bandml`). Data and results live in the project files
(strategy/hourly_ml/, data-download/kalshi-hourly-range/).

1. `build_brti.py` one 1-second BRTI array, 2025-11-30 to 2026-10-31.
2. `build_bands.py` minute snapshots for the hourly range bands (KXBTC, 14 nearest bands at T-30, 1-30 min left, two-sided
   quotes with spread <= 10c) and features (`feat.py`: settlement-aware probability with the 40-s shrink, distance to the
   band edges in volatility units, returns, volatility, distance from recent highs and lows, hour, weekday).
3. `dl_btcd.py`, `build_btcd.py` the same for the above/below markets (KXBTCD, 7 strikes nearest spot at T-30).
4. `models.py` Direct, Correction (adds Kalshi mid and spread) and Lasso. Mar-Aug monthly walk-forward, Sep-Oct frozen
   Dec-Aug fit. `score.py` Brier, AUC, trades at 3/5/10-point gaps.
5. `train_live.py` the shipped model: Correction fit on Dec-Aug without the 24-hour high/low inputs (the bot keeps 4 hours of BRTI).
   `anat.py` breaks the YES trades down by price, minute, band position and volatility.
6. `check_live_features.py` proves the bot code gives the same probabilities as the research code.

Shipped rule (YES only, 7-24 minutes left, BRTI inside the band, model - ask - fee >= 5 points, first qualifying minute per
band, 10 contracts), one trade per band, shipped model, entry at the candle ask:
Mar-Aug (walk-forward) 1,678 trades, 57.7% vs 52.2c, +3.86c, 6/6 months; Sep-Oct (frozen) 428 trades, 62.4% vs 54.9c, +5.85c;
Oct 26-31 41 trades, 65.9% vs 55.6c, +8.61c. The window and the inside-band filter were picked after seeing the data.
