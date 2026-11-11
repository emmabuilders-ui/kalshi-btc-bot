# RVI indicators and DVS (default: 15-minute BRTI bars)

Code: `kalshi_btc_bot/ta_indicators.py`. Bars come from the 1-s BRTI series via `make_bars(price, "15m")`
(labelled by bar end, so no look-ahead). Not a trading signal: no Kalshi edge test has been run.

| Indicator | Formula | Defaults |
|---|---|---|
| Relative Vigor Index (Ehlers) | num=SWMA(close-open), den=SWMA(high-low); RVI=SMA(num,N)/SMA(den,N); signal=SWMA(RVI); SWMA weights 1-2-2-1 | N=10 |
| Relative Volatility Index (Dorsey) | std=10-bar stdev of close; up/down bucket by close change; Wilder-smooth both; 100*up/(up+down) | std 10, smooth 14 |
| DVS (Directional Volatility Strength) | ((Dorsey-50)/50 + Ehlers RVI)/2, range -1..+1 | as above |

Positive DVS: volatility and bar conviction both lean up. Negative: both lean down.
