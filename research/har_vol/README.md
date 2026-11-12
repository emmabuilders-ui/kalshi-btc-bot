# HAR volatility forecast test (2026-11-12)

Data: BRTI 1-second prints Nov 30 2025 - Oct 25 2026 -> 1-minute bars. Realized variance (RV) = sum of squared 1-minute log returns.
Scripts: repo research/har_vol/ (vol1.py, har1.py, har_h.py, stage2_hourly.py, stage2_15m.py). Tables: har_vol_forecast_test.xlsx.

## Stage 1: does HAR forecast BTC volatility better than our inputs? YES
QLIKE (lower is better), frozen Dec-Aug fit, Sep-Oct out of sample:

| forecaster | next 15 min | next 60 min | tau (15 min) |
|---|---|---|---|
| HAR3 (15m, 1h, 24h) | .392 | .310 | .553 |
| HAR4 (+4h) | .391 | .308 | .555 |
| ATR14 x scale | .402 | .353 | .574 |
| EWMA 0.97 | .425 | .397 | .554 |
| s100 (live input) | .477 | .425 | .525 |
| last-window RV | .668 | .438 | .511 |

HAR beats s100 in 11 of 11 months (frozen) and 10 of 10 out-of-sample months (expanding monthly refit), both horizons. Pass bar met.
By horizon (Sep-Oct QLIKE): HAR's lead is large at 30-60 min, shrinks to about equal with ATR14 at 2-10 min, and at 1 min HAR is the worst (lognormal correction over-inflates: bias 2.6x). ATR14 has the best rank correlation (tau) at every horizon 1-30 min.

## Stage 2: does the better forecast price Kalshi contracts better? NO
Each input -> sigma for the minutes left (minus 40 s), standardized-move tables rebuilt on Dec-Aug, Brier vs Kalshi mid.

Hourly bands, 1-30 min left, spread <= 5c (n 1.24M / 418k band-minutes):
| model | Dec-Aug | Sep-Oct | months < Kalshi |
|---|---|---|---|
| Kalshi mid | .06050 | .05643 | - |
| ATR14 x scale | .06032 | .05647 | 8/11 |
| EWMA 0.97 | .06038 | .05648 | 7/11 |
| s100 (live) | .06046 | .05657 | 5/11 |
| HAR3 frozen | .06046 | .05659 | 7/11 |
| Oracle (actual future RV, not tradable) | .05744 | .05408 | 11/11 |

15-min contracts, 1-14 min left, fresh quotes (n 266k / 69k):
| model | Dec-Aug | Sep-Oct |
|---|---|---|
| Kalshi mid | .15186 | .15534 |
| HAR3 frozen | .15198 | .15762 |
| EWMA 0.97 | .15202 | .15774 |
| ATR14 x scale | .15179 | .15788 |
| s100 (live) | .15223 | .15800 |
| Oracle | .14679 | .15027 |

HAR is a wash against s100 on Brier (differences at the 4th decimal), and all inputs are at or behind Kalshi in Sep-Oct. The oracle shows a real ceiling (about 0.002-0.005 Brier below Kalshi): knowing the next-15-minute variance would beat Kalshi every month. HAR captures almost none of that gap because band/strike outcomes at 1-14 min depend on the near-term move, where HAR has no advantage.

Trades (first qualifying minute per contract, fee included), hourly YES 90-99c model >= ask+2:
- s100: Dec-Aug 1,115 tr, 95.1% won at 92.4c, +2.1c; Sep-Oct 279 tr, 95.0% at 92.5c, +2.0c
- HAR3: Dec-Aug 919 tr, 94.2% at 91.9c, +1.8c; Sep-Oct 208 tr, 95.2% at 92.0c, +2.7c (fewer trades, same edge)
- 75-89c +2: s100 +3.1 / +1.5c; HAR3 +3.8 / -0.0c. 15-min: no cell positive in both periods for any input beyond what s100 already shows.

Conclusion: HAR is the best volatility forecaster we have tested (use it if a forecast of the next 15-60 min RV is needed), but a better RV forecast does not improve contract pricing against Kalshi. The ranking at 1-5 min (ATR14, tau .33-.51) matters more for Brier than the level accuracy HAR provides.
