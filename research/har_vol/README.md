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

## ETH and XRP, stage 1 (added 2026-11-12, Michael: "run this on the xrp and eth data")
Prices: Binance ETHUSDT / XRPUSDT 1-second klines (data.binance.vision), Nov 30 2025 - Oct 25 2026, no missing seconds. Kalshi's CF index passthrough was rate-limited to ~5 calls/min (key shared with the paper bots), so the Binance series stands in; it is a volatility proxy only, so no Brier-vs-Kalshi stage for these coins yet.
QLIKE Sep-Oct (frozen Dec-Aug fit), next 15 / next 60 min:
- ETH: HAR3 .417/.343, EWMA .465/.455, ATR14 .485/.483, s100 .520/.484. HAR beats s100 10/10 expanding months and beats ATR14 10/10.
- XRP: HAR3 .332/.265, ATR14 .336/.301, EWMA .358/.327, s100 .401/.351. HAR beats s100 10/10; beats ATR14 7/10 (15 min) and 9/10 (60 min).
Tau (15 min, Sep-Oct): ETH ATR14 .545 > HAR3 .527 > EWMA .524 > s100 .492; XRP ATR14 .574 > HAR3 .555 > EWMA .554 > s100 .526. Same pattern as BTC: HAR best on level (QLIKE), ATR14 best on ranking (tau).
Scripts: bn/dl.sh, bn/build.py, run_coin.sh in research/har_vol/.

## ETH and XRP, stage 2 (added 2026-11-14, Michael: "do it")
Kalshi hourly range bands (KXETH 1.46M band markets, KXXRP 593k), nearest 6-14 bands per hour (within 2.5% of price), 1-min candles for the last 30 min, spread <= 5c.
Price at each minute = Binance 1-s close + basis, basis = Kalshi settlement value minus Binance 60-s average at the PREVIOUS hour's close (ETH median -$1.04, XRP -$0.0005, about -0.04% both). Tables rebuilt on Dec-Aug per coin. XRP quotes are thin before Jul 2026 (2-8k band-minutes a month, none in Dec), so its Dec-Aug column is mostly Jul-Aug.

Brier Sep-Oct (lower better) / AUC:
- ETH (452k / 199k band-minutes): ATR14 .07077 (.902), Kalshi .07080 (.901), EWMA .07084 (.902), s100 .07096 (.901), HAR3 .07103 (.901), oracle .06810 (.912). ATR14 beats Kalshi 6/11 months, HAR 5/11.
- XRP (146k / 98k): ATR14 .03353 (.989), EWMA .03365, HAR3 .03366, s100 .03373, Kalshi .03394 (.989), oracle .03193 (.991). All inputs beat Kalshi on XRP (thin market, as found before); ATR14 best 10/10 months vs Kalshi; HAR beats s100 only 4/10 months.
Trades YES 90-99c model >= ask+2c, first minute per band, Sep-Oct: ETH ATR14 344 tr 96.5% won at 92.5c +3.5c vs s100 281 tr 95.4% at 92.6c +2.3c vs HAR3 211 tr 94.8% at 92.2c +2.1c; XRP ATR14 941 tr 96.5% at 92.8c +3.2c, s100 996 tr 96.5% at 92.8c +3.2c, HAR3 1,015 tr 96.2% at 92.8c +2.9c.
Same conclusion as BTC: HAR = s100 on Brier (slightly worse), no pricing gain; ATR14 is the best tradable input on all three coins; the oracle gap (0.002-0.003) is the ceiling.
Scripts: kx/markets.py, kx/candles.py, stage2_coin.py.

## Every minute x 10c price x YES/NO scan (added 2026-11-14 evening, Michael: "check every price/time/yes or no and compare")
File har_vol_price_time_scan.xlsx: one flat sheet per data set (BTC hourly, BTC 15-min, ETH hourly, XRP hourly) with n, avg price, win %, win-price, edge after fee, model gap avg/median, months positive, for Kalshi-as-is and each model (buy when model >= price + 2c). "Pass both periods" sheet = n>=100 each period, edge>0 both, above no-model both.
Head-to-head on the 355 cells with n>=100 for every model in both periods: best Sep-Oct model ATR14 in 118 cells, s100 75, HAR 72, EWMA 52, Kalshi-as-is 38. Mean Sep-Oct edge across cells: HAR -0.03c, EWMA -0.12c, ATR -0.13c, s100 -0.17c, no model -1.30c (Dec-Aug: ATR +1.46, EWMA +1.03, s100 +0.76, HAR +0.74, no model -1.13). Pass cells (of 635): ATR14 178, EWMA 163, s100 148, HAR 146. All four models add the same ~1c over Kalshi's raw price; no model separates from the others at any price/time/side, and HAR is never the standout.
