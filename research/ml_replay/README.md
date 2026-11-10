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

## Step 5: four models (no Kalshi data after the snapshot second is used anywhere)

8. `features.py`: BRTI returns over 10 s to 15 min, 1/5/15/60-min volatility, distance in volatility units, position vs the
   contract's high and low so far, volume, quote ages, hour and weekday.
9. `models.py`: gradient-boosted trees. Direct outcome model (BRTI features only) and correction model (plus Kalshi price).
   Walk-forward: monthly expanding refits for Feb to Aug, one frozen Dec-Aug fit for Sep-Oct, and a weekly-refit variant.
   Regime-switching gate: 27 cells (volatility tercile x minutes-left group x price group), picking the expert with the
   lowest Brier on prior out-of-fold rows. Rolling recalibration comes from step 3.
10. `score5.py`: Brier with day-block bootstrap CIs vs Kalshi, AUC, Brier by price and minutes left, trades at 5/7/10 points.

## Order-flow test

11. `build_flow.py`: per contract and snapshot, from the trade files: buyer-minus-seller volume share over 10/30/60/180 s and since
    open, volume and trade counts, VWAP vs mid, last-trade price change, large-trade (100+ contracts) imbalance. All at or before
    the snapshot second. May to Jul has no trade files, so those months are excluded.
12. `models_of.py`, `score_of.py`: the same walk-forward with and without order flow on identical rows (Feb-Apr and Aug
    out-of-fold, Sep-Oct holdout frozen and weekly refit).

## Recalibration every 15 minutes

13. `recal15.py`: for each Sep-Oct contract, refits a logistic calibration (logit p, logit p x minutes left, minutes left) on
    contracts that closed at or before it opened, over trailing 1, 7 and 30 days. Applied to Kalshi mid and to the weekly-refit
    correction and direct models. `score_recal15.py` scores it.

## Step 4: automatic feature discovery

14. `autofeat.py`: 111 generated features at each snapshot: BRTI log returns over 13 lags (1 s to 1 h), volatility, mean absolute
    move and trend efficiency over 9 windows, distance from rolling highs/lows and range, strike position in range, distance to
    strike in units of each volatility, return z-scores, volatility ratios, drift-projected distance, Kalshi mid changes over
    1/2/3/5 minutes, previous contract's move, hour and weekday.
15. `select4.py`: trains on Dec-Jun, ranks features by permutation importance on Jul-Aug, picks feature counts on Jul-Aug.
16. `wf4.py <model>`: refits the chosen candidates on Dec-Aug, predicts Sep-Oct frozen and with weekly expanding refits.
17. `score4.py`: Brier vs Kalshi, cents and months better, trades, and every cent x minute-left grid.
BTC spot is not used: spot files cover only Aug to Oct, so Dec-Aug training would lack it.

## Kalshi vs all models (compare_all.py, monthly.py, weakest.py)
- compare_all.py: Sep–Oct, 19 models vs Kalshi price by time-left x 10c band (Brier, calibration miss).
- monthly.py: same by month Mar–Oct (Mar–Aug only out-of-fold models; static/vol flagged in-sample).
- weakest.py: month x time x price cells sorted by Kalshi % wrong (actual win % − Kalshi avg price), z on contracts; plus pooled Mar–Oct with months-same-sign.

## Pivot workbook (pivot_data.py, pivot_build.py)
- Long table month x minute-left x 10c band x model; Excel pivot tables built through LibreOffice UNO (soffice listening on port 2002).

## Correction ML paper bot (2026-11-10)
`corr_live.py` refits Correction ML without the two quote-age inputs (the live bot reads a fresh order book, not trade
prints): monthly out-of-fold Mar-Aug, frozen Dec-Aug fit for Sep-Oct. `export_corr.py` writes the frozen trees to
`kalshi_btc_bot/data/model_gap_correction.json` (pure-Python predictions equal scikit-learn's exactly).
Rule (`CorrectionGapPaper`): NO 70-99c, model - ask - fee >= 3 pts, 6-14 min left, first look per contract.
| Period | Trades | Win % vs avg price | After fee |
|---|---|---|---|
| Mar-Aug | 2,830 | 81.0 vs 78.8 | +1.10c (4/6 months, t 1.5) |
| Sep-Oct | 685 | 82.6 vs 77.2 | +4.26c (t 3.0) |
Replaying the bot's decision code (which stops at the first look on either side) gives 678 Sep-Oct trades, +4.33c.

`backfill.py` refits every other model walk-forward through Mar-Aug (static, vol, weekly ML, 15-min recal,
order flow where taker side exists, auto features with feature ranking redone on data before March).
