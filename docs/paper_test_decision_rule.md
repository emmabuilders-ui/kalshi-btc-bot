# Paper-test decision rule: Direct and Correction model-gap bots

Fixed on 2026-11-10, before the first paper trade. Do not edit after seeing results.

Bots: `kalshi-model-gap` (Direct ML, cushion or breakout) and `kalshi-corr-gap` (Correction ML). Both are paper only.
Live trading stays locked off. Going live is Michael's decision alone.

## What is judged

- Only paper fills in each bot's `trades.csv`. Not the Mar-Oct backtests and not the Oct 26 - Nov 9 days.
- Edge = win % minus price paid, after Kalshi's fee, per contract. Always read win % next to the average price.
- One trade has a standard deviation of about 33c, so the error at n trades is about 33c / sqrt(n).

## Checkpoints

| Bot | Trades per day | Check | Outcome |
|---|---|---|---|
| Direct | about 9 | 300 trades (about 5 weeks, error about 1.9c) | edge under +1.0c: retire. +1.0c or more: keep running |
| Direct | about 9 | 800 trades (about 3 months, error about 1.2c) | +2.4c or more (t of 2): Michael may consider small real size. Under +1.0c: retire. In between: run to 1,300 trades |
| Correction | about 12 | 300 trades (about 3.5 weeks) | edge under +1.0c: retire. Otherwise same 800-trade check as Direct |

At every check the edge must also stay above zero if each fill were 1c worse than the logged ask.

## Guardrails

- No changes to rules, gap, price band, minutes or conditions during the test.
- A condition added later from `signals.csv` counts only on data after the day it was coded.
- No new scans of Mar-Oct to rescue a bot.
- Day one: compare paper fills to the logged asks.
- Fee confirmed on 2026-11-10 from Kalshi's fee schedule effective July 7, 2026: taker fee is 0.07 x p x (1-p) per contract, rounded up. KXBTC15M has no special multiplier.
- Report gap size against outcome with Kendall tau at each check.

Run `python research/ml_replay/paper_check.py <trades.csv> direct|correction` to print the current status.

## Longshot bot (`kalshi-longshot`), added 2026-11-10

Rule: a 15-minute side priced 15c or less, bought when the normal fair value (distance to the strike divided by the
15-minute volatility times the square root of the time left) beats the ask plus the taker fee by 8 points; first
qualifying minute per contract, 1-14 minutes left, 20 contracts, held to settlement. Paper only.

Backtest of the shipped code: Dec-Aug +1.94c per contract (1,497 trades), Sep-Oct +2.46c (184 trades). Both periods
were used to find the rule, so neither is a holdout.

Decision, fixed before the first paper trade and not to be changed afterwards. Judge on paper fills only, by win %
minus average price paid after the fee, with the win % always shown beside the price. Per-trade standard deviation is
about 30c at these prices, so the standard error is roughly 30c/sqrt(n).

- At 300 trades (about 7 weeks at 5-6 trades a day, standard error about 1.7c): edge under +1.0c retires the bot.
- At 800 trades (about 5 months, standard error about 1.1c): +2.0c or more with t at least 2 may be considered for
  small real size, which is Michael's call alone; under +1.0c retires it; in between runs on to 1,300 trades.
- Report the edge at the logged fills and again with each fill 1c worse, since a longshot ask moves easily.
- No changes to the margin, the 15c ceiling, the minutes checked or the side during the test. Any condition added
  from signals.csv counts only on data after the day it was coded.
- No new scans of Dec-Oct to rescue the bot. Live trading stays locked off.

## Band ML bot (`kalshi-bandml`), added 2026-11-11

Rule: hourly range bands (KXBTC, the 14 nearest BRTI 30 minutes before the close, "between" bands only). Buy YES at the
first minute with 24 down to 7 minutes left where BRTI is inside the band and the Correction model's chance minus the YES
ask minus the taker fee is at least 5 points; two-sided quote, spread at most 10c; one position per band; 10 contracts
(Michael's choice), held to settlement. The model is gradient-boosted trees fit once on Dec 2025 - Aug 2026 and frozen
(data/band_ml_correction.json). Paper only.

Backtest of the shipped model (research/hourly_ml/, one trade per band, candle ask, after fee): Mar-Aug (monthly walk-forward)
1,678 trades, 57.7% vs 52.2c, +3.9c, 6 of 6 months; Sep-Oct (frozen fit) 428 trades, 62.4% vs 54.9c, +5.9c, 2 of 2;
Oct 26-31 41 trades, 65.9% vs 55.6c, +8.6c. The 7-24 minute window and the inside-band condition were chosen after looking at
these same results, so the real edge is probably lower and only the paper test is clean.

Decision, fixed before the first paper trade; judge on paper fills only, win % minus average price after the fee, with the
win % always shown beside the price. Per-trade standard deviation is about 47c, so the standard error is roughly 47c/sqrt(n).

- At 300 trades (about 6 weeks at 7 a day, standard error about 2.7c): edge under +1.0c retires the bot.
- At 800 trades (standard error about 1.7c): +2.0c or more with t at least 2 may be considered for small real size, which
  is Michael's call alone; under +1.0c retires it; in between runs on to 1,300 trades.
- Report the edge at the logged fills and again with each fill 1c worse.
- No refit, no change to the model, the 5-point gap, the 7-24 minute window, the inside-band condition or the side during
  the test. A condition added later from signals.csv or skips.csv counts only on data after the day it was coded.
- skips.csv logs looks that cleared the gap but had BRTI outside the band, so that condition can be checked on new data.
- No new scans of Mar-Oct to rescue the bot. Live trading stays locked off.
