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
