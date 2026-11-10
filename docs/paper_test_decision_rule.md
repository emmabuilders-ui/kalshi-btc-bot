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
