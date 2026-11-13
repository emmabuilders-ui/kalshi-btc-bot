# Lock-in bot: shipped-code replay (2026-11-13)

`replay_bot.py` runs the bot's own decision code (`kalshi_btc_bot.lockin_paper`: `inputs`, `evaluate`, `limit_price`)
over the Kalshi daily trade files (`data-download/kalshi-daily/`, Aug 1 - Oct 25 2026, the months with trade prints) and
the 1-second BRTI. Price = the last Kalshi taker print on the bought side within 10 s before the check second (the
settlement-average test's definition; the paper bot pays the live ask, which can be worse). First qualifying check per
contract at 50/40/30/20/10 s left, favoured side only, 5-point margin, 20 contracts, settled on markets.csv. The spread
skip cannot be replayed from prints.

    PYTHONPATH=. python research/lockin/replay_bot.py data-download/kalshi-daily data-download/brti/BRTI_1sec_2026-08-01_to_2026-10-25.zip out.csv 2026-08-01 2026-10-25

| Period | Trades | Win % | Avg price | Win − price | EV / contract (SE) | EV with fills 3c worse | Trades/day |
|---|---|---|---|---|---|---|---|
| Aug 2026 | 124 | 82.3% | 68.6c | +13.7 pts | +12.5c (3.6) | +9.6c | 4.4 |
| Sep - Oct 25 2026 | 317 | 79.2% | 71.1c | +8.1 pts | +6.9c (2.2) | +4.0c | 6.2 |

By first-fire second (Aug-Oct): 50 s 237 trades +6.7c · 40 s 99 +8.8c · 30 s 46 +4.2c · 20 s 38 +12.7c · 10 s 21 +27.9c.
By side: YES 196 trades +9.4c, NO 245 +7.7c. Months: Aug +12.5c, Sep +4.6c, Oct +11.9c.

Against the settlement-average test (strategy/settlement_average_test.xlsx, sheet under60_rule, age 10 s, margin 5c,
"all"): Sep-Oct 160 trades, 76.2% vs 67.8c, +7.2c. The shipped code finds the same edge (+6.9c) and the same kind of
trade (favourites around 70c, 4 in 5 won) but twice the trades, so its quote definition is looser than the test's
(that script was not kept; a likely difference is one print per contract vs one print per side). The 5-point margin was
not re-chosen after this replay. The paper test is the real check: docs/paper_test_decision_rule.md.

Trade list: strategy/lockin/lockin_bot_replay_aug_oct_2026.csv (project files).
