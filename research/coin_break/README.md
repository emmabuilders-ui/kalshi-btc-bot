# Break-with-gate and lock-in on ETH, SOL and XRP (2026-11-12)

Cross-coin replay behind `kalshi_btc_bot/coin_break_paper.py`. Results: `strategy/other_coins_lockin_break_test.md` in the project files.

Run order (from this folder, with `KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY` set, read-only key):

1. `python3 markets.py KXETH15M` (and KXSOL15M, KXXRP15M): contract lists, strikes and results.
2. `python3 fetch_idx.py`: 1-second CF index per hour (`idx/<COIN>/`), about 1.3 requests a second, a few hours.
3. `python3 fetch_tr.py`: every taker trade in the last 200 s before each close (`tr/<COIN>/`).
4. `python3 replay.py ETH` (and SOL, XRP): writes `out_<COIN>_lockin.csv`, `out_<COIN>_break.csv`, `out_<COIN>_brier.csv`.
5. `python3 summ.py ETH SOL XRP`: tables by period (Dec-Aug, Sep-Oct, Nov) with win % next to the average price.
6. `python3 pack.py ETH`: bundles the raw data as gzipped CSV.

Prices in the replay are the last real taker print on the side bought (within 10 s for lock-in, 15 s for break), not a live ask.
Background jobs on the cloud container are stopped after 2 hours; the two fetchers resume where they left off.
