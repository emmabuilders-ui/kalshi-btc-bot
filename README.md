# Kalshi BTC 15-minute strategy

An indicator engine, an exit-method backtester and a paper trader for Kalshi's
15-minute Bitcoin YES/NO markets (series `KXBTC15M`).

**Live trading is off.** There is no live order routing in this version. The
`live` command refuses to run unless `trading.live_trading_enabled = true` is set in the
config **and** `KALSHI_LIVE_TRADING=I_ACCEPT_REAL_MONEY_RISK` is set in the
environment. Even with both set, it stops with "not implemented". Real order routing
will be added and reviewed separately, after paper trading.

## Quick start

```bash
pip install -e ".[dev]"
python -m pytest                                    # 29 tests

# Pipeline test on SYNTHETIC data (its P&L means nothing)
python -m kalshi_btc_bot synth --out data/synthetic --hours 48
python -m kalshi_btc_bot backtest --data data/synthetic --out results/synthetic

# Backtest on real data (format below)
python -m kalshi_btc_bot backtest --data path/to/folder --out results/real

# Paper trading on live Kalshi + BTC data. No orders are sent.
python -m kalshi_btc_bot paper --out runs/paper --minutes 240
```

A paper session also records `btc.csv`, `quotes.csv` and `markets.csv` into
`runs/paper/recorded_data/`. You can backtest that folder later, so a few days of
paper trading builds a real executable-price dataset.

## How the spec maps to the code

| Spec section | Code |
|---|---|
| 2. Asymmetrical Bollinger entry (MA50/250/500, SD250, Z, separate upper/lower breaches) | `indicators.EntryIndicator` |
| 3. Golden cross within 120 s of a lower breach, YES between 0.30 and 0.70, not expired | `EntryIndicator` + `engine.StrategyEngine._check_entry` |
| 4. Exit bands: 20 x 5 s candles, 2 SD | `indicators.ExitBands` |
| 5–6. Trailing stop: activates at BTC >= upper exit band, exits after 60 s continuously below middle (12 x 5 s closes), resets on a close at or above middle, else hold to settlement | `indicators.TrailingStop` |
| 7. Four-way exit comparison on identical entries | `backtest.run_backtest` |
| 8. Quantity, position limits, duplicate-order guard, logging, paper trading | `config.TradingConfig`, `broker.PaperBroker`, `engine.EventLog`, `paper.run_paper` |

The entry model (`EntryIndicator`) and the exit model (`ExitBands` + `TrailingStop`)
are separate classes with separate configs. `IndicatorHub` computes both once per
second and feeds every strategy engine. The backtester and the paper trader run
the same classes, so they produce the same signals on the same data.

### The four exit methods (`config.DEFAULT_EXIT_METHODS`)

| Name | Bands | Confirmation | Needs upper-band activation |
|---|---|---|---|
| `hold_to_settlement` | none | n/a | n/a |
| `orig_250s_mid_30s` | 250 x 1 s (MA250 middle) | 30 s below MA250 | no (configurable) |
| `bb20x5s_60s` | 20 x 5 s | 60 s (12 closes) | yes |
| `bb20x1m_60s` | 20 x 1 min | 60 s (1 close) | yes |

The spec does not say whether the original 250-second exit waited for an upper-band
touch. I read it as a plain middle-band exit. Set `require_activation = true` on that
method to gate it like the others.

## Execution model

* Every order fills against the Kalshi quote **1 second after** the decision (`latency_s`),
  never against the quote the decision used.
* **Executable pricing:** buys walk the YES asks up to a limit of signal price + $0.02 and
  are immediate-or-cancel (partial fills are allowed, and a missed fill is logged). Exits
  sell into the YES bids and keep working until filled or the market closes.
* **Last-trade pricing** fills everything at the last trade, which matches the Oct 26
  exploration. The report shows both.
* Fees use Kalshi's taker formula `ceil(0.07 × C × P × (1−P))` to the cent, on buys and
  early sells. Settlement has no fee. Check the current fee schedule for this series.
* Limits: one entry per contract (`max_entries_per_contract`) and `max_open_positions`
  per engine. Order IDs are unique, so the broker rejects a duplicate submission. With one
  entry per contract, every exit method sees the same entries, and each run verifies this.

## Settlement

Kalshi settles on the CF Benchmarks **BRTI 60-second average**, not on any one
exchange. The backtester uses, in order of preference:

1. the official `result` column in `markets.csv`;
2. otherwise a 60-second average of the `brti` column in `btc.csv`;
3. otherwise a 60-second average of spot. This is only an estimate, and the report says so.

YES if the average is `>=` the strike (`settlement.yes_if`). Confirm this against the
contract terms. Paper trading waits up to an hour for Kalshi's official result before it
falls back to the estimate.

Indicators use spot. Paper trading defaults to the median of Coinbase, Kraken and
Bitstamp, which are BRTI constituents. That is closer to BRTI than one venue, but it is
still not BRTI.

## Data format for backtests

`btc.csv`: one row per second (gaps are forward-filled and counted in the report)

```
timestamp,price,brti        # timestamp: epoch s, epoch ms, or ISO-8601; brti optional
```

`markets.csv`

```
ticker,open_time,close_time,strike,result     # result (yes/no) optional
```

`quotes.csv`: Kalshi YES top of book. Any cadence works; the latest quote within 10 s is used.

```
timestamp,ticker,yes_bid,yes_ask,yes_bid_size,yes_ask_size,last_price   # prices in dollars
```

For a last-trade-only dataset like the Oct 26 one, leave the bid and ask empty, fill
`last_price`, and set `entry.yes_price_field = "last"`. Only the last-trade table will
be meaningful.

## Outputs

* `report.md`: the exit-method comparison table (gross, fees, net) for both pricing modes
* `summary.csv`, `trades.csv`: per method and per trade (entry and exit times, prices, reasons, P&L)
* `events.jsonl`: every signal (with indicator values and the reason for any rejection),
  order, fill, trailing-stop activation, exit and settlement, with time remaining

## Futures and ETF mode

This mode runs the same buy signal and the same four exits on instruments whose P&L moves
one-for-one with price. Presets (`linear.PRESETS`):

| Symbol | Contract | $ per point | Commission per side | Default size | Hours |
|---|---|---:|---:|---:|---|
| MNQ | Micro E-mini Nasdaq-100 | 2 | $0.62 | 1 | 24h |
| MES | Micro E-mini S&P 500 | 5 | $0.62 | 1 | 24h |
| MBT | CME Micro Bitcoin (0.1 BTC) | 0.1 | $2.50 | 1 | 24h |
| TQQQ, SQQQ, UPRO, SOXL | 3x ETFs | 1 per share | $0 | 100 shares | 9:30–16:00 ET |

Commissions are typical retail figures, so check your broker's. Any other symbol works if
you define it in an `[instrument]` section.

```bash
python -m kalshi_btc_bot synth-linear --symbol MNQ --price 21000 --hours 48 --out data/syn_mnq   # SYNTHETIC
python -m kalshi_btc_bot backtest-linear --data path/to/price.csv --symbol MNQ --out results/mnq
python -m kalshi_btc_bot backtest-linear --data path/to/price.csv --symbol TQQQ --allow-short
```

Data: `price.csv` with columns `timestamp,price[,bid,ask]`, one row per second. Buys
fill at the ask and sells at the bid, 1 second after each decision. Without bid and ask
columns, half a tick of spread is assumed on each side.

Differences from Kalshi mode:

* **Hard stop:** entry minus 3 × SD250 at the signal (`hard_stop_sd`), or a fixed
  `hard_stop_points`. A binary contract caps the loss at the price paid, and these
  instruments don't.
* **Time stop:** a trade closes after 900 s (`max_hold_s`), standing in for the 15-minute
  settlement. "Hold to settlement" becomes "hold until the time stop".
* **No YES-price filter.** After an entry, new signals are locked out for `lockout_s`
  (default 900 s), so all four exits trade identical entries.
* **Shorts (optional, `--allow-short`):** an upper-band breach followed by an MA50/MA250
  death cross within 120 s. The trailing stop is mirrored: it activates at the lower exit
  band and exits after 60 s of closes above the middle band.
* **ETF hours:** ETFs only enter during regular hours, take no new entries in the last
  15 minutes, and are flattened 60 s before the close. For every instrument, a data gap
  longer than 5 s resets the indicators and closes open trades at the last price, so
  overnight and weekend gaps never turn into fake flat prices.

Settings go in a TOML file passed with `--linear-config`:

```toml
[linear]
max_hold_s = 900
hard_stop_sd = 3.0
allow_short = false

[instrument]          # optional overrides for the preset
commission_per_side = 0.85
```

There is no live or paper trading for futures and ETFs yet. That needs a broker
connection, such as Interactive Brokers or Alpaca.

## Known limits

* The Kalshi and exchange clients (`live_data.py`) follow the public APIs but could not be
  run against the real endpoints from the build environment. Check the first paper
  session's log.
* Paper trading polls REST once per second. Use a WebSocket feed for lower latency.
* Synthetic data (`synth`) only exercises the code. Its P&L tells you nothing.
* The Oct 26 numbers (12 entries) are too few to choose an exit method. Run the comparison
  on weeks of recorded executable-price data first.
