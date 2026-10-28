# Exit method comparison

**Data:** SYNTHETIC data (pipeline test only; P&L is meaningless)

- Entries identical across exit methods: **True**

## Executable prices (buy at YES ask, sell at YES bid, book size limits, fees)

| Exit method | Entries | Early exits | Winners | Gross P&L | Fees | Net P&L |
|---|---:|---:|---:|---:|---:|---:|
| hold_to_settlement | 96 | 0 | 46 | $-450.82 | $150.37 | $-601.19 |
| orig_250s_mid_30s | 96 | 88 | 26 | $-390.11 | $268.20 | $-658.31 |
| bb20x5s_60s | 96 | 80 | 36 | $-566.39 | $245.72 | $-812.11 |
| bb20x1m_60s | 96 | 22 | 42 | $-202.64 | $180.13 | $-382.77 |

## Last-trade prices (same method as the Oct 26 exploration)

| Exit method | Entries | Early exits | Winners | Gross P&L | Fees | Net P&L |
|---|---:|---:|---:|---:|---:|---:|
| hold_to_settlement | 102 | 0 | 51 | $96.00 | $170.87 | $-74.87 |
| orig_250s_mid_30s | 102 | 94 | 37 | $-32.00 | $307.03 | $-339.03 |
| bb20x5s_60s | 102 | 86 | 45 | $-147.00 | $286.81 | $-433.81 |
| bb20x1m_60s | 102 | 21 | 51 | $293.00 | $201.38 | $91.62 |

Per-trade detail is in trades.csv, every signal/order/fill/settlement in events.jsonl.
