"""KXBTCD hourly above/below snapshots: 1-min candle ask/bid, minutes 1-30 before close, BRTI at the minute end,
settlement = BRTI 60-s average before close (recomputed), result from the data (matches 100% on Sep-Oct per README)."""
import numpy as np, pandas as pd
T0 = 1764460800; B = np.load('work/brti.npy'); Bf = pd.Series(B).ffill().to_numpy()
q = pd.read_parquet('/mnt/project-files/data-download/kalshi-hourly-range/btcd_quotes_1min_last31_dec2025_oct2026.parquet')
q = q.dropna(subset=['yes_ask', 'yes_bid']).copy()
q['min_left'] = ((q.close - q.t) // 60).astype(int)
q = q[(q.min_left >= 1) & (q.min_left <= 30)]
q = q[q.yes_ask >= q.yes_bid - 1e-9]
e = (q.t.to_numpy().astype(np.int64) - T0)
ok = (e >= 0) & (e < len(Bf)); q = q[ok]; e = e[ok]
q['brti'] = Bf[e]
idx = (q.close.to_numpy().astype(np.int64) - T0)[:, None] + np.arange(-60, 0)[None, :]
okc = (idx >= 0).all(1) & (idx < len(B)).all(1)
W = np.full(idx.shape, np.nan); W[okc] = B[idx[okc]]
q['n_prints'] = np.isfinite(W).sum(1); q['settle'] = np.nanmean(W, 1)
q = q[q.n_prints == 60].copy()
q['res_brti'] = (q.settle >= q.strike).astype(int)
r = q.result.astype(str).str.upper().map({'YES': 1, 'NO': 0})
print('result listed', r.notna().mean().round(3), 'agree with BRTI', (r[r.notna()] == q.res_brti[r.notna()]).mean().round(4))
q['res'] = q.res_brti
q = q.rename(columns={'yes_ask': 'ask', 'yes_bid': 'bid', 't': 'snap_ts', 'close': 'close_ts'})
q['mid'] = (q.ask + q.bid) / 2
q['period'] = np.where(q.snap_ts < pd.Timestamp('2026-09-01', tz='UTC').timestamp(), 'Dec-Aug', 'Sep-Oct')
q['month'] = pd.to_datetime(q.snap_ts, unit='s').dt.strftime('%Y-%m')
q['dist'] = np.log(q.brti / q.strike)
q[['ticker', 'close_ts', 'strike', 'snap_ts', 'min_left', 'ask', 'bid', 'mid', 'brti', 'settle', 'res', 'period', 'month', 'dist']].to_parquet('work/hourly.parquet')
print(len(q), 'hourly snapshots', q.ticker.nunique(), 'markets'); print(q.groupby('period').size())
