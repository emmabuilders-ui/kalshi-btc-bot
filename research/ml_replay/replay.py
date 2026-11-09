import numpy as np, pandas as pd
T0 = 1764460800; B = np.load('work/brti.npy')
Bf = pd.Series(B).ffill().to_numpy()
L = np.full(len(Bf), np.nan); L[1:] = np.diff(np.log(Bf))
c1 = np.concatenate([[0], np.nancumsum(np.nan_to_num(L))]); c2 = np.concatenate([[0], np.nancumsum(np.nan_to_num(L) ** 2)])
def sd(end, w):  # std of 1s log returns over [end-w, end)
    a = end - w; s1 = c1[end] - c1[a]; s2 = c2[end] - c2[a]
    return np.sqrt(np.maximum(s2 / w - (s1 / w) ** 2, 0))
M = pd.read_parquet('work/markets_audited.parquet')
M = M[(M.n_prints == 60) & M.strike.notna() & (M.strike > 1000)]
M = M[~M.ticker.isin(['KXBTC15M-26APR241400-00'])]
S = pd.read_parquet('work/snaps_raw.parquet').merge(M[['ticker', 'open_ts', 'close_ts', 'strike', 'res', 'month']], on='ticker')
S = S[(S.ask_age < 20) & (S.bid_age < 20) & S.ask.notna() & S.bid.notna()].copy()
S = S[(S.ask > S.bid - 1e-9)]  # drop crossed quotes (stale side)
e = (S.snap_ts.to_numpy() - T0).astype(np.int64)  # info strictly at or before snap second
S['brti'] = Bf[e]
S['sig15'] = sd(e + 1, 900); S['sig60'] = sd(e + 1, 3600)
S['secs_left'] = S.close_ts - S.snap_ts; S['min_left'] = 15 - S.k
S['dist'] = np.log(S.brti / S.strike)
S['mid'] = (S.ask + S.bid) / 2; S['spread'] = S.ask - S.bid
S['period'] = np.where(S.snap_ts < pd.Timestamp('2026-09-01', tz='UTC').timestamp(), 'Dec-Aug', 'Sep-Oct')
S['day'] = pd.to_datetime(S.snap_ts, unit='s').dt.floor('D')
S.to_parquet('work/replay.parquet')
print(len(S), 'fresh snapshots,', S.ticker.nunique(), 'contracts')
print(S.groupby('period').agg(snaps=('k', 'size'), contracts=('ticker', 'nunique'), spread_med=('spread', 'median')))
