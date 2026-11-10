import numpy as np, pandas as pd
T0 = 1764460800; B = np.load('work/brti.npy')
M = pd.read_parquet('work/markets.parquet'); A = pd.read_csv('work/zip_audit.csv')
M['res'] = M.result.str.upper().map({'YES': 1, 'NO': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0})
idx = (M.close_ts.to_numpy()[:, None] - T0) + np.arange(-60, 0)[None, :]
ok = (idx >= 0).all(1) & (idx < len(B)).all(1)
W = np.full(idx.shape, np.nan); W[ok] = B[idx[ok]]
M['n_prints'] = np.isfinite(W).sum(1); M['brti_avg'] = np.nanmean(W, 1)
M['brti_res'] = (M.brti_avg >= M.strike).astype(float); M.loc[M.n_prints == 0, 'brti_res'] = np.nan
M['month'] = pd.to_datetime(M.close_ts, unit='s').dt.strftime('%Y-%m')
full = M[(M.n_prints == 60) & M.res.notna()]
mis = full[full.brti_res != full.res]
print('markets', len(M), 'with result', M.res.notna().sum(), 'full 60 prints', len(full), 'result mismatches', len(mis))
ev = full.dropna(subset=['expiration_value']); d = (ev.brti_avg - ev.expiration_value).abs()
print('expiration_value check: n', len(ev), 'within 1c', (d < 0.011).mean(), 'max diff', d.max())
print(mis[['ticker', 'strike', 'brti_avg', 'expiration_value', 'res']].head(10))
print('close margin of mismatches ($):', (mis.brti_avg - mis.strike).abs().describe())
S = pd.read_parquet('work/snaps_raw.parquet')
S['fresh'] = (S.ask_age < 20) & (S.bid_age < 20) & S.ask.notna() & S.bid.notna()
S = S.merge(M[['ticker', 'month']], on='ticker')
print(S.groupby('month').agg(snaps=('k', 'size'), fresh=('fresh', 'mean')).round(3))
A['month'] = A.day.str[:7]
print(A.groupby('month').frac_subcent.mean().round(3))
M.to_parquet('work/markets_audited.parquet')
