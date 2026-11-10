import numpy as np, pandas as pd, itertools
T0 = 1764460800
B = pd.Series(np.load('work/brti.npy')).ffill().bfill().to_numpy(); lB = np.log(B)
L = np.zeros(len(B)); L[1:] = np.diff(lB)
c1 = np.concatenate([[0], np.cumsum(L)]); c2 = np.concatenate([[0], np.cumsum(L ** 2)]); ca = np.concatenate([[0], np.cumsum(np.abs(L))])
S = pd.read_parquet('work/replay.parquet').sort_values('snap_ts').reset_index(drop=True)
e = (S.snap_ts.to_numpy() - T0).astype(np.int64); E = e + 1
Teff = np.clip(S.secs_left.to_numpy() - 40, 1, None); dist = np.log(S.brti / S.strike).to_numpy()
F = {}
LAGS = [1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 900, 1800, 3600]
WINS = [10, 30, 60, 120, 300, 600, 900, 1800, 3600]
for l in LAGS: F[f'ret_{l}'] = lB[e] - lB[e - l]
vol = {}
for w in WINS:
    a = E - w; s1 = c1[E] - c1[a]; s2 = c2[E] - c2[a]
    vol[w] = np.sqrt(np.maximum(s2 / w - (s1 / w) ** 2, 1e-14)); F[f'vol_{w}'] = vol[w]
    F[f'absmove_{w}'] = (ca[E] - ca[a]) / w  # mean absolute 1s move
    F[f'eff_{w}'] = np.abs(s1) / np.maximum(ca[E] - ca[a], 1e-12)  # trend efficiency
for w in [60, 300, 900, 3600]:
    rmax = pd.Series(lB).rolling(w, min_periods=1).max().to_numpy(); rmin = pd.Series(lB).rolling(w, min_periods=1).min().to_numpy()
    F[f'from_hi_{w}'] = lB[e] - rmax[e]; F[f'from_lo_{w}'] = lB[e] - rmin[e]; F[f'range_{w}'] = rmax[e] - rmin[e]
    F[f'strike_in_range_{w}'] = (np.log(S.strike.to_numpy()) - rmin[e]) / np.maximum(rmax[e] - rmin[e], 1e-9)
F['dist'] = dist; F['secs_left'] = S.secs_left.to_numpy().astype(float)
for w in WINS: F[f'z_{w}'] = dist / (vol[w] * np.sqrt(Teff))
for l in [10, 30, 60, 300, 900]:
    for w in [60, 300, 900, 3600]: F[f'retz_{l}_{w}'] = F[f'ret_{l}'] / (vol[w] * np.sqrt(l))
for a, b in itertools.combinations([10, 60, 300, 900, 3600], 2): F[f'volratio_{a}_{b}'] = vol[a] / vol[b]
# drift-projected distance: extrapolate recent drift to settlement
for l in [60, 300, 900]: F[f'proj_dist_{l}'] = dist + F[f'ret_{l}'] / l * Teff
# Kalshi price history within the contract (earlier minute snapshots of the same contract, as of their own seconds)
R = pd.read_parquet('work/snaps_raw.parquet')[['ticker', 'k', 'ask', 'bid', 'ask_age', 'bid_age']]
R['mid_h'] = np.where((R.ask_age < 20) & (R.bid_age < 20), (R.ask + R.bid) / 2, np.nan)
P = R.pivot(index='ticker', columns='k', values='mid_h')
mid = S.mid.to_numpy()
for d in [1, 2, 3, 5]:
    prev = np.array([P.at[t, k - d] if (k - d) >= 1 and t in P.index else np.nan for t, k in zip(S.ticker, S.k)], dtype=float)
    F[f'kmid_chg_{d}m'] = mid - prev
F['kmid'] = mid; F['klogit'] = np.log(np.clip(mid, 1e-3, 1 - 1e-3) / (1 - np.clip(mid, 1e-3, 1 - 1e-3))); F['kspread'] = S.spread.to_numpy()
F['k_vs_vol'] = F['klogit'] - np.log(np.clip(pd.Series(F['z_900']).pipe(lambda z: 0.5 * (1 + np.tanh(z * 0.8))), 1e-3, 1 - 1e-3) / (1 - np.clip(0.5 * (1 + np.tanh(F['z_900'] * 0.8)), 1e-3, 1 - 1e-3)))
# previous contract's move (known at open)
M = pd.read_parquet('work/markets_audited.parquet')[['ticker', 'open_ts', 'strike', 'brti_avg']].sort_values('open_ts')
M['prev_move'] = np.log(M.strike / M.strike.shift(1)); pm = dict(zip(M.ticker, M.prev_move))
F['prev_contract_move'] = S.ticker.map(pm).to_numpy()
t = pd.to_datetime(S.snap_ts, unit='s'); F['hour'] = t.dt.hour.to_numpy(); F['dow'] = t.dt.dayofweek.to_numpy()
X = pd.DataFrame(F); X = X.replace([np.inf, -np.inf], np.nan)
out = pd.concat([S[['ticker', 'k', 'snap_ts', 'open_ts', 'close_ts', 'min_left', 'res', 'mid', 'ask', 'bid', 'day', 'month', 'period']], X], axis=1)
out.to_parquet('work/autofeat.parquet'); print(out.shape, 'features', X.shape[1])
