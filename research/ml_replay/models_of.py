import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
S = pd.read_parquet('work/features.parquet'); F = pd.read_parquet('work/flow.parquet')
S = S[S.src == 'trades'].merge(F, on=['ticker', 'k'], how='inner').sort_values('snap_ts').reset_index(drop=True)
S['of_px_vs_mid'] = S.last_px - S.mid
for w in (10, 30, 60, 180): S[f'of_vwap_vs_mid_{w}'] = (S[f'of_vwap_{w}'] - S.mid).fillna(0)
OF = ['of_signed_open'] + [f'of_{a}_{w}' for w in (10, 30, 60, 180) for a in ('imb', 'vol', 'n')] + ['of_big_180', 'of_dpx_30', 'of_dpx_60', 'of_dpx_180', 'of_px_vs_mid'] + [f'of_vwap_vs_mid_{w}' for w in (10, 30, 60, 180)]
DIRECT = ['dist', 'z', 'z60', 'secs_left', 'sig1', 'sig5', 'sig15', 'sig60', 'vol_ratio', 'vol_ratio_short', 'ret_10', 'ret_30', 'ret_60', 'ret_300', 'ret_900', 'dist_hi', 'dist_lo', 'hour_utc', 'dow']
CORR = DIRECT + ['x', 'spread', 'log_cumvol', 'ask_age_c', 'bid_age_c']
SETS = {'p_direct': DIRECT, 'p_direct_of': DIRECT + OF, 'p_corr': CORR, 'p_corr_of': CORR + OF, 'p_kof': ['x', 'secs_left', 'spread'] + OF}
mk = lambda: HGB(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=300, l2_regularization=1.0, early_stopping=False, random_state=0)
ts = lambda s: pd.Timestamp(s, tz='UTC').timestamp(); cut = ts('2026-09-01')
for c in SETS: S[c] = np.nan; S[c + '_wk'] = np.nan
def run(tr, te, suf=''):
    for c, cols in SETS.items(): S.loc[te.index, c + suf] = mk().fit(tr[cols], tr.res).predict_proba(te[cols])[:, 1]
for m in ['2026-02-01', '2026-03-01', '2026-04-01', '2026-08-01']:
    a = ts(m); b = (pd.Timestamp(m, tz='UTC') + pd.offsets.MonthBegin(1)).timestamp()
    run(S[S.close_ts <= a], S[(S.snap_ts >= a) & (S.snap_ts < b)]); print('oof', m, flush=True)
run(S[S.close_ts <= cut], S[S.snap_ts >= cut])
for w in pd.date_range('2026-09-01', '2026-10-26', freq='7D', tz='UTC'):
    te = S[(S.snap_ts >= w.timestamp()) & (S.snap_ts < (w + pd.Timedelta(days=7)).timestamp())]
    if len(te): run(S[S.close_ts <= w.timestamp()], te, '_wk')
print('trained', flush=True)
# regime gate over kalshi, rolling recal, corr+OF, direct+OF using out-of-fold rows (Feb-Apr, Aug)
EX = ['p_kalshi', 'p_roll', 'p_corr_of', 'p_direct_of']
oof = S[(S.snap_ts >= ts('2026-02-01')) & (S.snap_ts < cut)].dropna(subset=EX)
q1, q2 = oof.sig15.quantile([1 / 3, 2 / 3])
S['cell'] = (np.where(S.sig15 < q1, 0, np.where(S.sig15 < q2, 1, 2)) * 9 + np.where(S.min_left <= 3, 0, np.where(S.min_left <= 8, 1, 2)) * 3 + np.where(S.mid < .2, 0, np.where(S.mid < .8, 1, 2)))
oof = S.loc[oof.index]
gate = {c: (min(EX, key=lambda e: np.mean((h[e] - h.res) ** 2)) if len(h) >= 500 else 'p_kalshi') for c, h in oof.groupby('cell')}
te = S[S.snap_ts >= cut].dropna(subset=EX); S['p_regime_of'] = np.nan
S.loc[te.index, 'p_regime_of'] = [te.at[i, gate.get(c, 'p_kalshi')] for i, c in zip(te.index, te.cell)]
print('gate', pd.Series(gate).value_counts().to_dict())
pd.Series(gate).rename('expert').to_frame().to_csv('work/regime_gate_of.csv')
S.to_parquet('work/models_of.parquet')
