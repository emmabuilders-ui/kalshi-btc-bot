import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
S = pd.read_parquet('work/features.parquet').sort_values('snap_ts').reset_index(drop=True)
DIRECT = ['dist', 'z', 'z60', 'secs_left', 'sig1', 'sig5', 'sig15', 'sig60', 'vol_ratio', 'vol_ratio_short',
          'ret_10', 'ret_30', 'ret_60', 'ret_300', 'ret_900', 'dist_hi', 'dist_lo', 'hour_utc', 'dow']
CORR = DIRECT + ['x', 'spread', 'log_cumvol', 'ask_age_c', 'bid_age_c']
def mk(): return HGB(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=300, l2_regularization=1.0, early_stopping=False, random_state=0)
def fitpred(tr, te, cols):
    m = mk().fit(tr[cols], tr.res); return m.predict_proba(te[cols])[:, 1]
ts = lambda s: pd.Timestamp(s, tz='UTC').timestamp()
for c in ['p_direct', 'p_corr', 'p_direct_wk', 'p_corr_wk']: S[c] = np.nan
# A) Out-of-fold, Dec-Aug: expanding monthly folds Feb..Aug (train on contracts closed before month start)
for m in pd.date_range('2026-02-01', '2026-08-01', freq='MS', tz='UTC'):
    a, b = m.timestamp(), (m + pd.offsets.MonthBegin(1)).timestamp()
    tr = S[S.close_ts <= a]; te = S[(S.snap_ts >= a) & (S.snap_ts < b)]
    S.loc[te.index, 'p_direct'] = fitpred(tr, te, DIRECT); S.loc[te.index, 'p_corr'] = fitpred(tr, te, CORR); print('oof', m.date(), len(tr), len(te), flush=True)
# B) Holdout, frozen: train Dec-Aug once, predict Sep-Oct
cut = ts('2026-09-01'); tr = S[S.close_ts <= cut]; te = S[S.snap_ts >= cut]
S.loc[te.index, 'p_direct'] = fitpred(tr, te, DIRECT); S.loc[te.index, 'p_corr'] = fitpred(tr, te, CORR)
# C) Holdout, weekly refit on everything closed before each week
for w in pd.date_range('2026-09-01', '2026-10-26', freq='7D', tz='UTC'):
    a, b = w.timestamp(), (w + pd.Timedelta(days=7)).timestamp()
    tr = S[S.close_ts <= a]; te = S[(S.snap_ts >= a) & (S.snap_ts < b)]
    if te.empty: continue
    S.loc[te.index, 'p_direct_wk'] = fitpred(tr, te, DIRECT); S.loc[te.index, 'p_corr_wk'] = fitpred(tr, te, CORR); print('wk', w.date(), flush=True)
# D) Regime-switching gate: 27 cells (vol tercile x minutes-left group x price group); expert with lowest Brier on prior out-of-fold rows
experts = ['p_kalshi', 'p_roll', 'p_direct', 'p_corr']
oof = S[(S.snap_ts >= ts('2026-02-01')) & (S.snap_ts < cut) & S.p_roll.notna()]
q1, q2 = oof.sig15.quantile([1/3, 2/3])
def cell(d):
    v = np.where(d.sig15 < q1, 0, np.where(d.sig15 < q2, 1, 2)); t = np.where(d.min_left <= 3, 0, np.where(d.min_left <= 8, 1, 2))
    p = np.where(d.mid < 0.2, 0, np.where(d.mid < 0.8, 1, 2)); return v * 9 + t * 3 + p
S['cell'] = cell(S); oof = S.loc[oof.index]
def gate(hist):
    g = {}
    for c, h in hist.groupby('cell'):
        h = h.dropna(subset=experts)
        if len(h) < 500: g[c] = 'p_kalshi'; continue
        g[c] = min(experts, key=lambda e: np.mean((h[e] - h.res) ** 2))
    return g
S['p_regime'] = np.nan; gates = {}
for m in pd.date_range('2026-03-01', '2026-08-01', freq='MS', tz='UTC'):  # gate itself out-of-fold
    a, b = m.timestamp(), (m + pd.offsets.MonthBegin(1)).timestamp()
    g = gate(oof[oof.close_ts <= a]); te = S[(S.snap_ts >= a) & (S.snap_ts < b)].dropna(subset=experts)
    S.loc[te.index, 'p_regime'] = [te.at[i, g.get(c, 'p_kalshi')] for i, c in zip(te.index, te.cell)]
g = gate(oof); gates['holdout'] = g; te = S[S.snap_ts >= cut].dropna(subset=experts)
S.loc[te.index, 'p_regime'] = [te.at[i, g.get(c, 'p_kalshi')] for i, c in zip(te.index, te.cell)]
print('holdout gate choices:', pd.Series(g).value_counts().to_dict())
pd.Series(g).rename('expert').to_frame().to_csv('work/regime_gate.csv')
S.to_parquet('work/models_scored.parquet')
