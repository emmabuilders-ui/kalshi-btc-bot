# Correction ML without quote-age inputs (the live bot reads a fresh order book, so ages are not comparable).
# Monthly out-of-fold Mar-Aug + frozen Dec-Aug fit for Sep-Oct, then the NO 70-99c gap rule for both versions.
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
S = pd.read_parquet('work/models_scored.parquet').sort_values('snap_ts').reset_index(drop=True)
DIRECT = ['dist', 'z', 'z60', 'secs_left', 'sig1', 'sig5', 'sig15', 'sig60', 'vol_ratio', 'vol_ratio_short', 'ret_10', 'ret_30', 'ret_60', 'ret_300', 'ret_900', 'dist_hi', 'dist_lo', 'hour_utc', 'dow']
CL = DIRECT + ['x', 'spread', 'log_cumvol']
mk = lambda: HGB(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=300, l2_regularization=1.0, early_stopping=False, random_state=0)
ts = lambda s: pd.Timestamp(s, tz='UTC').timestamp(); cut = ts('2026-09-01')
S['p_cl'] = np.nan
for m in pd.date_range('2026-03-01', '2026-08-01', freq='MS', tz='UTC'):
    a, b = m.timestamp(), (m + pd.offsets.MonthBegin(1)).timestamp(); tr = S[S.close_ts <= a]; te = S[(S.snap_ts >= a) & (S.snap_ts < b)]
    S.loc[te.index, 'p_cl'] = mk().fit(tr[CL], tr.res).predict_proba(te[CL])[:, 1]; print(m.date(), flush=True)
tr = S[S.close_ts <= cut]; te = S[S.snap_ts >= cut]; final = mk().fit(tr[CL], tr.res)
S.loc[te.index, 'p_cl'] = final.predict_proba(te[CL])[:, 1]
import pickle; pickle.dump((final, CL), open('work/corr_live_model.pkl', 'wb'))
S[['ticker', 'k', 'p_cl']].dropna().to_parquet('work/corr_live.parquet')
A = S[S.snap_ts >= ts('2026-03-01')].dropna(subset=['p_corr', 'p_cl', 'ask', 'bid']).copy()
A['per'] = np.where(A.snap_ts >= cut, 'Sep-Oct', 'Mar-Aug')
fee = lambda p: .07 * p * (1 - p)
for m in ('p_corr', 'p_cl'):
    print(m, 'Brier vs Kalshi x1000', {p: round(1000 * (np.mean((h[m] - h.res) ** 2) - np.mean((h.p_kalshi - h.res) ** 2)), 3) for p, h in A.groupby('per')})
    for G in (1, 2, 3, 5):
        px = 1 - A.bid
        for lo_ml, hi_ml in ((1, 14), (6, 14)):
            T = A.assign(px=px, win=1 - A.res)[(100 * ((1 - A[m]) - px - fee(px)) >= G) & (px >= .70) & (px <= .99) & A.min_left.between(lo_ml, hi_ml)]
            T = T.sort_values('min_left', ascending=False).drop_duplicates('ticker'); T['pnl'] = T.win - T.px - fee(T.px)
            s = []
            for p, h in T.groupby('per'):
                mo = h.groupby('month').pnl.mean()
                s.append(f'{p}: {len(h)} tr {100*h.win.mean():.1f} vs {100*h.px.mean():.1f} {100*h.pnl.mean():+.2f}c t{h.pnl.mean()/(h.pnl.std()/np.sqrt(len(h))):.1f} {(mo>0).sum()}/{len(mo)}')
            print(f'  NO gap>={G} {lo_ml}-{hi_ml}min |', ' | '.join(s))
