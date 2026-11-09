import numpy as np, pandas as pd
ts = lambda s: pd.Timestamp(s, tz='UTC').timestamp(); cut = ts('2026-09-01')
A = pd.read_parquet('work/models_scored.parquet', columns=['ticker', 'k', 'snap_ts', 'min_left', 'res', 'mid', 'ask', 'bid', 'day', 'month', 'p_kalshi', 'p_static', 'p_roll', 'p_vol', 'p_direct', 'p_corr', 'p_direct_wk', 'p_corr_wk', 'p_regime'])
A = A[A.snap_ts >= cut]
R = pd.read_parquet('work/recal15.parquet', columns=['ticker', 'k', 'p_kalshi_cal30d', 'p_corr_cal30d', 'p_direct_cal30d'])
A = A.merge(R, on=['ticker', 'k'], how='left')
for c in ['p_lasso_corr', 'p_gbm_corr_top5', 'p_gbm_dir_top10']: A = A.merge(pd.read_parquet(f'work/wf4_{c}.parquet'), on=['ticker', 'k'], how='left')
O = pd.read_parquet('work/models_of.parquet', columns=['ticker', 'k', 'p_corr_of', 'p_kof_wk', 'p_regime_of']); A = A.merge(O, on=['ticker', 'k'], how='left')
N = {'p_kalshi': 'Kalshi price', 'p_static': 'Static calibration', 'p_roll': 'Rolling 30-day recal (daily)', 'p_kalshi_cal30d': 'Kalshi recal every 15 min (30d)',
     'p_vol': 'Simple volatility model', 'p_direct': 'Direct ML', 'p_direct_wk': 'Direct ML weekly', 'p_corr': 'Correction ML', 'p_corr_wk': 'Correction ML weekly',
     'p_corr_cal30d': 'Correction ML + 15-min recal', 'p_direct_cal30d': 'Direct ML + 15-min recal', 'p_regime': 'Regime-switching',
     'p_corr_of': 'Correction ML + order flow', 'p_kof_wk': 'Kalshi + order flow weekly', 'p_regime_of': 'Regime + order flow',
     'p_lasso_corr': 'Auto features, sparse logistic', 'p_lasso_corr_wk': 'Auto features, sparse logistic weekly', 'p_gbm_corr_top5_wk': 'Auto features, trees top 5 weekly', 'p_gbm_dir_top10_wk': 'Auto features, trees BRTI only weekly'}
M = list(N)
A = A.dropna(subset=[m for m in M if m not in ('p_corr_of', 'p_kof_wk', 'p_regime_of')])  # order-flow models lack nothing in Sep-Oct, keep check below
print('rows', len(A), 'order-flow coverage', A.p_corr_of.notna().mean().round(3))
A = A.dropna(subset=M)
A['band'] = (np.clip(np.floor(A.mid * 100), 1, 99) // 10 * 10).astype(int)
A['time'] = pd.cut(A.min_left, [0, 2, 5, 9, 14], labels=['1-2 min', '3-5 min', '6-9 min', '10-14 min'])
rows = []
for keys, g in [(('All', 'All'), A)] + [((t, 'All'), h) for t, h in A.groupby('time', observed=True)] + [(('All', b), h) for b, h in A.groupby('band')] + [((t, b), h) for (t, b), h in A.groupby(['time', 'band'], observed=True)]:
    r = dict(time_left=keys[0], kalshi_price=keys[1] if keys[1] == 'All' else f'{keys[1]}-{keys[1]+9}c', snapshots=len(g), contracts=g.ticker.nunique(), actual_win_pct=round(100 * g.res.mean(), 2))
    kb = np.mean((g.p_kalshi - g.res) ** 2)
    best = None
    for m in M:
        b = np.mean((g[m] - g.res) ** 2)
        r[N[m] + ' | avg prob %'] = round(100 * g[m].mean(), 2)
        r[N[m] + ' | Brier vs Kalshi x1000'] = round(1000 * (b - kb), 3)
        if best is None or b < best[1]: best = (N[m], b)
    r['best model'] = best[0]; r['best vs Kalshi x1000'] = round(1000 * (best[1] - kb), 3)
    r['models beating Kalshi'] = sum(np.mean((g[m] - g.res) ** 2) < kb for m in M[1:])
    rows.append(r)
T = pd.DataFrame(rows)
# Calibration of each model by its OWN predicted price: predicted vs actual win rate, by time
cal = []
for m in M:
    b = (np.clip(np.floor(A[m] * 100), 0, 99) // 10 * 10).astype(int)
    for (t, bb), h in A.assign(pb=b).groupby(['time', 'pb'], observed=True):
        cal.append(dict(model=N[m], time_left=t, predicted_band=f'{bb}-{bb+9}%', snapshots=len(h), avg_pred_pct=round(100 * h[m].mean(), 2), actual_win_pct=round(100 * h.res.mean(), 2), miss_pts=round(100 * (h.res.mean() - h[m].mean()), 2)))
C = pd.DataFrame(cal)
summ = C.assign(w=C.snapshots * C.miss_pts.abs()).groupby(['model', 'time_left'], observed=True).apply(lambda h: round(h.w.sum() / h.snapshots.sum(), 3)).unstack()
summ['overall'] = C.assign(w=C.snapshots * C.miss_pts.abs()).groupby('model').apply(lambda h: round(h.w.sum() / h.snapshots.sum(), 3))
summ = summ.sort_values('overall')
pd.set_option('display.width', 250)
print(T[['time_left', 'kalshi_price', 'snapshots', 'actual_win_pct', 'best model', 'best vs Kalshi x1000', 'models beating Kalshi']].to_string())
print('\nAverage calibration miss (pts), each model bucketed by its own prediction:'); print(summ.to_string())
with pd.ExcelWriter('work/kalshi_vs_all_models_by_time_price.xlsx') as w:
    T.to_excel(w, sheet_name='Time_x_price', index=False); summ.to_excel(w, sheet_name='Calibration_miss_summary')
    C.to_excel(w, sheet_name='Calibration_by_own_price', index=False)
