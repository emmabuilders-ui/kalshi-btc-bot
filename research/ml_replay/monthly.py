# Kalshi price vs every model, by month (and month x time-left x price). Sep-Oct = holdout for all models;
# Mar-Aug only for models that have out-of-sample values there (static and vol-model are in-sample before Sep, flagged).
import numpy as np, pandas as pd
A = pd.read_parquet('work/models_scored.parquet', columns=['ticker','k','snap_ts','min_left','res','mid','month','p_kalshi','p_static','p_roll','p_vol','p_direct','p_corr','p_direct_wk','p_corr_wk','p_regime'])
A = A[A.month >= '2026-03']
R = pd.read_parquet('work/recal15.parquet', columns=['ticker','k','p_kalshi_cal30d','p_corr_cal30d','p_direct_cal30d'])
A = A.merge(R, on=['ticker','k'], how='left')
for c in ['p_lasso_corr','p_gbm_corr_top5','p_gbm_dir_top10']: A = A.merge(pd.read_parquet(f'work/wf4_{c}.parquet'), on=['ticker','k'], how='left')
O = pd.read_parquet('work/models_of.parquet', columns=['ticker','k','p_corr_of','p_kof_wk','p_regime_of']); A = A.merge(O, on=['ticker','k'], how='left')
N = {'p_kalshi':'Kalshi price','p_static':'Static calibration','p_roll':'Rolling 30-day recal (daily)','p_kalshi_cal30d':'Kalshi recal every 15 min (30d)',
     'p_vol':'Simple volatility model','p_direct':'Direct ML','p_direct_wk':'Direct ML weekly','p_corr':'Correction ML','p_corr_wk':'Correction ML weekly',
     'p_corr_cal30d':'Correction ML + 15-min recal','p_direct_cal30d':'Direct ML + 15-min recal','p_regime':'Regime-switching',
     'p_corr_of':'Correction ML + order flow','p_kof_wk':'Kalshi + order flow weekly','p_regime_of':'Regime + order flow',
     'p_lasso_corr':'Auto features, sparse logistic','p_lasso_corr_wk':'Auto features, sparse logistic weekly','p_gbm_corr_top5_wk':'Auto features, trees top 5 weekly','p_gbm_dir_top10_wk':'Auto features, trees BRTI only weekly'}
M = list(N); INS = {'p_static','p_vol'}
A['band'] = (np.clip(np.floor(A.mid*100),1,99)//10*10).astype(int)
A['time'] = pd.cut(A.min_left,[0,2,5,9,14],labels=['1-2 min','3-5 min','6-9 min','10-14 min'])
months = sorted(A.month.unique())
def cell(g, m):
    h = g.dropna(subset=[m]); 
    if len(h) < 0.95*len(g) or len(h) == 0: return None
    kb = np.mean((h.p_kalshi-h.res)**2); b = np.mean((h[m]-h.res)**2)
    pb = (np.clip(np.floor(h[m]*100),0,99)//10*10)
    miss = (h.groupby(pb).apply(lambda x: len(x)*abs(x.res.mean()-x[m].mean())).sum()/len(h))*100
    return dict(n=len(h), win=100*h.res.mean(), avg=100*h[m].mean(), d=1000*(b-kb), miss=miss)
rows = []
for mo in months:
    g = A[A.month == mo]
    for m in M:
        c = cell(g, m)
        if c is None: continue
        rows.append(dict(month=mo, model=N[m], sample=('in-sample' if (m in INS and mo < '2026-09') else ('holdout' if mo >= '2026-09' else 'out-of-fold')),
                         snapshots=c['n'], actual_win_pct=round(c['win'],2), avg_prob_pct=round(c['avg'],2), brier_vs_kalshi_x1000=round(c['d'],3), calib_miss_pts=round(c['miss'],3)))
L = pd.DataFrame(rows)
W = L.pivot(index='model', columns='month', values='brier_vs_kalshi_x1000').reindex([N[m] for m in M])
W['months better than Kalshi'] = (W[months] < 0).sum(axis=1).astype(str) + ' of ' + W[months].notna().sum(axis=1).astype(str)
CM = L.pivot(index='model', columns='month', values='calib_miss_pts').reindex([N[m] for m in M])
# month x time x price: best model and how many beat Kalshi (holdout-comparable models only per month)
r2 = []
for (mo, t, b), g in A.groupby(['month','time','band'], observed=True):
    kb = np.mean((g.p_kalshi-g.res)**2); res = []
    for m in M[1:]:
        if m in INS and mo < '2026-09': continue
        h = g.dropna(subset=[m])
        if len(h) < 0.95*len(g): continue
        res.append((N[m], 1000*(np.mean((h[m]-h.res)**2)-np.mean((h.p_kalshi-h.res)**2))))
    if not res: continue
    best = min(res, key=lambda x: x[1])
    r2.append(dict(month=mo, time_left=t, kalshi_price=f'{b}-{b+9}c', snapshots=len(g), actual_win_pct=round(100*g.res.mean(),2), kalshi_avg_pct=round(100*g.p_kalshi.mean(),2),
                   models_compared=len(res), models_beating_kalshi=sum(x[1]<0 for x in res), best_model=best[0], best_vs_kalshi_x1000=round(best[1],3)))
T = pd.DataFrame(r2)
# how often each model beats Kalshi across month x time x price cells (cells with >=200 snapshots)
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(W.to_string()); print('\nCalibration miss pts by month'); print(CM.round(2).to_string())
TT = T[T.snapshots>=200]
print('\nCells (month x time x price, >=200 snaps):', len(TT), ' cells where best model beats Kalshi:', (TT.best_vs_kalshi_x1000<0).sum())
print(TT.groupby('month').apply(lambda h: pd.Series(dict(cells=len(h), any_beat=(h.models_beating_kalshi>0).sum(), majority_beat=(h.models_beating_kalshi>h.models_compared/2).sum()))).to_string())
# consistency: for each time x price cell, months where majority of models beat Kalshi
C2 = TT.assign(maj=TT.models_beating_kalshi>TT.models_compared/2).groupby(['time_left','kalshi_price'],observed=True).agg(months=('month','count'), months_majority_beat=('maj','sum')).reset_index()
print(C2[C2.months_majority_beat>=C2.months*0.75].to_string())
with pd.ExcelWriter('work/kalshi_vs_all_models_monthly.xlsx') as w:
    W.to_excel(w, sheet_name='Brier_vs_Kalshi_by_month'); CM.round(3).to_excel(w, sheet_name='Calib_miss_by_month')
    L.to_excel(w, sheet_name='Month_x_model_detail', index=False); T.to_excel(w, sheet_name='Month_x_time_x_price', index=False)
    C2.to_excel(w, sheet_name='Cell_consistency', index=False)
