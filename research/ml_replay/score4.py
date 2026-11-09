import numpy as np, pandas as pd, json
from sklearn.metrics import roc_auc_score
D = pd.read_parquet('work/autofeat.parquet'); te = D[D.snap_ts >= pd.Timestamp('2026-09-01', tz='UTC').timestamp()][['ticker','k','snap_ts','res','mid','ask','bid','day','month']]
for c in ['p_lasso_corr','p_gbm_corr_top5','p_gbm_dir_top10']: te = te.merge(pd.read_parquet(f'work/wf4_{c}.parquet'), on=['ticker','k'])
te['p_kalshi'] = te.mid
N = {'p_kalshi': 'Kalshi mid', 'p_lasso_corr': 'Sparse logistic, 111 auto features (frozen)', 'p_lasso_corr_wk': 'Sparse logistic, weekly walk-forward',
     'p_gbm_corr_top5': 'Boosted trees, top 5 incl. Kalshi (frozen)', 'p_gbm_corr_top5_wk': 'Boosted trees top 5, weekly walk-forward',
     'p_gbm_dir_top10': 'Boosted trees, top 10 BRTI only (frozen)', 'p_gbm_dir_top10_wk': 'Boosted trees top 10 BRTI, weekly walk-forward'}
g = te.dropna(subset=list(N)); kb = np.mean((g.p_kalshi - g.res) ** 2); rng = np.random.default_rng(0)
g['cent'] = np.clip(np.floor(g.mid * 100), 1, 99).astype(int)
def ci(m, n=500):
    d = ((g[m] - g.res) ** 2 - (g.p_kalshi - g.res) ** 2).groupby(g.day).agg(['sum', 'size']); s, k = d['sum'].to_numpy(), d['size'].to_numpy()
    return np.percentile([s[i].sum() / k[i].sum() for i in (rng.integers(0, len(s), len(s)) for _ in range(n))], [2.5, 97.5])
R = []
for m in N:
    b = np.mean((g[m] - g.res) ** 2); lo, hi = ci(m) if m != 'p_kalshi' else (0, 0)
    bc = g.groupby('cent').apply(lambda h: np.mean((h[m] - h.res) ** 2) - np.mean((h.p_kalshi - h.res) ** 2))
    mon = g.groupby('month').apply(lambda h: np.mean((h[m] - h.res) ** 2) - np.mean((h.p_kalshi - h.res) ** 2))
    R.append(dict(model=N[m], snapshots=len(g), brier=round(b, 5), vs_kalshi=round(b - kb, 5), ci95_lo=round(lo, 5), ci95_hi=round(hi, 5), auc=round(roc_auc_score(g.res, g[m]), 4),
                  cents_better=f"{(bc < 0).sum()}/{len(bc)}", months_better=f"{(mon < 0).sum()}/{len(mon)}"))
R = pd.DataFrame(R); print(R.to_string())
fee = lambda p: 0.07 * p * (1 - p); T = []
for m in list(N)[1:]:
    for thr in (0.05, 0.07, 0.10):
        y = g[g[m] - g.ask >= thr].assign(side='YES', price=lambda d: d.ask, win=lambda d: d.res)
        n = g[g.bid - g[m] >= thr].assign(side='NO', price=lambda d: 1 - d.bid, win=lambda d: 1 - d.res)
        t = pd.concat([y, n]).sort_values('snap_ts').drop_duplicates('ticker'); t['model'] = N[m]; t['thr_pts'] = int(thr * 100)
        t['cost'] = t.price + fee(t.price); t['pnl'] = t.win - t.cost; T.append(t)
T = pd.concat(T)
tr = T.groupby(['model', 'thr_pts']).apply(lambda h: pd.Series(dict(trades=len(h), win_pct=100 * h.win.mean(), avg_price_c=100 * h.price.mean(), breakeven_pct=100 * h.cost.mean(),
     edge_after_fees_c=100 * h.pnl.mean(), se_c=100 * h.pnl.std() / np.sqrt(max(len(h), 1)), months_positive=f"{(h.groupby('month').pnl.mean() > 0).sum()}/{h.month.nunique()}"))).reset_index().round(2)
print(tr.to_string())
sel = json.load(open('work/select4.json'))
V = pd.DataFrame([dict(family=f, candidate=f'gbm top {k}', val_brier_vs_kalshi=v) for f in sel for k, v in sel[f]['sel'].items()] + [dict(family=f, candidate='sparse logistic, all', val_brier_vs_kalshi=sel[f]['lasso']) for f in sel])
I = pd.DataFrame([dict(family=f, feature=c, val_importance=v) for f in sel for c, v in list(sel[f]['imp'].items())[:30]])
with pd.ExcelWriter('work/ml_step4_autofeatures.xlsx') as w:
    R.to_excel(w, sheet_name='SepOct_Brier', index=False); tr.to_excel(w, sheet_name='SepOct_trades', index=False)
    V.round(5).to_excel(w, sheet_name='JulAug_validation', index=False); I.to_excel(w, sheet_name='Top_features_JulAug', index=False)
    pd.read_csv('work/lasso_coef.csv').to_excel(w, sheet_name='Logistic_coefficients', index=False)
# price x time-to-settlement breakdown (Michael 2026-11-09): every cent x every minute left, plus 10c x minute summary
g['min_left'] = 15 - g.k
def grid(m, by):
    return g.groupby(by).apply(lambda h: pd.Series(dict(snapshots=len(h), contracts=h.ticker.nunique(), win_pct=100 * h.res.mean(), kalshi_price_c=100 * h.mid.mean(),
        win_minus_price_pts=100 * (h.res.mean() - h.mid.mean()), model_prob_pct=100 * h[m].mean(),
        brier_vs_kalshi_x1000=1000 * (np.mean((h[m] - h.res) ** 2) - np.mean((h.p_kalshi - h.res) ** 2))))).reset_index().round(3)
g['band10'] = (g.cent // 10) * 10
summ = {}
with pd.ExcelWriter('work/ml_step4_autofeatures.xlsx', mode='a', engine='openpyxl') as w:
    for m in list(N)[1:]:
        tag = m.replace('p_', '')[:24]
        grid(m, ['cent', 'min_left']).to_excel(w, sheet_name=f'cent_min_{tag}'[:31], index=False)
        b = grid(m, ['band10', 'min_left']); b.to_excel(w, sheet_name=f'band_min_{tag}'[:31], index=False)
        pv = b.pivot(index='band10', columns='min_left', values='brier_vs_kalshi_x1000'); summ[N[m]] = pv
        cnt = (pv < 0).sum().sum(); print(N[m], 'band x minute cells better than Kalshi:', cnt, 'of', pv.size)
for k, pv in summ.items():
    if 'Sparse' in k: print(k); print(pv.round(2).to_string())
