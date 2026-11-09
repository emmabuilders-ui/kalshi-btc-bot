import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
S = pd.read_parquet('work/recal15.parquet')
cols = ['p_kalshi', 'corr_live', 'direct_live'] + [f'p_{b}_cal{w}' for b in ('kalshi', 'corr', 'direct') for w in ('1d', '7d', '30d')]
lab = {'p_kalshi': 'Kalshi mid (raw)', 'corr_live': 'Correction, weekly refit (no 15-min recal)', 'direct_live': 'Direct, weekly refit (no 15-min recal)'}
for b, nm in (('kalshi', 'Kalshi'), ('corr', 'Correction'), ('direct', 'Direct')):
    for w in ('1d', '7d', '30d'): lab[f'p_{b}_cal{w}'] = f'{nm} + recal every 15 min, trailing {w}'
g = S.dropna(subset=cols); kb = np.mean((g.p_kalshi - g.res) ** 2); rng = np.random.default_rng(0)
def ci(m, n=500):
    d = ((g[m] - g.res) ** 2 - (g.p_kalshi - g.res) ** 2).groupby(g.day).agg(['sum', 'size']); s, k = d['sum'].to_numpy(), d['size'].to_numpy()
    return np.percentile([s[i].sum() / k[i].sum() for i in (rng.integers(0, len(s), len(s)) for _ in range(n))], [2.5, 97.5])
R = []
for m in cols:
    b = np.mean((g[m] - g.res) ** 2); lo, hi = ci(m) if m != 'p_kalshi' else (0, 0)
    mon = g.groupby('month').apply(lambda h: np.mean((h[m] - h.res) ** 2) - np.mean((h.p_kalshi - h.res) ** 2))
    R.append(dict(model=lab[m], snapshots=len(g), brier=round(b, 5), vs_kalshi=round(b - kb, 5), ci95_lo=round(lo, 5), ci95_hi=round(hi, 5), auc=round(roc_auc_score(g.res, g[m]), 4),
                  months_better_than_kalshi=f"{(mon < 0).sum()}/{len(mon)}"))
R = pd.DataFrame(R); print(R.to_string())
fee = lambda p: 0.07 * p * (1 - p); T = []
for m in cols[1:]:
    for thr in (0.05, 0.07, 0.10):
        y = g[g[m] - g.ask >= thr].assign(side='YES', price=lambda d: d.ask, win=lambda d: d.res)
        n = g[g.bid - g[m] >= thr].assign(side='NO', price=lambda d: 1 - d.bid, win=lambda d: 1 - d.res)
        t = pd.concat([y, n]).sort_values('snap_ts').drop_duplicates('ticker'); t['model'] = lab[m]; t['thr_pts'] = int(thr * 100)
        t['cost'] = t.price + fee(t.price); t['pnl'] = t.win - t.cost; T.append(t)
T = pd.concat(T)
tr = T.groupby(['model', 'thr_pts']).apply(lambda h: pd.Series(dict(trades=len(h), win_pct=100 * h.win.mean(), avg_price_c=100 * h.price.mean(), breakeven_pct=100 * h.cost.mean(),
     edge_after_fees_c=100 * h.pnl.mean(), se_c=100 * h.pnl.std() / np.sqrt(max(len(h), 1)), months_positive=f"{(h.groupby('month').pnl.mean() > 0).sum()}/{h.month.nunique()}"))).reset_index().round(2)
print(tr.to_string())
with pd.ExcelWriter('work/ml_recal_every_15min.xlsx') as w:
    R.to_excel(w, sheet_name='Brier_SepOct', index=False); tr.to_excel(w, sheet_name='Trades_SepOct', index=False)
