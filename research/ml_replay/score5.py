import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
S = pd.read_parquet('work/models_scored.parquet')
ts = lambda s: pd.Timestamp(s, tz='UTC').timestamp()
S['period'] = np.where(S.snap_ts >= ts('2026-09-01'), 'Sep-Oct holdout', np.where(S.snap_ts >= ts('2026-03-01'), 'Mar-Aug walk-forward', 'early'))
N = {'p_kalshi': 'Kalshi mid', 'p_roll': 'Rolling recalibration', 'p_direct': 'Direct outcome model', 'p_corr': 'Correction model',
     'p_regime': 'Regime-switching', 'p_direct_wk': 'Direct, weekly refit', 'p_corr_wk': 'Correction, weekly refit'}
rng = np.random.default_rng(0)
def ci(g, m, n=500):
    d = ((g[m] - g.res) ** 2 - (g.p_kalshi - g.res) ** 2).groupby(g.day).agg(['sum', 'size']); s, k = d['sum'].to_numpy(), d['size'].to_numpy()
    b = [s[i].sum() / k[i].sum() for i in (rng.integers(0, len(s), len(s)) for _ in range(n))]; return np.percentile(b, [2.5, 97.5])
rows = []
for per in ['Mar-Aug walk-forward', 'Sep-Oct holdout']:
    ms = [m for m in N if per.startswith('Sep') or not m.endswith('_wk')]
    g = S[(S.period == per)].dropna(subset=ms)
    kb = np.mean((g.p_kalshi - g.res) ** 2)
    for m in ms:
        b = np.mean((g[m] - g.res) ** 2); lo, hi = ci(g, m) if m != 'p_kalshi' else (0, 0)
        rows.append(dict(period=per, model=N[m], snapshots=len(g), contracts=g.ticker.nunique(), brier=round(b, 5), vs_kalshi=round(b - kb, 5),
                         ci95_lo=round(lo, 5), ci95_hi=round(hi, 5), auc=round(roc_auc_score(g.res, g[m]), 4)))
R = pd.DataFrame(rows); print(R.to_string())
# Brier gain vs Kalshi by cell (price x minutes left), Sep-Oct, correction & direct
g = S[S.period == 'Sep-Oct holdout'].dropna(subset=list(N)).copy()
g['price'] = pd.cut(g.mid, [0, .2, .4, .6, .8, 1.0001], labels=['0-19c', '20-39c', '40-59c', '60-79c', '80-99c'], right=False)
g['mins'] = pd.cut(g.min_left, [0, 3, 8, 15], labels=['1-3', '4-8', '9-14'])
cellt = g.groupby(['price', 'mins'], observed=True).apply(lambda h: pd.Series({'snapshots': len(h), **{N[m] + ' vs Kalshi (x1000)': 1000 * (np.mean((h[m] - h.res) ** 2) - np.mean((h.p_kalshi - h.res) ** 2)) for m in ['p_corr', 'p_direct', 'p_regime', 'p_corr_wk']}})).round(3).reset_index()
print(cellt.to_string())
fee = lambda p: 0.07 * p * (1 - p)
T = []
for m in list(N)[1:]:
    for thr in (0.05, 0.07, 0.10):
        h = S[S.period.isin(['Mar-Aug walk-forward', 'Sep-Oct holdout']) & S[m].notna()]
        y = h[h[m] - h.ask >= thr].assign(side='YES', price=lambda d: d.ask, win=lambda d: d.res)
        n = h[h.bid - h[m] >= thr].assign(side='NO', price=lambda d: 1 - d.bid, win=lambda d: 1 - d.res)
        t = pd.concat([y, n]).sort_values('snap_ts').drop_duplicates('ticker'); t['model'] = N[m]; t['thr_pts'] = int(thr * 100)
        t['cost'] = t.price + fee(t.price); t['pnl'] = t.win - t.cost; T.append(t[['model', 'thr_pts', 'side', 'ticker', 'period', 'month', 'min_left', 'price', 'cost', 'win', 'pnl']])
T = pd.concat(T)
def summ(g):
    mo = g.groupby('month').pnl.mean()
    return pd.Series(dict(trades=len(g), win_pct=100 * g.win.mean(), avg_price_c=100 * g.price.mean(), breakeven_pct=100 * g.cost.mean(),
                          edge_after_fees_c=100 * g.pnl.mean(), se_c=100 * g.pnl.std() / np.sqrt(max(len(g), 1)), months_positive=f"{(mo > 0).sum()}/{len(mo)}"))
tr = T.groupby(['model', 'thr_pts', 'period']).apply(summ).reset_index().round(2); print(tr.to_string())
with pd.ExcelWriter('work/ml_step5_models.xlsx') as w:
    R.to_excel(w, sheet_name='Brier_AUC', index=False); cellt.to_excel(w, sheet_name='SepOct_by_price_minutes', index=False)
    tr.to_excel(w, sheet_name='Trades_summary', index=False)
    T.groupby(['model', 'thr_pts', 'period', 'side']).apply(summ).reset_index().round(2).to_excel(w, sheet_name='Trades_by_side', index=False)
    pd.read_csv('work/regime_gate.csv').to_excel(w, sheet_name='Regime_gate_holdout', index=False)
