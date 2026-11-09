import numpy as np, pandas as pd
S = pd.read_parquet('work/recal15.parquet')
M = {'p_kalshi_cal30d': 'Kalshi recal 30d', 'p_kalshi_cal7d': 'Kalshi recal 7d', 'p_kalshi_cal1d': 'Kalshi recal 1d',
     'p_corr_cal30d': 'Correction recal 30d', 'corr_live': 'Correction weekly', 'p_direct_cal30d': 'Direct recal 30d'}
g = S.dropna(subset=list(M)).copy(); g['cent'] = np.clip(np.round(g.mid * 100), 1, 99).astype(int)
rows = []
for c, h in g.groupby('cent'):
    kb = np.mean((h.p_kalshi - h.res) ** 2)
    r = dict(cent=c, snapshots=len(h), contracts=h.ticker.nunique(), win_pct=round(100 * h.res.mean(), 2), win_minus_price_pts=round(100 * (h.res.mean() - h.mid.mean()), 2), kalshi_brier=round(kb, 4))
    for m, nm in M.items():
        r[nm + ' vs Kalshi (x1000)'] = round(1000 * (np.mean((h[m] - h.res) ** 2) - kb), 3)
        r[nm + ' avg prob'] = round(100 * h[m].mean(), 2)
    rows.append(r)
C = pd.DataFrame(rows)
cols = [c for c in C.columns if 'vs Kalshi' in c]
print(C[['cent', 'snapshots', 'win_pct', 'win_minus_price_pts'] + cols].to_string())
print('cents where model beats Kalshi (count of 99):', {c: int((C[c] < 0).sum()) for c in cols})
C.to_excel('work/recal15_by_cent.xlsx', index=False)
