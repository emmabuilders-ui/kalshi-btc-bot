# Long table for the pivot: month x minute-left x 10c Kalshi band x model.
import numpy as np, pandas as pd
exec(open('monthly.py').read().split("months = sorted")[0])
rows = []
for (mo, ml, b), g in A.groupby(['month', 'min_left', 'band'], observed=True):
    kse = (g.p_kalshi - g.res) ** 2
    tg = '1-2 min' if ml <= 2 else '3-5 min' if ml <= 5 else '6-9 min' if ml <= 9 else '10-14 min'
    for m in M:
        h = g[g[m].notna()]
        if len(h) < 0.95 * len(g) or len(h) == 0: continue
        se = (h[m] - h.res) ** 2
        rows.append({'Month': mo, 'Period': 'Sep-Oct holdout' if mo >= '2026-09' else 'Mar-Aug',
                     'Sample': 'in-sample' if (m in INS and mo < '2026-09') else ('holdout' if mo >= '2026-09' else 'out-of-fold'),
                     'Minutes left': int(ml), 'Time group': tg, 'Kalshi price band': f'{b:02d}-{b+9}c', 'Model': N[m],
                     'Snapshots': len(h), 'Contracts': h.ticker.nunique(), 'Wins': int(h.res.sum()),
                     'Actual win %': round(100 * h.res.mean(), 2), 'Avg prob %': round(100 * h[m].mean(), 2),
                     '% wrong (win - prob, pts)': round(100 * (h.res.mean() - h[m].mean()), 2),
                     'Abs % wrong': round(100 * abs(h.res.mean() - h[m].mean()), 2),
                     'Kalshi avg price %': round(100 * h.p_kalshi.mean(), 2),
                     'Brier': round(se.mean(), 5), 'Brier vs Kalshi x1000': round(1000 * (se.mean() - kse[h.index].mean()), 3),
                     'Beats Kalshi (1=yes)': int(se.mean() < kse[h.index].mean()) if m != 'p_kalshi' else 0})
D = pd.DataFrame(rows); D.to_parquet('work/pivot_long.parquet'); print(D.shape); print(D.groupby('Model').size())
