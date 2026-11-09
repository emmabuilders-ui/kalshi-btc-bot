import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
D = pd.read_parquet('work/autofeat.parquet'); ts = lambda s: pd.Timestamp(s, tz='UTC').timestamp()
D['kl'] = np.log(np.clip(D.mid, 1e-3, 1 - 1e-3) / (1 - np.clip(D.mid, 1e-3, 1 - 1e-3)))
splits = {'Jul-Aug': (D.close_ts <= ts('2026-07-01'), (D.snap_ts >= ts('2026-07-01')) & (D.snap_ts < ts('2026-09-01'))),
          'Sep-Oct': (D.close_ts <= ts('2026-09-01'), D.snap_ts >= ts('2026-09-01'))}
THREE = [('vol_300', 12, '5-min volatility'), ('vol_600', 12, '10-min volatility'), ('vol_900', 9, '15-min volatility')]
Xb = lambda d: np.column_stack([d.kl, d.kl * d.min_left, d.min_left])
fee = lambda p: 0.07 * p * (1 - p)
rows = []
for name, (tri, tei) in splits.items():
    tr, te = D[tri], D[tei]; pb = LogisticRegression(C=1e4, max_iter=500).fit(Xb(tr), tr.res).predict_proba(Xb(te))[:, 1]
    for f, m, lab in THREE:
        mu, sd = tr[f].mean(), tr[f].std()
        xf = lambda d: ((d[f] - mu) / sd).fillna(0).clip(-6, 6).to_numpy()
        X = lambda d: np.column_stack([Xb(d), xf(d), xf(d) * d.kl, xf(d) * d.min_left])
        p = LogisticRegression(C=1e4, max_iter=500).fit(X(tr), tr.res).predict_proba(X(te))[:, 1]
        sel = (te.min_left == m).to_numpy(); h = te[sel].copy(); h['p'] = p[sel]; h['pb'] = pb[sel]
        h['band'] = (np.clip(np.floor(h.mid * 100), 1, 99) // 10 * 10).astype(int)
        h['d'] = (h.p - h.res) ** 2 - (h.pb - h.res) ** 2
        # model vs Kalshi direction: does the feature push the probability up or down vs calibrated Kalshi?
        h['shift_pts'] = 100 * (h.p - h.pb)
        for b, g in h.groupby('band'):
            rows.append(dict(feature=lab, min_left=m, period=name, kalshi_price=f'{b}-{b+9}c', snapshots=len(g), win_pct=round(100 * g.res.mean(), 1),
                             avg_price_c=round(100 * g.mid.mean(), 1), win_minus_price_pts=round(100 * (g.res.mean() - g.mid.mean()), 2),
                             brier_gain_x1000=round(1000 * g.d.mean(), 3), avg_model_shift_pts=round(g.shift_pts.mean(), 2), avg_abs_shift_pts=round(g.shift_pts.abs().mean(), 2)))
R = pd.DataFrame(rows)
W = R.pivot_table(index=['feature', 'min_left', 'kalshi_price'], columns='period', values=['snapshots', 'win_minus_price_pts', 'brier_gain_x1000', 'avg_abs_shift_pts'], aggfunc='first')
W.columns = [f'{a} {b}' for a, b in W.columns]; W = W.reset_index()
print(W[['feature', 'kalshi_price', 'snapshots Jul-Aug', 'snapshots Sep-Oct', 'brier_gain_x1000 Jul-Aug', 'brier_gain_x1000 Sep-Oct', 'avg_abs_shift_pts Sep-Oct']].to_string())
with pd.ExcelWriter('work/three_vol_features_by_price.xlsx') as w:
    W.to_excel(w, sheet_name='By_10c_band', index=False); R.to_excel(w, sheet_name='Long', index=False)
