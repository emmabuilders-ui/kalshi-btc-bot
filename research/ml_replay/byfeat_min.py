import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
D = pd.read_parquet('work/autofeat.parquet'); ts = lambda s: pd.Timestamp(s, tz='UTC').timestamp()
META = ['ticker', 'k', 'snap_ts', 'open_ts', 'close_ts', 'min_left', 'res', 'mid', 'ask', 'bid', 'day', 'month', 'period']
FE = [c for c in D.columns if c not in META and c not in ('klogit', 'kmid')]
D['kl'] = np.log(np.clip(D.mid, 1e-3, 1 - 1e-3) / (1 - np.clip(D.mid, 1e-3, 1 - 1e-3)))
splits = {'JulAug': (D.close_ts <= ts('2026-07-01'), (D.snap_ts >= ts('2026-07-01')) & (D.snap_ts < ts('2026-09-01'))),
          'SepOct': (D.close_ts <= ts('2026-09-01'), D.snap_ts >= ts('2026-09-01'))}
Xb = lambda d: np.column_stack([d.kl, d.kl * d.min_left, d.min_left])
rows = []
for name, (tri, tei) in splits.items():
    tr, te = D[tri], D[tei]; pb = LogisticRegression(C=1e4, max_iter=500).fit(Xb(tr), tr.res).predict_proba(Xb(te))[:, 1]
    ml = te.min_left.to_numpy(); day = te.day.to_numpy()
    for f in FE:
        mu, sd = tr[f].mean(), tr[f].std() or 1.0
        xf = lambda d: ((d[f] - mu) / sd).fillna(0).clip(-6, 6).to_numpy()
        X = lambda d: np.column_stack([Xb(d), xf(d), xf(d) * d.kl, xf(d) * d.min_left])
        p = LogisticRegression(C=1e4, max_iter=500).fit(X(tr), tr.res).predict_proba(X(te))[:, 1]
        d = (p - te.res.to_numpy()) ** 2 - (pb - te.res.to_numpy()) ** 2
        g = pd.DataFrame(dict(m=ml, day=day, d=d))
        for m, h in g.groupby('m'):
            dd = h.groupby('day').d.mean()
            rows.append(dict(period=name, feature=f, min_left=m, gain_x1000=1000 * h.d.mean(), se_x1000=1000 * dd.std() / np.sqrt(len(dd)), snapshots=len(h)))
    print(name, 'done', flush=True)
R = pd.DataFrame(rows)
P = R.pivot_table(index=['feature', 'min_left'], columns='period', values=['gain_x1000', 'se_x1000']).reset_index()
P.columns = ['feature', 'min_left'] + [f'{a}_{b}' for a, b in P.columns[2:]]
P['helps_both'] = (P.gain_x1000_JulAug < 0) & (P.gain_x1000_SepOct < 0)
P['sig_both'] = (P.gain_x1000_JulAug + 2 * P.se_x1000_JulAug < 0) & (P.gain_x1000_SepOct + 2 * P.se_x1000_SepOct < 0)
print('feature x minute cells:', len(P), 'help both:', int(P.helps_both.sum()), 'significant both (2 SE):', int(P.sig_both.sum()))
print('by minute left: cells helping both / significant both')
print(P.groupby('min_left').agg(help_both=('helps_both', 'sum'), sig_both=('sig_both', 'sum'), best_SepOct=('gain_x1000_SepOct', 'min')).to_string())
print(P[P.sig_both].sort_values('gain_x1000_SepOct').round(3).to_string())
W1 = P.pivot(index='feature', columns='min_left', values='gain_x1000_JulAug').round(3)
W2 = P.pivot(index='feature', columns='min_left', values='gain_x1000_SepOct').round(3)
with pd.ExcelWriter('work/ml_results_by_feature_and_minute.xlsx') as w:
    P.round(4).to_excel(w, sheet_name='Feature_x_minute', index=False)
    W1.to_excel(w, sheet_name='Grid_JulAug'); W2.to_excel(w, sheet_name='Grid_SepOct')
    P[P.sig_both].round(4).to_excel(w, sheet_name='Significant_both', index=False)
