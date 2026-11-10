import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
D = pd.read_parquet('work/autofeat.parquet'); ts = lambda s: pd.Timestamp(s, tz='UTC').timestamp()
META = ['ticker', 'k', 'snap_ts', 'open_ts', 'close_ts', 'min_left', 'res', 'mid', 'ask', 'bid', 'day', 'month', 'period']
FE = [c for c in D.columns if c not in META and c not in ('klogit', 'kmid')]
D['kl'] = np.log(np.clip(D.mid, 1e-3, 1 - 1e-3) / (1 - np.clip(D.mid, 1e-3, 1 - 1e-3)))
splits = {'JulAug': (D.close_ts <= ts('2026-07-01'), (D.snap_ts >= ts('2026-07-01')) & (D.snap_ts < ts('2026-09-01'))),
          'SepOct': (D.close_ts <= ts('2026-09-01'), D.snap_ts >= ts('2026-09-01'))}
Xb = lambda d: np.column_stack([d.kl, d.kl * d.min_left, d.min_left])
rc, rb, rbm = [], [], []
def agg(g, keys, out, name, f):
    for k, h in g.groupby(keys):
        dd = h.groupby('day').d.mean()
        out.append(dict(period=name, feature=f, **dict(zip(keys if isinstance(keys, list) else [keys], k if isinstance(k, tuple) else (k,))),
                        gain_x1000=1000 * h.d.mean(), se_x1000=1000 * dd.std() / np.sqrt(max(len(dd), 2)), snapshots=len(h)))
for name, (tri, tei) in splits.items():
    tr, te = D[tri], D[tei]; pb = LogisticRegression(C=1e4, max_iter=500).fit(Xb(tr), tr.res).predict_proba(Xb(te))[:, 1]
    cent = np.clip(np.floor(te.mid.to_numpy() * 100), 1, 99).astype(int)
    base = pd.DataFrame(dict(cent=cent, band=(cent // 10) * 10, min_left=te.min_left.to_numpy(), day=te.day.to_numpy()))
    for f in FE:
        mu, sd = tr[f].mean(), tr[f].std() or 1.0
        xf = lambda d: ((d[f] - mu) / sd).fillna(0).clip(-6, 6).to_numpy()
        X = lambda d: np.column_stack([Xb(d), xf(d), xf(d) * d.kl, xf(d) * d.min_left])
        p = LogisticRegression(C=1e4, max_iter=500).fit(X(tr), tr.res).predict_proba(X(te))[:, 1]
        g = base.assign(d=(p - te.res.to_numpy()) ** 2 - (pb - te.res.to_numpy()) ** 2)
        agg(g, 'cent', rc, name, f); agg(g, 'band', rb, name, f); agg(g, ['band', 'min_left'], rbm, name, f)
    print(name, 'done', flush=True)
def wide(rows, keys):
    R = pd.DataFrame(rows); P = R.pivot_table(index=['feature'] + keys, columns='period', values=['gain_x1000', 'se_x1000', 'snapshots']).reset_index()
    P.columns = ['feature'] + keys + [f'{a}_{b}' for a, b in P.columns[1 + len(keys):]]
    P['helps_both'] = (P.gain_x1000_JulAug < 0) & (P.gain_x1000_SepOct < 0)
    P['sig_both'] = (P.gain_x1000_JulAug + 2 * P.se_x1000_JulAug < 0) & (P.gain_x1000_SepOct + 2 * P.se_x1000_SepOct < 0)
    return P
C, B, BM = wide(rc, ['cent']), wide(rb, ['band']), wide(rbm, ['band', 'min_left'])
for nm, P in (('cent', C), ('band', B), ('band x minute', BM)):
    print(nm, 'cells', len(P), 'help both', int(P.helps_both.sum()), 'sig both', int(P.sig_both.sum()))
print(B.groupby('band').agg(help_both=('helps_both', 'sum'), sig_both=('sig_both', 'sum'), best_SepOct=('gain_x1000_SepOct', 'min')).round(3).to_string())
print('significant both, by band:'); print(B[B.sig_both].sort_values('gain_x1000_SepOct').round(3).to_string())
print('significant both, by cent:'); print(C[C.sig_both].sort_values('gain_x1000_SepOct').round(3).head(25).to_string())
print('significant both, band x minute (top 20):'); print(BM[BM.sig_both].sort_values('gain_x1000_SepOct').round(3).head(20).to_string())
with pd.ExcelWriter('work/ml_results_by_feature_and_price.xlsx') as w:
    B.round(4).to_excel(w, sheet_name='Feature_x_10c_band', index=False)
    B.pivot(index='feature', columns='band', values='gain_x1000_SepOct').round(3).to_excel(w, sheet_name='Grid_band_SepOct')
    B.pivot(index='feature', columns='band', values='gain_x1000_JulAug').round(3).to_excel(w, sheet_name='Grid_band_JulAug')
    C.round(4).to_excel(w, sheet_name='Feature_x_cent', index=False)
    BM.round(4).to_excel(w, sheet_name='Feature_x_band_x_minute', index=False)
    pd.concat([B[B.sig_both].assign(level='band'), C[C.sig_both].assign(level='cent'), BM[BM.sig_both].assign(level='band x minute')]).round(4).to_excel(w, sheet_name='Significant_both', index=False)
