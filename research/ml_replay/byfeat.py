import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import kendalltau
D = pd.read_parquet('work/autofeat.parquet'); ts = lambda s: pd.Timestamp(s, tz='UTC').timestamp()
META = ['ticker', 'k', 'snap_ts', 'open_ts', 'close_ts', 'min_left', 'res', 'mid', 'ask', 'bid', 'day', 'month', 'period']
FE = [c for c in D.columns if c not in META and c not in ('klogit', 'kmid')]
D['kl'] = np.log(np.clip(D.mid, 1e-3, 1 - 1e-3) / (1 - np.clip(D.mid, 1e-3, 1 - 1e-3))); D['resid'] = D.res - D.mid
splits = {'Jul-Aug (train Dec-Jun)': (D.close_ts <= ts('2026-07-01'), (D.snap_ts >= ts('2026-07-01')) & (D.snap_ts < ts('2026-09-01'))),
          'Sep-Oct (train Dec-Aug)': (D.close_ts <= ts('2026-09-01'), D.snap_ts >= ts('2026-09-01'))}
base_cols = ['kl', 'min_left']
rng = np.random.default_rng(0); rows = []
def prep(X, mu, sd): return ((X - mu) / sd).fillna(0).clip(-6, 6)
for name, (tri, tei) in splits.items():
    tr, te = D[tri], D[tei]
    Xb = lambda d: np.column_stack([d.kl, d.kl * d.min_left, d.min_left])
    b0 = LogisticRegression(C=1e4, max_iter=500).fit(Xb(tr), tr.res); pb = b0.predict_proba(Xb(te))[:, 1]
    bb = np.mean((pb - te.res) ** 2); kb = np.mean((te.mid - te.res) ** 2)
    days = te.day.to_numpy(); ud, inv = np.unique(days, return_inverse=True)
    for f in FE:
        mu, sd = tr[f].mean(), tr[f].std() or 1.0
        xf = lambda d: prep(d[f], mu, sd).to_numpy()
        X = lambda d: np.column_stack([Xb(d), xf(d), xf(d) * d.kl, xf(d) * d.min_left])
        p = LogisticRegression(C=1e4, max_iter=500).fit(X(tr), tr.res).predict_proba(X(te))[:, 1]
        d = (p - te.res) ** 2 - (pb - te.res) ** 2
        s = np.bincount(inv, d.to_numpy()); n = np.bincount(inv)
        bs = [s[i].sum() / n[i].sum() for i in (rng.integers(0, len(ud), len(ud)) for _ in range(200))]
        v = te[f].to_numpy(); ok = np.isfinite(v)
        tau = kendalltau(v[ok][::7], te.resid.to_numpy()[ok][::7])[0]
        auc = roc_auc_score(te.res[ok], v[ok]); 
        rows.append(dict(split=name, feature=f, brier_gain_vs_calibrated_kalshi_x1000=1000 * d.mean(), ci95_lo=1000 * np.percentile(bs, 2.5), ci95_hi=1000 * np.percentile(bs, 97.5),
                         brier_vs_raw_kalshi_x1000=1000 * (np.mean((p - te.res) ** 2) - kb), auc_alone=max(auc, 1 - auc), tau_vs_kalshi_error=tau))
    print(name, 'done; calibrated-Kalshi baseline vs raw x1000:', round(1000 * (bb - kb), 3), flush=True)
R = pd.DataFrame(rows).round(4)
W = R.pivot(index='feature', columns='split', values=['brier_gain_vs_calibrated_kalshi_x1000', 'ci95_hi', 'tau_vs_kalshi_error', 'auc_alone'])
W.columns = [f'{a} | {b}' for a, b in W.columns]; W = W.reset_index()
ja, so = 'brier_gain_vs_calibrated_kalshi_x1000 | Jul-Aug (train Dec-Jun)', 'brier_gain_vs_calibrated_kalshi_x1000 | Sep-Oct (train Dec-Aug)'
W['helps_both'] = (W[ja] < 0) & (W[so] < 0)
W['significant_both'] = (W['ci95_hi | Jul-Aug (train Dec-Jun)'] < 0) & (W['ci95_hi | Sep-Oct (train Dec-Aug)'] < 0)
W = W.sort_values(so)
with pd.ExcelWriter('work/ml_results_by_feature.xlsx') as w:
    W.to_excel(w, sheet_name='By_feature', index=False); R.to_excel(w, sheet_name='Long', index=False)
print('features helping in both periods:', W.helps_both.sum(), 'significant both:', W.significant_both.sum(), 'of', len(W))
print(W[['feature', ja, so, 'ci95_hi | Sep-Oct (train Dec-Aug)', 'tau_vs_kalshi_error | Sep-Oct (train Dec-Aug)', 'helps_both', 'significant_both']].head(20).to_string())
print(W[['feature', ja, so]].tail(5).to_string())
