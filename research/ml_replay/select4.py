import numpy as np, pandas as pd, json
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
D = pd.read_parquet('work/autofeat.parquet')
ts = lambda s: pd.Timestamp(s, tz='UTC').timestamp()
META = ['ticker', 'k', 'snap_ts', 'open_ts', 'close_ts', 'min_left', 'res', 'mid', 'ask', 'bid', 'day', 'month', 'period']
ALL = [c for c in D.columns if c not in META]
KAL = ['kmid', 'klogit', 'kspread', 'k_vs_vol'] + [c for c in ALL if c.startswith('kmid_chg')]
DIR = [c for c in ALL if c not in KAL]
tr = D[D.close_ts <= ts('2026-07-01')]; va = D[(D.snap_ts >= ts('2026-07-01')) & (D.snap_ts < ts('2026-09-01'))]
kb = np.mean((va.mid - va.res) ** 2)
gbm = lambda: HGB(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=300, l2_regularization=1.0, early_stopping=False, random_state=0)
lasso = lambda: make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(penalty='l1', C=0.05, solver='saga', max_iter=300))
res = {}
for fam, cols in (('correction', ALL), ('direct', DIR)):
    m = gbm().fit(tr[cols], tr.res); p = m.predict_proba(va[cols])[:, 1]; base = np.mean((p - va.res) ** 2)
    rng = np.random.default_rng(0); imp = {}
    for c in cols:
        v = va[cols].copy(); v[c] = rng.permutation(v[c].to_numpy()); imp[c] = np.mean((m.predict_proba(v)[:, 1] - va.res) ** 2) - base
    rank = sorted(imp, key=imp.get, reverse=True)
    sel = {}
    for k in (5, 10, 20, 40, len(cols)):
        cc = rank[:k]; pk = gbm().fit(tr[cc], tr.res).predict_proba(va[cc])[:, 1]; sel[k] = np.mean((pk - va.res) ** 2) - kb
        print(fam, 'gbm top', k, 'val Brier vs Kalshi', round(sel[k], 5), flush=True)
    pl = lasso().fit(tr[cols], tr.res).predict_proba(va[cols])[:, 1]; ls = np.mean((pl - va.res) ** 2) - kb
    print(fam, 'lasso all', round(ls, 5), flush=True)
    bestk = min(sel, key=sel.get)
    res[fam] = dict(rank=rank, imp={c: float(imp[c]) for c in rank}, sel={str(k): float(v) for k, v in sel.items()}, lasso=float(ls), bestk=int(bestk), gbm_all_val=float(base - kb))
json.dump(res, open('work/select4.json', 'w'), indent=1)
for fam in res: print(fam, 'best k', res[fam]['bestk'], 'top10', res[fam]['rank'][:10])
