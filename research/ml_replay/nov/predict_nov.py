# Every model's predictions for the untouched Oct 26 - Nov 9 2026 contracts, using only data closed before each refit.
# Frozen models = the Dec-Aug fits (as scored on Sep-Oct and as deployed); weekly/recal/gate versions use everything before.
import numpy as np, pandas as pd, json, pickle, sys, time, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
ML = sys.argv[1]
ts = lambda s: pd.Timestamp(s, tz='UTC').timestamp(); SEP, NOV = ts('2026-09-01'), ts('2026-10-26')
T0 = time.time(); log = lambda *a: print(round(time.time() - T0), *a, flush=True)
WEEKS = [(ts('2026-10-26'), ts('2026-11-02')), (ts('2026-11-02'), ts('2026-11-09')), (ts('2026-11-09'), ts('2026-11-10'))]
DIRECT = ['dist', 'z', 'z60', 'secs_left', 'sig1', 'sig5', 'sig15', 'sig60', 'vol_ratio', 'vol_ratio_short', 'ret_10', 'ret_30', 'ret_60', 'ret_300', 'ret_900', 'dist_hi', 'dist_lo', 'hour_utc', 'dow']
CORR = DIRECT + ['x', 'spread', 'log_cumvol', 'ask_age_c', 'bid_age_c']; CL = DIRECT + ['x', 'spread', 'log_cumvol']
mk = lambda: HGB(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=300, l2_regularization=1.0, early_stopping=False, random_state=0)
S = pd.read_parquet('work/features.parquet').sort_values('snap_ts').reset_index(drop=True)
te = S[S.snap_ts >= NOV]; tr = S[S.close_ts <= SEP]; out = te[['ticker', 'k']].copy()
for n, cols in (('p_direct', DIRECT), ('p_corr', CORR), ('p_cl', CL)): out[n] = mk().fit(tr[cols], tr.res).predict_proba(te[cols])[:, 1]
log('frozen')
for n, cols in (('p_direct_wk', DIRECT), ('p_corr_wk', CORR)):
    out[n] = np.nan
    for a, b in WEEKS:
        w = te[(te.snap_ts >= a) & (te.snap_ts < b)]; trw = S[S.close_ts <= a]
        out.loc[w.index, n] = mk().fit(trw[cols], trw.res).predict_proba(w[cols])[:, 1]
log('weekly')
# 15-min recal on trailing 30 days. Live base = weekly refits (Sep-Oct from the old run, Nov from above)
O = pd.read_parquet(f'{ML}/work/models_scored.parquet', columns=['ticker', 'k', 'p_direct_wk', 'p_corr_wk'])
R = S[S.snap_ts >= ts('2026-09-24')][['ticker', 'k', 'snap_ts', 'open_ts', 'close_ts', 'min_left', 'res', 'p_kalshi']].merge(O, on=['ticker', 'k'], how='left')
R = R.set_index(['ticker', 'k']); o2 = out.set_index(['ticker', 'k'])
R.loc[o2.index, 'p_direct_wk'] = o2.p_direct_wk; R.loc[o2.index, 'p_corr_wk'] = o2.p_corr_wk; R = R.reset_index().dropna(subset=['p_direct_wk', 'p_corr_wk'])
eps = 1e-4; lg = lambda p: np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
BASE = {'kalshi': 'p_kalshi', 'corr': 'p_corr_wk', 'direct': 'p_direct_wk'}
X = {b: np.column_stack([lg(R[c]), lg(R[c]) * R.min_left, R.min_left]) for b, c in BASE.items()}
y = R.res.to_numpy(); close = R.close_ts.to_numpy(); cal = {b: np.full(len(R), np.nan) for b in BASE}
for op, g in R[R.snap_ts >= NOV].groupby('open_ts'):
    idx = g.index.to_numpy(); trn = np.where((close <= op) & (close > op - 30 * 86400))[0]
    for b in BASE: cal[b][idx] = LogisticRegression(C=1.0, max_iter=200).fit(X[b][trn], y[trn]).predict_proba(X[b][idx])[:, 1]
for b in BASE: R[f'p_{b}_cal30d'] = cal[b]
out = out.merge(R[['ticker', 'k', 'p_kalshi_cal30d', 'p_corr_cal30d', 'p_direct_cal30d']], on=['ticker', 'k'], how='left'); log('recal')
# order flow: frozen Dec-Aug corr+OF / direct+OF, weekly Kalshi+OF
Fl = pd.read_parquet('work/flow.parquet'); SO = S[S.src == 'trades'].merge(Fl, on=['ticker', 'k'], how='inner').sort_values('snap_ts').reset_index(drop=True)
SO['of_px_vs_mid'] = SO.last_px - SO.mid
for w in (10, 30, 60, 180): SO[f'of_vwap_vs_mid_{w}'] = (SO[f'of_vwap_{w}'] - SO.mid).fillna(0)
OF = ['of_signed_open'] + [f'of_{a}_{w}' for w in (10, 30, 60, 180) for a in ('imb', 'vol', 'n')] + ['of_big_180', 'of_dpx_30', 'of_dpx_60', 'of_dpx_180', 'of_px_vs_mid'] + [f'of_vwap_vs_mid_{w}' for w in (10, 30, 60, 180)]
teo = SO[SO.snap_ts >= NOV]; tro = SO[SO.close_ts <= SEP]; oo = teo[['ticker', 'k']].copy()
oo['p_corr_of'] = mk().fit(tro[CORR + OF], tro.res).predict_proba(teo[CORR + OF])[:, 1]
oo['p_direct_of'] = mk().fit(tro[DIRECT + OF], tro.res).predict_proba(teo[DIRECT + OF])[:, 1]
KOF = ['x', 'secs_left', 'spread'] + OF; oo['p_kof_wk'] = np.nan
for a, b in WEEKS:
    w = teo[(teo.snap_ts >= a) & (teo.snap_ts < b)]; trw = SO[SO.close_ts <= a]
    oo.loc[w.index, 'p_kof_wk'] = mk().fit(trw[KOF], trw.res).predict_proba(w[KOF])[:, 1]
out = out.merge(oo, on=['ticker', 'k'], how='left'); log('flow')
# regime gates: same cells and expert choices as the Sep-Oct holdout (regime_gate*.csv), thresholds from Feb-Aug out-of-fold rows
OLD = pd.read_parquet(f'{ML}/work/models_scored.parquet', columns=['snap_ts', 'sig15']); oof = OLD[(OLD.snap_ts >= ts('2026-02-01')) & (OLD.snap_ts < SEP)]
q1, q2 = oof.sig15.quantile([1 / 3, 2 / 3])
OOF = pd.read_parquet(f'{ML}/work/models_of.parquet', columns=['snap_ts', 'sig15', 'p_kalshi', 'p_roll', 'p_corr_of', 'p_direct_of']).dropna()
o1, o2_ = OOF[(OOF.snap_ts >= ts('2026-02-01')) & (OOF.snap_ts < SEP)].sig15.quantile([1 / 3, 2 / 3])
X2 = te[['ticker', 'k', 'sig15', 'min_left', 'mid', 'p_kalshi', 'p_roll']].merge(out, on=['ticker', 'k'])
cellf = lambda d, a, b: np.where(d.sig15 < a, 0, np.where(d.sig15 < b, 1, 2)) * 9 + np.where(d.min_left <= 3, 0, np.where(d.min_left <= 8, 1, 2)) * 3 + np.where(d.mid < .2, 0, np.where(d.mid < .8, 1, 2))
g1 = pd.read_csv(f'{ML}/work/regime_gate.csv', index_col=0).expert.to_dict(); g2 = pd.read_csv(f'{ML}/work/regime_gate_of.csv', index_col=0).expert.to_dict()
X2['p_regime'] = [X2.at[i, g1.get(c, 'p_kalshi')] for i, c in zip(X2.index, cellf(X2, q1, q2))]
X2['p_regime_of'] = [X2.at[i, g2.get(c, 'p_kalshi')] if pd.notna(X2.at[i, 'p_corr_of']) else np.nan for i, c in zip(X2.index, cellf(X2, o1, o2_))]
out = X2.drop(columns=['sig15', 'min_left', 'mid', 'p_kalshi', 'p_roll']); log('regime')
# auto features: lasso frozen Dec-Aug + weekly; trees top-5 / BRTI top-10 weekly with the original ranking (Jul-Aug validation)
D = pd.read_parquet('work/autofeat.parquet').sort_values('snap_ts').reset_index(drop=True)
META = ['ticker', 'k', 'snap_ts', 'open_ts', 'close_ts', 'min_left', 'res', 'mid', 'ask', 'bid', 'day', 'month', 'period']
ALL = [c for c in D.columns if c not in META]; sel = json.load(open(f'{ML}/work/select4.json')) if __import__('os').path.exists(f'{ML}/work/select4.json') else None
gbm = lambda: HGB(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=300, l2_regularization=1.0, early_stopping=False, random_state=0)
lasso = lambda: make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(penalty='l1', C=0.05, solver='saga', max_iter=300))
rank = {f: sel[f]['rank'] for f in ('correction', 'direct')}   # same ranking as the Sep-Oct version (Jul-Aug validation, all before Oct 26)
ted = D[D.snap_ts >= NOV]; ao = ted[['ticker', 'k']].copy()
trd = D[D.close_ts <= SEP]; ao['p_lasso_corr'] = lasso().fit(trd[ALL], trd.res).predict_proba(ted[ALL])[:, 1]; log('lasso frozen')
for n, f, cols in (('p_lasso_corr_wk', lasso, ALL), ('p_gbm_corr_top5_wk', gbm, rank['correction'][:5]), ('p_gbm_dir_top10_wk', gbm, rank['direct'][:10])):
    ao[n] = np.nan
    for a, b in WEEKS:
        w = ted[(ted.snap_ts >= a) & (ted.snap_ts < b)]; trw = D[D.close_ts <= a]
        ao.loc[w.index, n] = f().fit(trw[cols], trw.res).predict_proba(w[cols])[:, 1]
    log(n)
out = out.merge(ao, on=['ticker', 'k'], how='left')
out.to_parquet('work/nov_models.parquet'); log('saved', out.shape)
