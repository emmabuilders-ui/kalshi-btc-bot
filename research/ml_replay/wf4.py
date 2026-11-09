import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
D = pd.read_parquet('work/autofeat.parquet'); sel = json.load(open('work/select4.json'))
ts = lambda s: pd.Timestamp(s, tz='UTC').timestamp(); cut = ts('2026-09-01')
META = ['ticker', 'k', 'snap_ts', 'open_ts', 'close_ts', 'min_left', 'res', 'mid', 'ask', 'bid', 'day', 'month', 'period']
ALL = [c for c in D.columns if c not in META]
gbm = lambda: HGB(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=300, l2_regularization=1.0, early_stopping=False, random_state=0)
lasso = lambda: make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(penalty='l1', C=0.05, solver='saga', max_iter=300))
CAND = {'p_lasso_corr': (lasso, ALL), 'p_gbm_corr_top5': (gbm, sel['correction']['rank'][:5]), 'p_gbm_dir_top10': (gbm, sel['direct']['rank'][:10])}
import sys
ONLY = sys.argv[1]
te = D[D.snap_ts >= cut].copy(); tr = D[D.close_ts <= cut]
for n, (f, cols) in [(ONLY, CAND[ONLY])]:
    te[n] = f().fit(tr[cols], tr.res).predict_proba(te[cols])[:, 1]; te[n + '_wk'] = np.nan; print('frozen', n, flush=True)
    for w in pd.date_range('2026-09-01', '2026-10-26', freq='7D', tz='UTC'):
        idx = te[(te.snap_ts >= w.timestamp()) & (te.snap_ts < (w + pd.Timedelta(days=7)).timestamp())].index
        if len(idx) == 0: continue
        trw = D[D.close_ts <= w.timestamp()]
        te.loc[idx, n + '_wk'] = f().fit(trw[cols], trw.res).predict_proba(te.loc[idx, cols])[:, 1]
    print('weekly', n, flush=True)
if ONLY == 'p_lasso_corr':
  m = lasso().fit(tr[ALL], tr.res); coef = pd.Series(m[-1].coef_[0], index=ALL); coef = coef[coef != 0].sort_values(key=abs, ascending=False)
  coef.to_csv('work/lasso_coef.csv'); print('lasso nonzero', len(coef)); print(coef.head(15))
te[['ticker','k',ONLY,ONLY+'_wk']].to_parquet(f'work/wf4_{ONLY}.parquet')
