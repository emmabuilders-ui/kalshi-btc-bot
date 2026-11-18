import numpy as np, pandas as pd, json, sys
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
sys.path.insert(0, '/home/user/kalshi-btc-bot')
from kalshi_btc_bot.model_gap_paper import DirectModel
S = pd.read_parquet('work/features.parquet').sort_values('snap_ts').reset_index(drop=True)
D = ['dist','z','z60','secs_left','sig1','sig5','sig15','sig60','vol_ratio','vol_ratio_short','ret_10','ret_30','ret_60','ret_300','ret_900','dist_hi','dist_lo','hour_utc','dow']
mk = lambda: HGB(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=300, l2_regularization=1.0, early_stopping=False, random_state=0)
ts = lambda s: pd.Timestamp(s, tz='UTC').timestamp(); cut = ts('2026-09-01')
S['p'] = np.nan
for m in pd.date_range('2026-03-01', '2026-08-01', freq='MS', tz='UTC'):
    a, b = m.timestamp(), (m + pd.offsets.MonthBegin(1)).timestamp()
    tr = S[S.close_ts <= a]; te = S[(S.snap_ts >= a) & (S.snap_ts < b)]
    S.loc[te.index, 'p'] = mk().fit(tr[D], tr.res).predict_proba(te[D])[:, 1]; print(m.date(), flush=True)
dm = DirectModel(json.load(open('/home/user/kalshi-btc-bot/kalshi_btc_bot/data/model_gap_direct.json')))
m = S.snap_ts >= cut
S.loc[m, 'p'] = [dm.predict(r) for r in S.loc[m, dm.features].to_dict('records')]
S.to_parquet('work/unleashed_pred_direct.parquet'); print('done', S.p.notna().sum())
