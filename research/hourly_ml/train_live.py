import numpy as np, pandas as pd, pickle
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
S=pd.read_parquet('work/bands_snaps.parquet').sort_values('t').reset_index(drop=True)
S['klogit']=np.log(np.clip(S.mid,1e-3,1-1e-3)/(1-np.clip(S.mid,1e-3,1-1e-3))); S['plogit']=np.log(np.clip(S.p_sa,1e-3,1-1e-3)/(1-np.clip(S.p_sa,1e-3,1-1e-3)))
# live feature set: no 24-hour high/low (the bot keeps 4 hours of BRTI)
C=['secs_left','vol_60','vol_300','vol_900','vol_3600','dist_lo','dist_hi','z_lo','z_hi','p_sa','plogit','inside','abs_dist_edge','z_edge']+[f'ret_{l}' for l in (10,30,60,300,900,1800,3600)]+[f'{a}_{w}' for a in('from_hi','from_lo') for w in (900,3600)]+['volratio_60_900','volratio_300_3600','prior_hr_vol_pct','hour','dow','mid','klogit','spread']
mk=lambda:HGB(max_iter=250,learning_rate=0.05,max_leaf_nodes=31,min_samples_leaf=300,l2_regularization=1.0,early_stopping=False,random_state=0)
ts=lambda s:pd.Timestamp(s,tz='UTC').timestamp()
S['p_live']=np.nan
for m in pd.date_range('2026-03-01','2026-08-01',freq='MS',tz='UTC'):
    a,b=m.timestamp(),(m+pd.offsets.MonthBegin(1)).timestamp()
    tr=S[S.close<=a]; te=S[(S.t>=a)&(S.t<b)]
    S.loc[te.index,'p_live']=mk().fit(tr[C],tr.res).predict_proba(te[C])[:,1]; print('oof',m.date(),flush=True)
cut=ts('2026-09-01'); tr=S[S.close<=cut]; te=S[S.t>=cut]
M=mk().fit(tr[C],tr.res); S.loc[te.index,'p_live']=M.predict_proba(te[C])[:,1]
pickle.dump((M,C),open('work/band_live_model.pkl','wb')); S.to_parquet('work/bands_live_scored.parquet'); print('done',len(tr))
