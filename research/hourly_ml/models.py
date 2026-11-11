import sys, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
name=sys.argv[1]
S=pd.read_parquet(f'work/{name}_snaps.parquet').sort_values('t').reset_index(drop=True)
S['klogit']=np.log(np.clip(S.mid,1e-3,1-1e-3)/(1-np.clip(S.mid,1e-3,1-1e-3)))
S['plogit']=np.log(np.clip(S.p_sa,1e-3,1-1e-3)/(1-np.clip(S.p_sa,1e-3,1-1e-3)))
BR=['secs_left','vol_60','vol_300','vol_900','vol_3600','dist_lo','dist_hi','z_lo','z_hi','p_sa','plogit','inside','abs_dist_edge','z_edge']+[f'ret_{l}' for l in (10,30,60,300,900,1800,3600)]+[f'{a}_{w}' for a in('from_hi','from_lo') for w in (900,3600,86400)]+['volratio_60_900','volratio_300_3600','prior_hr_vol_pct','hour','dow']
if name=='btcd': BR=[c for c in BR if c not in('dist_hi','z_hi')]
KA=['mid','klogit','spread']
M={'direct':BR,'corr':BR+KA,'lasso':BR+KA}
def hgb(): return HGB(max_iter=250,learning_rate=0.05,max_leaf_nodes=31,min_samples_leaf=300,l2_regularization=1.0,early_stopping=False,random_state=0)
def lasso(): return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(penalty='l1',C=0.05,solver='saga',max_iter=300))
def fit(kind,tr,cols):
    if kind=='lasso':
        tr=tr.sample(min(len(tr),500000),random_state=0)
        return lasso().fit(tr[cols],tr.res)
    return hgb().fit(tr[cols],tr.res)
ts=lambda s: pd.Timestamp(s,tz='UTC').timestamp()
for k in M: S['p_'+k]=np.nan
# Mar-Aug walk-forward (monthly expanding, train on markets closed before month start); direct+corr+lasso
for m in pd.date_range('2026-03-01','2026-08-01',freq='MS',tz='UTC'):
    a,b=m.timestamp(),(m+pd.offsets.MonthBegin(1)).timestamp()
    tr=S[S.close<=a]; te=S[(S.t>=a)&(S.t<b)]
    if len(tr)<50000 or te.empty: continue
    for k,c in M.items(): S.loc[te.index,'p_'+k]=fit(k,tr,c).predict_proba(te[c])[:,1]
    print('oof',m.date(),len(tr),len(te),flush=True)
# Sep-Oct frozen on Dec-Aug
cut=ts('2026-09-01'); tr=S[S.close<=cut]; te=S[S.t>=cut]
for k,c in M.items(): S.loc[te.index,'p_'+k]=fit(k,tr,c).predict_proba(te[c])[:,1]; print('frozen',k,flush=True)
S['p_kalshi']=S.mid
S.to_parquet(f'work/{name}_scored.parquet'); print('saved')
