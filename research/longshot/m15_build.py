import numpy as np, pandas as pd
T0 = 1764460800
W='/home/user/kalshi-btc-bot/research/ml_replay/work/'
B = pd.Series(np.load(W+'brti.npy')).ffill().to_numpy()
L = np.zeros(len(B)); L[1:] = np.nan_to_num(np.diff(np.log(B)))
c1 = np.concatenate([[0], np.cumsum(L)]); c2 = np.concatenate([[0], np.cumsum(L**2)])
def sd(end, w): a=end-w; s1=c1[end]-c1[a]; s2=c2[end]-c2[a]; return np.sqrt(np.maximum(s2/w-(s1/w)**2,0))
S = pd.read_parquet(W+'replay.parquet')
e = (S.snap_ts.to_numpy()-T0).astype(np.int64)
for w in (10,60,300,900): S[f'ret_{w}'] = np.log(B[e]/B[e-w])
S['sig1']=sd(e+1,60); S['sig5']=sd(e+1,300)
S['vol_ratio']=S.sig15/S.sig60.clip(lower=1e-9)
Teff=(S.secs_left-40).clip(lower=1)
S['z']=S.dist/(S.sig15.clip(lower=1e-7)*np.sqrt(Teff)); S['z60']=S.dist/(S.sig60.clip(lower=1e-7)*np.sqrt(Teff))
op=(S.open_ts.to_numpy()-T0).astype(np.int64)
hi=np.empty(len(S)); lo=np.empty(len(S))
for i,(a,b) in enumerate(zip(op,e)): seg=B[a:b+1]; hi[i]=seg.max(); lo[i]=seg.min()
S['hi']=hi; S['lo']=lo
S['ret_open']=np.log(S.brti/B[op])
# prior-hour high/low vs strike (breakout context)
ph=np.empty(len(S)); pl=np.empty(len(S))
for i,a in enumerate(op): seg=B[a-3600:a]; ph[i]=seg.max(); pl[i]=seg.min()
S['prev_hi']=ph; S['prev_lo']=pl
dt=pd.to_datetime(S.snap_ts,unit='s',utc=True).dt.tz_convert('US/Eastern')
S['hour_et']=dt.dt.hour; S['dow']=dt.dt.dayofweek
S['per']=np.where(S.period=='Dec-Aug','DecAug','SepOct')
S.to_parquet('m15_rows.parquet'); print(S.shape, S.groupby('per').size().to_dict())
