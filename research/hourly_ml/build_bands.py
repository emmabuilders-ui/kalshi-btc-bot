import numpy as np, pandas as pd
from feat import features
d=pd.read_parquet('/mnt/project-files/data-download/kalshi-hourly-range/bands_dec2025_oct2026.parquet')
q=pd.read_parquet('/mnt/project-files/data-download/kalshi-hourly-range/quotes_1min_last60_dec2025_oct2026.parquet')
d=d[d.strike_type=='between'].copy()
q=q.merge(d[['ticker','close','floor_strike','cap_strike','result']],on='ticker')
q['secs_left']=q.close-q.t; q=q[(q.secs_left>=60)&(q.secs_left<=1800)]
q=q[(q.yes_bid>0)&(q.yes_ask<=1)&(q.yes_ask>q.yes_bid)&((q.yes_ask-q.yes_bid)<=0.10+1e-9)].copy()
q['mid']=(q.yes_ask+q.yes_bid)/2; q['spread']=q.yes_ask-q.yes_bid
q['res']=(q.result=='yes').astype(int)
F=features(q.t.to_numpy(),q.close.to_numpy(),q.floor_strike.to_numpy(),q.cap_strike.to_numpy())
out=pd.concat([q[['ticker','t','close','secs_left','yes_ask','yes_bid','mid','spread','res','floor_strike','cap_strike']].reset_index(drop=True).rename(columns={'secs_left':'s0'}),F],axis=1)
out['min_left']=(out.secs_left//60).astype(int)
out['period']=np.where(out.t<pd.Timestamp('2026-09-01',tz='UTC').timestamp(),'Dec-Aug','Sep-Oct')
out.to_parquet('work/bands_snaps.parquet'); print(out.shape); print(out.groupby('period').agg(n=('res','size'),mk=('ticker','nunique'),yes=('res','mean')))
print(out.groupby('min_left').size().head(3))
