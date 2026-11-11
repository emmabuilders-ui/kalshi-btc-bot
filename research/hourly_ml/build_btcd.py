import numpy as np, pandas as pd, glob
from feat import features
q=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('work/btcd_q_p*.parquet'))]).drop_duplicates(['ticker','t'])
q['secs_left']=q.close-q.t; q=q[(q.secs_left>=60)&(q.secs_left<=1800)]
q=q[(q.yes_bid>0)&(q.yes_ask<=1)&(q.yes_ask>q.yes_bid)&((q.yes_ask-q.yes_bid)<=0.10+1e-9)].copy()
q['mid']=(q.yes_ask+q.yes_bid)/2; q['spread']=q.yes_ask-q.yes_bid; q['res']=(q.result=='yes').astype(int)
F=features(q.t.to_numpy(),q.close.to_numpy(),q.strike.to_numpy(),np.full(len(q),np.nan))
out=pd.concat([q[['ticker','t','close','yes_ask','yes_bid','mid','spread','res']].reset_index(drop=True),F],axis=1)
out['min_left']=(out.secs_left//60).astype(int)
out['period']=np.where(out.t<pd.Timestamp('2026-09-01',tz='UTC').timestamp(),'Dec-Aug','Sep-Oct')
out.to_parquet('work/btcd_snaps.parquet'); print(out.shape); print(out.groupby('period').agg(n=('res','size'),mk=('ticker','nunique'),yes=('res','mean')))
