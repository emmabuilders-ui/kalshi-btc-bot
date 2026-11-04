import numpy as np, pandas as pd
b=np.load('../tl/brti.npy'); T0=int(open('../tl/brti_t0.txt').read())
c=pd.read_pickle('snaps_tl.pkl')
c['lo_']=c.lo.fillna(-np.inf); c['hi_']=(c.hi+0.01).fillna(np.inf)
H=np.full(len(c),np.nan); P=H.copy(); Q=H.copy()
c=c.reset_index(drop=True)
for close,g in c.groupby('close'):
    s=close-7200-T0
    if s<0: continue
    w=b[s:close-T0]
    for (lo,hi),gg in g.groupby(['lo_','hi_']):
        ins=((w>=lo)&(w<hi)).astype(np.int32); cs=np.concatenate([[0],np.cumsum(ins)])
        tt=(gg.t.values-(close-7200))   # index of second t within window
        h0=3600
        H[gg.index]=(cs[tt]-cs[h0])/(tt-h0)
        P[gg.index]=cs[h0]/3600
        Q[gg.index]=(cs[tt]-cs[tt-900])/900
c['in_this_hour']=H; c['in_prev_hour']=P; c['in_last15']=Q
c.to_pickle('snaps_tib.pkl'); print(c[['in_this_hour','in_prev_hour','in_last15']].describe().T)
