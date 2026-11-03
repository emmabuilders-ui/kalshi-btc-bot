import numpy as np, pandas as pd
b=np.load('../tl/brti.npy'); T0=int(open('../tl/brti_t0.txt').read())
c=pd.read_pickle('snaps_tib.pkl'); c=c[c.in_this_hour.notna()&c.lo.notna()&c.hi.notna()].reset_index(drop=True)
A15=np.full(len(c),np.nan); B15=A15.copy(); A60=A15.copy(); B60=A15.copy()
for close,g in c.groupby('close'):
    s=close-7200-T0
    if s<0: continue
    w=b[s:close-T0]
    for (lo,hi),gg in g.groupby(['lo','hi']):
        ab=np.concatenate([[0],np.cumsum(w>hi+0.01)]); be=np.concatenate([[0],np.cumsum(w<lo)])
        tt=gg.t.values-(close-7200)
        A15[gg.index]=(ab[tt]-ab[tt-900])/900; B15[gg.index]=(be[tt]-be[tt-900])/900
        A60[gg.index]=(ab[tt]-ab[tt-3600])/3600; B60[gg.index]=(be[tt]-be[tt-3600])/3600
c['above15']=A15; c['below15']=B15; c['above60']=A60; c['below60']=B60
c.to_pickle('snaps_ab.pkl'); print(c[['above15','below15','above60','below60']].describe().T)
