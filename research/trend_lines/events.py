"""All trend line breaks over the whole BRTI series (same line rules as tl.py), for any time of day."""
import numpy as np, pandas as pd, sys
from tl import pivots, b, T0, B, TOL, MAXSPAN, LOOK
N=int(sys.argv[1]); lo,hi=pivots(N); out=[]
for side,piv in (('no',lo),('yes',hi)):
    sign=1 if side=='no' else -1; conf=piv+N
    for k in range(1,len(piv)-1):
        i2=piv[k]; p2=b[i2]; start=conf[k]+1; end=min(conf[k+1],len(b)-1)
        if start>end: continue
        best=None
        for j in range(k-1,max(k-LOOK,0)-1,-1):
            i1=piv[j]; p1=b[i1]
            if i2-i1>MAXSPAN: break
            if sign*(p2-p1)<=0: continue
            sl=(p2-p1)/(i2-i1); seg=np.arange(i1,i2+1)
            if (sign*(b[seg]-(p2+sl*(seg-i2)))).min()< -TOL: continue
            mid=piv[j+1:k]; t=2+int(np.sum(np.abs(b[mid]-(p2+sl*(mid-i2)))<=TOL))
            if best is None or t>=best[0]: best=(t,i1,sl)
        if best is None: continue
        t,i1,sl=best
        seg=np.arange(i2,end+1); dev=sign*(b[seg]-(p2+sl*(seg-i2)))
        w=np.where(dev< -B)[0]
        if not len(w): continue
        s=seg[w[0]]
        if s<start: continue
        out.append((int(s+T0),side,N,t,sl*60,(i2-i1)/60))
pd.DataFrame(out,columns=['ts','side','N','touches','slope_per_min','span_min']).to_csv(f'ev_N{N}.csv',index=False); print(N,len(out))
