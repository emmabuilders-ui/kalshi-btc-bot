"""Diagonal trend line breaks near the close of KXBTC15M sessions (no lookahead).

Swing low at i: b[i] is the min of b[i-N..i+N]; known (confirmed) at i+N. Swing highs mirror.
Support line = last two confirmed swing lows, second higher than the first (rising).
Resistance line = last two confirmed swing highs, second lower (falling).
Line must hold between anchors (no print more than TOL beyond it) and must not have been broken
($B beyond) before. Signal = first print $B beyond the line at second s; entry at next 5-s mark.
Breaking rising support -> buy NO. Breaking falling resistance -> buy YES.
"""
import numpy as np, pandas as pd, sys
b=np.load('brti.npy'); T0=int(open('brti_t0.txt').read())
B=5.0; TOL=15.0; MAXSPAN=4*3600; LOOK=10
def pivots(N):
    s=pd.Series(b)
    mn=s.rolling(2*N+1,center=True).min().values
    mx=s.rolling(2*N+1,center=True).max().values
    lo=np.where(b==mn)[0]; hi=np.where(b==mx)[0]
    # collapse plateaus: keep first of consecutive indices
    lo=lo[np.r_[True,np.diff(lo)>N]]; hi=hi[np.r_[True,np.diff(hi)>N]]
    return lo,hi
def scan(N,piv,side,sessions):
    sign=1 if side=='no' else -1   # no: support below price, break = price < line - B
    conf=piv+N
    out=[]
    for o in sessions:
        oi=o-T0
        lo_s,hi_s=oi+660,oi+870   # breaks with 240..30 s left
        # candidate pairs active at any second in window
        k0=np.searchsorted(conf,lo_s,side='right')-1
        k1=np.searchsorted(conf,hi_s,side='right')-1
        done=False
        for k in range(max(k0,1),k1+1):
            i2=piv[k]; p2=b[i2]
            start=max(conf[k]+1,lo_s); end=hi_s if k==k1 else min(conf[k+1],hi_s)
            if start>end or lo_s-i2>MAXSPAN: continue
            best=None
            for j in range(k-1,max(k-LOOK,0)-1,-1):
                i1=piv[j]; p1=b[i1]
                if i2-i1>MAXSPAN: break
                if sign*(p2-p1)<=0: continue
                sl=(p2-p1)/(i2-i1)
                seg=np.arange(i1,i2+1)
                if (sign*(b[seg]-(p2+sl*(seg-i2)))).min()< -TOL: continue
                mid=piv[j+1:k]
                t=2+int(np.sum(np.abs(b[mid]-(p2+sl*(mid-i2)))<=TOL))
                if best is None or t>best[0] or (t==best[0]): best=(t,i1,p1,sl)   # later j wins ties -> keep longest
            if best is None: continue
            touches,i1,p1,sl=best
            seg=np.arange(i2,end+1); dev=sign*(b[seg]-(p2+sl*(seg-i2)))
            brk=np.where(dev< -B)[0]
            if len(brk)==0: continue
            s=seg[brk[0]]
            if s<start: continue
            out.append(dict(open_unix=o,side=side,N=N,brk_t=int(s-oi),i1=int(i1+T0),i2=int(i2+T0),
                            slope_per_min=sl*60,span_min=(i2-i1)/60,age_min=(s-i2)/60,touches=touches,
                            line=p2+sl*(s-i2),px=b[s]))
            done=True; break
    return out
if __name__=='__main__':
    N=int(sys.argv[1])
    sessions=np.arange(T0+2*86400 - (T0%900), T0+len(b)-900, 900)
    lo,hi=pivots(N)
    r=scan(N,lo,'no',sessions)+scan(N,hi,'yes',sessions)
    pd.DataFrame(r).to_csv(f'sig_N{N}.csv',index=False); print(N,len(r))
