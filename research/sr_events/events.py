# Level set fixed at each contract open; events at each minute snapshot (last 60 s): rejection ($5 touch, $10 reversal, no break) and first break ($5 through).
import numpy as np, pandas as pd
T0=1764460800
b=pd.Series(np.load('work/brti.npy').astype(float)).ffill().to_numpy()
nmin=len(b)//60; M=b[:nmin*60].reshape(nmin,60); H1=M.max(1); L1=M.min(1)
S=pd.read_parquet('work/replay_scored.parquet',columns=['ticker','k','snap_ts','ask','bid','open_ts','close_ts','strike','res','period','mid','min_left','brti'])
print('align',np.nanmax(np.abs(b[(S.snap_ts-T0).to_numpy()]-S.brti.to_numpy())))
def rng(a,m0,m1,f): return f(a[max(m0,0):m1]) if m1>max(m0,0) else np.nan
def levels(open_ts):
    m0=(open_ts-T0)//60; i0=open_ts-T0; p=b[i0]; L=[]
    def add(kind,v):
        if np.isfinite(v): L.append((kind,v))
    add('prev15 hi',rng(H1,m0-15,m0,np.max)); add('prev15 lo',rng(L1,m0-15,m0,np.min))
    add('60m hi',rng(H1,m0-60,m0,np.max)); add('60m lo',rng(L1,m0-60,m0,np.min))
    add('4h hi',rng(H1,m0-240,m0,np.max)); add('4h lo',rng(L1,m0-240,m0,np.min))
    d0=(m0//1440)*1440
    if d0>=1440:
        ph=rng(H1,d0-1440,d0,np.max); pl=rng(L1,d0-1440,d0,np.min); pc=M[d0-1,-1]; P=(ph+pl+pc)/3
        for k,v in [('prevday hi',ph),('prevday lo',pl),('pivot P',P),('pivot R1',2*P-pl),('pivot S1',2*P-ph),('pivot R2',P+ph-pl),('pivot S2',P-(ph-pl))]: add(k,v)
    if m0>d0+15: add('today hi',rng(H1,d0,m0,np.max)); add('today lo',rng(L1,d0,m0,np.min))
    add('day open',M[d0,0])
    for step in (500,1000):
        add(f'round {step}',np.floor(p/step)*step); add(f'round {step}',np.ceil(p/step)*step)
    out={}
    for kind,v in L:
        if abs(v-p)>10:
            key=(kind.split(' ')[0]+' '+kind.split(' ')[1] if kind.startswith('round') else kind, v>p)
            if key not in out or abs(v-p)<abs(out[key]-p): out[key]=v
    return p,[(k[0],k[1],v) for k,v in out.items()]
rows=[]
for tk,g in S.groupby('ticker',sort=False):
    o=int(g.open_ts.iloc[0]); i0=o-T0; p,L=levels(o)
    if not L: continue
    for r in g.itertuples():
        i=int(r.snap_ts-T0); w=b[i-60:i+1]; mxw=w.max(); mnw=w.min(); seg=b[i0:i+1]; mxa=seg.max(); mna=seg.min(); cur=b[i]
        for kind,isres,v in L:
            if isres:
                if mxw>=v-5 and mxa<v+5 and cur<=mxw-10: rows.append((tk,r.k,'reject','res',kind,v,'NO'))
                f=np.argmax(seg>=v+5) if mxa>=v+5 else -1
                if f>=0 and i-60<i0+f<=i: rows.append((tk,r.k,'break','res',kind,v,'YES'))
            else:
                if mnw<=v+5 and mna>v-5 and cur>=mnw+10: rows.append((tk,r.k,'reject','sup',kind,v,'YES'))
                f=np.argmax(seg<=v-5) if mna<=v-5 else -1
                if f>=0 and i-60<i0+f<=i: rows.append((tk,r.k,'break','sup',kind,v,'NO'))
E=pd.DataFrame(rows,columns=['ticker','k','ev','dir','level','lv','side'])
E.to_parquet('work/sr_events.parquet'); print(E.shape); print(E.groupby(['ev','dir']).size())
