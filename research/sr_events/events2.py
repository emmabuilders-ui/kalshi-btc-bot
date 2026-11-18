# v2: adds zigzag hard levels (late_breakdown rule: $75 zigzag, 24h pivots, $40 clusters, >=2 touches) and signal strength; runs on minute marks AND on the final-minute (45/30/15 s) snapshots.
import sys, numpy as np, pandas as pd, time
sys.path.insert(0,'/home/user/kalshi-btc-bot')
from kalshi_btc_bot.late_breakdown import zigzag, ZZ, CLUSTER, MIN_TOUCH
T0=1764460800
b=pd.Series(np.load('work/brti.npy').astype(float)).ffill().to_numpy()
nmin=len(b)//60; M=b[:nmin*60].reshape(nmin,60); H1=M.max(1); L1=M.min(1)
t=time.time(); pt,pp,cf=zigzag(b,ZZ); print('zigzag',len(pt),time.time()-t,flush=True)
def hard(o):
    m=(cf<o)&(pt>=o-86400); v=np.sort(pp[m])
    if len(v)==0: return []
    gs=[[v[0]]]
    for z in v[1:]:
        if z-gs[-1][0]<=CLUSTER: gs[-1].append(z)
        else: gs.append([z])
    return [float(np.mean(g)) for g in gs if len(g)>=MIN_TOUCH]
def rng(a,m0,m1,f): return f(a[max(m0,0):m1]) if m1>max(m0,0) else np.nan
def levels(open_ts):
    i0=open_ts-T0; m0=i0//60; p=b[i0]; L=[]
    add=lambda k,v: L.append((k,v)) if np.isfinite(v) else None
    add('prev15 hi',rng(H1,m0-15,m0,np.max)); add('prev15 lo',rng(L1,m0-15,m0,np.min))
    add('60m hi',rng(H1,m0-60,m0,np.max)); add('60m lo',rng(L1,m0-60,m0,np.min))
    add('4h hi',rng(H1,m0-240,m0,np.max)); add('4h lo',rng(L1,m0-240,m0,np.min))
    d0=(m0//1440)*1440
    if d0>=1440:
        ph=rng(H1,d0-1440,d0,np.max); pl=rng(L1,d0-1440,d0,np.min); pc=M[d0-1,-1]; P=(ph+pl+pc)/3
        for k,v in [('prevday hi',ph),('prevday lo',pl),('pivot P',P),('pivot R1',2*P-pl),('pivot S1',2*P-ph),('pivot R2',P+ph-pl),('pivot S2',P-(ph-pl))]: add(k,v)
    if m0>d0+15: add('today hi',rng(H1,d0,m0,np.max)); add('today lo',rng(L1,d0,m0,np.min))
    add('day open',M[d0,0])
    for st in (500,1000): add(f'round {st}',np.floor(p/st)*st); add(f'round {st}',np.ceil(p/st)*st)
    for v in hard(i0): add('hard',v)
    out={}
    for kind,v in L:
        if abs(v-p)>10:
            key=(kind if not kind.startswith('round') else kind,v>p)
            if key not in out or abs(v-p)<abs(out[key]-p): out[key]=v
    return [(k[0],k[1],v) for k,v in out.items()]
def events(S,tag):
    rows=[]
    for tk,g in S.groupby('ticker',sort=False):
        o=int(g.open_ts.iloc[0]); i0=o-T0; L=levels(o)
        if not L: continue
        for r in g.itertuples():
            i=int(r.snap_ts-T0); w=b[i-60:i+1]; mxw=w.max(); mnw=w.min(); seg=b[i0:i+1]; mxa=seg.max(); mna=seg.min(); cur=b[i]
            vol=np.diff(b[max(i-900,i0-900):i+1]).std()*np.sqrt(60)+1e-9
            for kind,isres,v in L:
                if isres:
                    if mxw>=v-5 and mxa<v+5 and cur<=mxw-10: rows.append((tk,r.k,'reject','res',kind,v,'NO',(mxw-cur)/vol,(v-mxw)/vol))
                    f=np.argmax(seg>=v+5) if mxa>=v+5 else -1
                    if f>=0 and i-60<i0+f<=i: rows.append((tk,r.k,'break','res',kind,v,'YES',(cur-v)/vol,(i-(i0+f))))
                else:
                    if mnw<=v+5 and mna>v-5 and cur>=mnw+10: rows.append((tk,r.k,'reject','sup',kind,v,'YES',(cur-mnw)/vol,(mnw-v)/vol))
                    f=np.argmax(seg<=v-5) if mna<=v-5 else -1
                    if f>=0 and i-60<i0+f<=i: rows.append((tk,r.k,'break','sup',kind,v,'NO',(v-cur)/vol,(i-(i0+f))))
    E=pd.DataFrame(rows,columns=['ticker','k','ev','dir','level','lv','side','strength','aux']); E.to_parquet(f'work/sr2_events_{tag}.parquet'); print(tag,E.shape,flush=True)
S=pd.read_parquet('work/replay_scored.parquet',columns=['ticker','k','snap_ts','open_ts'])
events(S,'min')
L=pd.read_parquet('work/lastmin.parquet',columns=['ticker','off','snap_ts','open_ts']).rename(columns={'off':'k'})
events(L,'last')
