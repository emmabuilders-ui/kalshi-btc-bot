import numpy as np, pandas as pd, pickle
T0=1764460800
b=pd.Series(np.load('work/brti.npy')).ffill().to_numpy().astype(float)
nmin=len(b)//60; M=b[:nmin*60].reshape(nmin,60); C1=pd.Series(M[:,-1])
n5=nmin//5; C5=pd.Series(C1[:n5*5].to_numpy().reshape(n5,5)[:,-1])
def bb(c,n=20):
    m=c.rolling(n).mean(); s=c.rolling(n).std(); return m.to_numpy(), s.to_numpy()
m1,s1=bb(C1); m5,s5=bb(C5)
A=pd.read_parquet('work/rsi_all.parquet')
i1=((A.snap_ts-T0)//60-1).astype(int).to_numpy(); i5=((A.snap_ts-T0)//300-1).astype(int).to_numpy()
sg=np.where(A.side=='YES',1,-1); px=A.brti.to_numpy()
for tf,(m,s,ii) in {1:(m1,s1,i1),5:(m5,s5,i5)}.items():
    mid=m[ii]; sd=s[ii]
    A[f'bbp{tf}']=sg*100*(px/mid-1); A[f'bbz{tf}']=sg*(px-mid)/np.where(sd>0,sd,np.nan)
zb=[-100,-2,-1,0,1,2,100]; zl=['<-2','-2..-1','-1..0','0..1','1..2','>2']
for tf in (1,5): A[f'z{tf}b']=pd.cut(A[f'bbz{tf}'],zb,right=False,labels=zl)
A.to_parquet('work/bb_all.parquet')
A=A[A.gg.isin(['3-5','5+'])].copy(); A['pn']=A.pnl*100
def flags(A,col,shuffle=False,seed=0):
    rng=np.random.default_rng(seed); out=[]
    for key,g in A.groupby(['model','db','gg','mg'],observed=True):
        lab=g[col].astype(str).to_numpy().copy()
        if shuffle: rng.shuffle(lab)
        g=g.assign(lab=lab); base={p:g[g.per==p].pn.mean() for p in ('Mar-Aug','Sep-Oct')}
        for l in zl:
            row={}; ok=True
            for p in ('Mar-Aug','Sep-Oct'):
                x=g[(g.per==p)&(g.lab==l)]
                if len(x)<60 or len(g[g.per==p])<150: ok=False; break
                row[p]=(len(x),x.pn.mean(),x.pn.mean()-base[p])
            if ok: out.append((key,l,row))
    return out
cnt=lambda o:(sum(1 for k,l,r in o if r['Mar-Aug'][2]>=1 and r['Sep-Oct'][2]>=1),sum(1 for k,l,r in o if r['Mar-Aug'][2]<=-1 and r['Sep-Oct'][2]<=-1),len(o))
res={}
for tf in (1,5):
    real=flags(A,f'z{tf}b'); sh=[cnt(flags(A,f'z{tf}b',True,s)) for s in range(10)]
    print('TF',tf,'real',cnt(real),'shuffled mean better/worse',np.mean([x[0] for x in sh]),np.mean([x[1] for x in sh]),'testable',sh[0][2]); res[tf]=real
    L=[t for t in real if t[2]['Mar-Aug'][2]>=1 and t[2]['Sep-Oct'][2]>=1]; L.sort(key=lambda t:-min(t[2]['Mar-Aug'][2],t[2]['Sep-Oct'][2]))
    for k,l,r in L[:8]: print('  +',k,l,{p:(r[p][0],round(r[p][1],1),round(r[p][2],1)) for p in r})
