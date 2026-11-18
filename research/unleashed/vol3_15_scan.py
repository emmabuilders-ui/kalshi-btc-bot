import numpy as np, pandas as pd
T0=1764460800
b=pd.Series(np.load('work/brti.npy')).ffill().to_numpy().astype(float)
d=np.diff(b,prepend=b[0]); c1=np.cumsum(d); c2=np.cumsum(d*d)
def sd(i,w):
    n=w; s1=c1[i]-c1[i-w]; s2=c2[i]-c2[i-w]; return np.sqrt(np.maximum(s2/n-(s1/n)**2,0))
A=pd.read_parquet('work/rsi_all.parquet')
i=(A.snap_ts-T0).to_numpy().astype(int); px=A.brti.to_numpy()
A['v3']=sd(i,180)*np.sqrt(60)/px*100        # 3-min vol, % per minute
A['v15']=sd(i,900)*np.sqrt(60)/px*100
A['v3r']=A.v3/A.v15.clip(lower=1e-6)
zl=['Q1','Q2','Q3','Q4','Q5']
for c in ('v3r',):
    e=A[A.per=='Mar-Aug'][c].quantile([.2,.4,.6,.8]).to_numpy(); print(c,'edges',e.round(4))
    A[c+'b']=pd.cut(A[c],[-1,*e,1e9],labels=zl)
A.to_parquet('work/vol315_all.parquet')
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
for c in ('v3rb',):
    real=flags(A,c); sh=[cnt(flags(A,c,True,s)) for s in range(10)]
    print(c,'real',cnt(real),'shuffled better/worse',np.mean([x[0] for x in sh]),np.mean([x[1] for x in sh]))
    for m in ('Direct','Correction'):
        r=[t for t in real if t[0][0]==m]; print('  ',m,cnt(r))
# pocket + all-trade tables
P=A[(A.model=='Direct')&(A.gg=='5+')&(A.db.isin(['.1..+.2','>+.2']))&(A.mg!='1-3')]
def s(g):
    n=len(g); return f"{n}, win {100*g.win.mean():.0f}% vs {100*g.ask_s.mean():.0f}, avg {g.pn.mean():+.1f} med {g.pn.median():+.1f}"
for name,D in (('DIRECT pocket',P),('DIRECT gap>=3',A[A.model=='Direct']),('CORRECTION gap>=3',A[A.model=='Correction'])):
  for c in ('v3rb',):
    print(name,c)
    for l in zl: print(' ',l,' | '.join(s(D[(D.per==p)&(D[c].astype(str)==l)]) for p in ('Mar-Aug','Sep-Oct')))
