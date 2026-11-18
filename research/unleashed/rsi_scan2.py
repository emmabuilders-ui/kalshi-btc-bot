import numpy as np, pandas as pd
A=pd.read_parquet('work/rsi_all.parquet'); A=A[A.gg.isin(['3-5','5+'])].copy()
A['pn']=A.pnl*100
rl=['<30','30-45','45-55','55-70','>70']
def flags(A,tf,shuffle=False,seed=0):
    rng=np.random.default_rng(seed); out=[]
    col='r1b' if tf==1 else 'r5b'
    for key,g in A.groupby(['model','db','gg','mg'],observed=True):
        lab=g[col].astype(str).to_numpy().copy()
        if shuffle: rng.shuffle(lab)
        g=g.assign(lab=lab)
        base={p:g[g.per==p].pn.mean() for p in ('Mar-Aug','Sep-Oct')}
        for l in rl:
            row={}
            ok=True
            for p in ('Mar-Aug','Sep-Oct'):
                x=g[(g.per==p)&(g.lab==l)]
                if len(x)<60 or len(g[g.per==p])<150: ok=False; break
                row[p]=(len(x),x.pn.mean(),x.pn.mean()-base[p])
            if ok: out.append((key,l,row))
    return out
def cnt(out,thr=1.0):
    return sum(1 for k,l,r in out if r['Mar-Aug'][2]>=thr and r['Sep-Oct'][2]>=thr), sum(1 for k,l,r in out if r['Mar-Aug'][2]<=-thr and r['Sep-Oct'][2]<=-thr), len(out)
for tf in (1,5):
    real=flags(A,tf); print('TF',tf,'real (better,worse,testable):',cnt(real))
    sh=[cnt(flags(A,tf,True,s)) for s in range(10)]; print('  shuffled mean (better,worse,testable):',np.mean([x[0] for x in sh]),np.mean([x[1] for x in sh]),sh[0][2])
    if tf==1: R1=real
    else: R5=real
import pickle; pickle.dump((R1,R5),open('work/rsi_flags.pkl','wb'))
# list the strongest consistent improvements
for tf,real in ((1,R1),(5,R5)):
    L=[(k,l,r) for k,l,r in real if r['Mar-Aug'][2]>=1 and r['Sep-Oct'][2]>=1]
    L.sort(key=lambda t:-(min(t[2]['Mar-Aug'][2],t[2]['Sep-Oct'][2])))
    print('TF',tf,'top improvements')
    for k,l,r in L[:14]: print(' ',k,l,{p:(r[p][0],round(r[p][1],1),round(r[p][2],1)) for p in r})
    L=[(k,l,r) for k,l,r in real if r['Mar-Aug'][2]<=-1 and r['Sep-Oct'][2]<=-1]
    L.sort(key=lambda t:max(t[2]['Mar-Aug'][2],t[2]['Sep-Oct'][2]))
    print('TF',tf,'top worsenings')
    for k,l,r in L[:8]: print(' ',k,l,{p:(r[p][0],round(r[p][1],1),round(r[p][2],1)) for p in r})
