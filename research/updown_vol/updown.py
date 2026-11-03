import numpy as np, pandas as pd, json
b=np.load('../tl/brti.npy'); T0=int(open('../tl/brti_t0.txt').read())
SEP=int(pd.Timestamp('2026-09-01',tz='UTC').timestamp())
def feats(t):
    i=t-T0
    idx=i[:,None]-1-60*(100-np.arange(101))[None,:]
    r=np.diff(np.log(b[idx]),axis=1)
    s100=r.std(axis=1,ddof=1); s10=r[:,-10:].std(axis=1,ddof=1)
    up=np.sqrt(2*np.mean(np.maximum(r,0)**2,axis=1)); dn=np.sqrt(2*np.mean(np.minimum(r,0)**2,axis=1))
    vb=np.digitize(np.log(s10/s100),[-0.3,0.3])
    return s100,up,dn,vb
# training samples: every minute mark with 1..30 min left before an hourly close, Dec-Aug
closes=np.arange((T0//3600+30)*3600, SEP, 3600)
t=(closes[:,None]-60*np.arange(1,31)[None,:]).ravel(); left=np.tile(np.arange(1,31),len(closes))
settle=np.array([b[c-T0-60:c-T0].mean() for c in closes]).repeat(30)
p=b[t-T0]; R=np.log(settle/p)
s100,up,dn,vb=feats(t)
ok=(s100>0)&(up>0)&(dn>0)
bk=np.where(left>20,30,np.where(left>8,15,5)); te=np.sqrt(left-0.5)
z_sym=R/(s100*te); z_ud=np.where(R>=0,R/(up*te),R/(dn*te))
G={}
for k in (30,15,5):
    for v in (0,1,2):
        m=ok&(bk==k)&(vb==v)
        G[('sym',k,v)]=np.sort(z_sym[m]); G[('ud',k,v)]=np.sort(z_ud[m])
print('train samples',ok.sum())
np.savez('updown_G.npz',**{f'{a}_{k}_{v}':G[(a,k,v)] for (a,k,v) in G})
# score snapshots
c=pd.read_pickle('snaps_tib.pkl'); c=c[c.model_now.notna()].copy()
ut=np.sort(c.t.unique()); S100,UP,DN,VB=feats(ut)
F=pd.DataFrame({'t':ut,'S100':S100,'UP':UP,'DN':DN,'VB2':VB}); c=c.merge(F,on='t')
c['bk']=np.where(c.left>20,30,np.where(c.left>8,15,5)); c['te']=np.sqrt(c.left-0.5)
def cdf(kind,x,bk,v):
    out=np.zeros(len(x))
    for k in (30,15,5):
        for vv in (0,1,2):
            m=(bk==k)&(v==vv); g=G[(kind,k,vv)]
            out[m]=np.searchsorted(g,x[m],side='right')/len(g)
    return out
lo=np.log(c.lo.values/c.brti.values); hi=np.log((c.hi.values+0.01)/c.brti.values)
def P(kind):
    if kind=='sym':
        xl=lo/(c.S100.values*c.te.values); xh=hi/(c.S100.values*c.te.values)
    else:
        sl=np.where(lo>=0,c.UP.values,c.DN.values); sh=np.where(hi>=0,c.UP.values,c.DN.values)
        xl=lo/(sl*c.te.values); xh=hi/(sh*c.te.values)
    Fh=np.where(np.isnan(xh),1.0,cdf(kind,np.nan_to_num(xh),c.bk.values,c.VB2.values))
    Fl=np.where(np.isnan(xl),0.0,cdf(kind,np.nan_to_num(xl),c.bk.values,c.VB2.values))
    return Fh-Fl
c['p_sym']=P('sym'); c['p_ud']=P('ud')
br=lambda p,y: float(np.mean((p-y)**2))
B=[]
for per in ('Dec-Aug','Sep-Oct'):
    g=c[c.per==per]; B.append(dict(period=per,snapshots=len(g),kalshi=br(g.mid,g.y),current_model=br(g.model_now,g.y),refit_same_as_current=br(g.p_sym,g.y),up_down_split=br(g.p_ud,g.y)))
B=pd.DataFrame(B); print(B.round(5).T)
c['cent']=(c.mid*100).round().astype(int)
def sc(g): return pd.Series(dict(snapshots=len(g),actual_yes_pct=g.y.mean()*100,kalshi_c=g.mid.mean()*100,current_c=g.model_now.mean()*100,refit_c=g.p_sym.mean()*100,updown_c=g.p_ud.mean()*100,
   brier_kalshi=br(g.mid,g.y),brier_current=br(g.model_now,g.y),brier_refit=br(g.p_sym,g.y),brier_updown=br(g.p_ud,g.y)))
bc=c.groupby(['cent','per']).apply(sc).round(5).reset_index(); bt=c.groupby(['left','per']).apply(sc).round(5).reset_index()
# trades with the bot's rule using each model (YES 75-89 model>=ask+10, 90-99 +2; first per band), for reference
with pd.ExcelWriter('/mnt/project-files/strategy/range_updown_vol_model.xlsx') as w:
    B.round(5).to_excel(w,sheet_name='overall',index=False); bc.to_excel(w,sheet_name='by_cent',index=False); bt.to_excel(w,sheet_name='by_minutes_left',index=False)
c[['ticker','t','per','left','mid','ask','bid','y','model_now','p_sym','p_ud','UP','DN','S100']].to_pickle('updown_scored.pkl')
s=bt[bt.per=='Sep-Oct'].set_index('left'); print(s.loc[[30,20,15,10,5,1],['brier_kalshi','brier_current','brier_refit','brier_updown']])
s=bc[bc.per=='Sep-Oct'].set_index('cent'); print(s.loc[[5,20,50,70,80,90],['actual_yes_pct','current_c','updown_c','brier_kalshi','brier_current','brier_refit','brier_updown']])
