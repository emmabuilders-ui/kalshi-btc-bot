import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/user/kalshi-btc-bot')
from kalshi_btc_bot.range_paper import RangeModel
M=RangeModel()
c=pd.read_pickle('snaps.pkl'); c=c[c.model_yes.notna()].copy()
c['per']=np.where(pd.to_datetime(c.t,unit='s')>='2026-09-01','Sep-Oct','Dec-Aug')
c['mid']=((c.ask+c.bid)/2).clip(0.005,0.995); c=c[(c.ask-c.bid).between(0,0.05)&(c.ask>0)].copy()
c['lo']=np.where(c.strike_type=='less',np.nan,c.floor_strike); c['hi']=np.where(c.strike_type=='greater',np.nan,c.cap_strike)
c['bk']=np.where(c.left>20,30,np.where(c.left>8,15,5))
def pvec(price,d):
    te=d.left.values-0.5; sc=d.s100.values*np.sqrt(te); out=np.zeros(len(d))
    xl=np.log(d.lo.values/price)/sc; xh=np.log((d.hi.values+0.01)/price)/sc
    for key,cdf in M.cdf.items():
        m=(d.bk.values==key[0])&(d.vb.values==key[1])
        if not m.any(): continue
        Fh=np.where(np.isnan(xh[m]),1.0,np.interp(np.nan_to_num(xh[m]),cdf,M.qs,left=0,right=1))
        Fl=np.where(np.isnan(xl[m]),0.0,np.interp(np.nan_to_num(xl[m]),cdf,M.qs,left=0,right=1))
        out[m]=Fh-Fl
    return out
base=pvec(c.brti.values,c); print('check vs stored', np.abs(base-c.model_yes.values).max())
# trend features
ts=np.sort(c.t.unique())
for N in (30,60,120,300):
    e=pd.read_csv(f'../tl/ev_N{N}.csv').sort_values('ts'); et=e.ts.values; ed=np.where(e.side=='yes',1,-1)
    k=np.searchsorted(et,ts,side='left')-1      # last break strictly before t
    age=np.where(k>=0,(ts-et[np.maximum(k,0)])/60,np.inf); dr=np.where(k>=0,ed[np.maximum(k,0)],0)
    A=dict(zip(ts,age)); D=dict(zip(ts,dr))
    c[f'age{N}']=c.t.map(A); c[f'dir{N}']=c.t.map(D)
c.to_pickle('snaps_tl.pkl')
br=lambda p,y: np.mean((p-y)**2)
res=[]
for per in ('Dec-Aug','Sep-Oct'):
    g=c.per==per
    res.append(dict(model='Kalshi mid',window='',k=0,per=per,brier=br(c.mid[g],c.y[g])))
    res.append(dict(model='Range model (as is)',window='',k=0,per=per,brier=br(base[g],c.y[g])))
tr=(c.per=='Dec-Aug').values
best=[]
for N in (30,60,120,300):
    for W in (5,15,30):
        dirv=np.where(c[f'age{N}'].values<=W,c[f'dir{N}'].values,0)
        sc=c.s100.values*np.sqrt(c.left.values-0.5)
        def P(k): return pvec(c.brti.values*np.exp(k*dirv*sc),c)
        grid=[-1,-0.5,-0.25,-0.1,-0.05,-0.02,0,0.02,0.05,0.1,0.25,0.5,1]
        bs=[br(P(k)[tr],c.y.values[tr]) for k in grid]; kb=grid[int(np.argmin(bs))]
        p=P(kb)
        for per in ('Dec-Aug','Sep-Oct'):
            g=(c.per==per).values
            res.append(dict(model=f'Range model + trend lines {N}s',window=f'break in last {W} min',k=kb,per=per,brier=br(p[g],c.y.values[g])))
        print(N,W,kb,flush=True)
r=pd.DataFrame(res); r.to_csv('fit_results.csv',index=False)
print(r.pivot_table(index=['model','window','k'],columns='per',values='brier').round(5).to_string())

# ---------- pick the best add-on on Dec-Aug, score by price
fr=r[r.model.str.startswith('Range model + trend')]
tri=fr[fr.per=='Dec-Aug'].sort_values('brier').iloc[0]
bN=int(tri.model.split()[-1].rstrip('s')); bW=int(tri.window.split()[3]); bk=float(tri.k)
dirv=np.where(c[f'age{bN}'].values<=bW,c[f'dir{bN}'].values,0)
c['model_tl']=pvec(c.brti.values*np.exp(bk*dirv*c.s100.values*np.sqrt(c.left.values-0.5)),c)
c['model_now']=base
import json
json.dump({'swing_N_s':bN,'window_min':bW,'k':bk},open('best_addon.json','w'))
c['cent']=(c.mid*100).round().astype(int)
def sc(g): return pd.Series(dict(snapshots=len(g),actual_yes_pct=g.y.mean()*100,kalshi_mid_c=g.mid.mean()*100,model_now_c=g.model_now.mean()*100,model_trend_c=g.model_tl.mean()*100,
  brier_kalshi=br(g.mid,g.y),brier_model_now=br(g.model_now,g.y),brier_model_trend=br(g.model_tl,g.y)))
byc=c.groupby(['cent','per']).apply(sc).round(5).reset_index()
tot=c.groupby('per').apply(sc).round(5).reset_index()
with pd.ExcelWriter('/mnt/project-files/strategy/range_model_trend_line_fit.xlsx') as w:
    tot.to_excel(w,sheet_name='overall',index=False)
    r.pivot_table(index=['model','window','k'],columns='per',values='brier').round(5).reset_index().to_excel(w,sheet_name='all_versions',index=False)
    for per in ('Dec-Aug','Sep-Oct'): byc[byc.per==per].to_excel(w,sheet_name=f'by_cent_{per}',index=False)
print(bN,bW,bk); print(tot.to_string())
c.to_pickle('snaps_tl.pkl')
