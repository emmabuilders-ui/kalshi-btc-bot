import numpy as np, pandas as pd, json
from sklearn.linear_model import LogisticRegression
c=pd.read_pickle('snaps_ab.pkl'); c=c[c.above15.notna()].copy()
ctr=(c.lo+c.hi+0.01)/2; half=(c.hi+0.01-c.lo)/2
c['d']=((c.brti-ctr)/half).abs(); up=c.brti>=ctr
c['same15']=np.where(up,c.above15,c.below15); c['opp15']=np.where(up,c.below15,c.above15)
c['same60']=np.where(up,c.above60,c.below60); c['opp60']=np.where(up,c.below60,c.above60)
c['dbin']=(np.floor(c.d*10)*10).clip(upper=300).astype(int)
D=pd.get_dummies(c.dbin,prefix='d').astype(float)
S=c[['same15','opp15','same60','opp60']]
T=pd.get_dummies(c.left,prefix='m').astype(float)
c['logt']=np.log(c.left-0.5)
X1=pd.concat([D,S],axis=1)
X2=pd.concat([D,S,T,(c.d.clip(upper=10)*c.logt).rename('d_x_logt')],axis=1)
tr=(c.per=='Dec-Aug').values
br=lambda p,y: float(np.mean((p-y)**2))
out={}
for name,X in (('strict (distance + above/below only)',X1),('plus minutes left',X2)):
    m=LogisticRegression(C=1.0,max_iter=5000).fit(X[tr],c.y[tr]); c[name]=m.predict_proba(X)[:,1]
    out[name]={'intercept':float(m.intercept_[0]),'coef':dict(zip(X.columns,map(float,m.coef_[0])))}
json.dump(out,open('ab_model.json','w'),indent=1)
# pure count table: settle-in-band frequency by distance bin x minutes left (Dec-Aug), no fitting
tab=c[tr].groupby(['dbin','left']).y.mean()
c['count_table']=[tab.get((a,b),np.nan) for a,b in zip(c.dbin,c.left)]
c['count_table']=c.count_table.fillna(c.y[tr].mean())
M=['strict (distance + above/below only)','plus minutes left','count_table']
B=[]
for per in ('Dec-Aug','Sep-Oct'):
    g=c[c.per==per]; r=dict(period=per,snapshots=len(g),kalshi=br(g.mid,g.y),current_model=br(g.model_now,g.y))
    for k in M: r[k]=br(g[k],g.y)
    B.append(r)
B=pd.DataFrame(B); print(B.round(5).T)
c['cent']=(c.mid*100).round().astype(int)
def sc(g):
    r=dict(snapshots=len(g),actual_yes_pct=g.y.mean()*100,brier_kalshi=br(g.mid,g.y),brier_current=br(g.model_now,g.y))
    for k in M: r['brier '+k]=br(g[k],g.y)
    return pd.Series(r)
bc=c.groupby(['cent','per']).apply(sc).round(5).reset_index(); bt=c.groupby(['left','per']).apply(sc).round(5).reset_index()
# how often settled above / inside / below, by distance bin and last-15 share
c['settle']=np.where(c.y==1,'inside',np.where(c.result.isna(),'',''))
with pd.ExcelWriter('/mnt/project-files/strategy/range_distance_abovebelow_model.xlsx') as w:
    B.round(5).to_excel(w,sheet_name='overall',index=False); bc.to_excel(w,sheet_name='by_cent',index=False); bt.to_excel(w,sheet_name='by_minutes_left',index=False)
    t=c[tr].groupby(['dbin','left']).agg(snapshots=('y','size'),settled_inside_pct=('y','mean')).reset_index(); t['settled_inside_pct']*=100
    t.round(2).to_excel(w,sheet_name='count_table_DecAug',index=False)
s=bt[bt.per=='Sep-Oct'].set_index('left'); print(s.loc[[30,20,15,10,5,1],['brier_kalshi','brier_current']+['brier '+k for k in M]])
s=bc[bc.per=='Sep-Oct'].set_index('cent'); print(s.loc[[5,20,50,80,90],['brier_kalshi','brier_current']+['brier '+k for k in M]])
