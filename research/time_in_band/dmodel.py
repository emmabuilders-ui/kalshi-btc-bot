import numpy as np, pandas as pd, json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
c=pd.read_pickle('snaps_tib.pkl'); c=c[c.in_this_hour.notna()&c.lo.notna()&c.hi.notna()&c.s100.notna()].copy()
te=c.left-0.5; mv=c.brti*c.s100*np.sqrt(te)            # typical $ move to settlement
ctr=(c.lo+c.hi+0.01)/2; half=(c.hi+0.01-c.lo)/2
c['dist_pct']=((c.brti-ctr)/half).abs()                 # 0 center, 1 edge
c['zc']=(c.brti-ctr).abs()/mv                           # distance from center in move units
c['wz']=half/mv                                         # half-width in move units
c['logt']=np.log(te)
for k in (1,2): c[f'vb{k}']=(c.vb==k).astype(int)
F=['zc','wz','logt','in_this_hour','in_prev_hour','in_last15','vb1','vb2']
X=c[F].copy()
X['zc2']=c.zc**2; X['wz2']=c.wz**2; X['zcwz']=c.zc*c.wz; X['zc_lt']=c.zc*c.logt; X['wz_lt']=c.wz*c.logt; X['dp']=c.dist_pct.clip(upper=5)
X['zc_clip']=c.zc.clip(upper=4); X['wz_clip']=c.wz.clip(upper=4)
cols=list(X.columns)
tr=(c.per=='Dec-Aug').values
lr=LogisticRegression(C=1.0,max_iter=5000).fit(X[tr],c.y[tr])
c['p_dist']=lr.predict_proba(X)[:,1]
gb=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,max_leaf_nodes=31,min_samples_leaf=200).fit(c.loc[tr,F+['dist_pct']],c.y[tr])
c['p_gb']=gb.predict_proba(c[F+['dist_pct']])[:,1]
json.dump({'features':cols,'intercept':float(lr.intercept_[0]),'coef':dict(zip(cols,map(float,lr.coef_[0])))},open('dist_model.json','w'),indent=1)
br=lambda p,y: float(np.mean((p-y)**2))
B=[]
for per in ('Dec-Aug','Sep-Oct'):
    g=c[c.per==per]
    B.append(dict(period=per,snapshots=len(g),kalshi=br(g.mid,g.y),current_model=br(g.model_now,g.y),new_distance_model=br(g.p_dist,g.y),new_model_boosted_trees=br(g.p_gb,g.y)))
B=pd.DataFrame(B); print(B.round(5).to_string())
c['cent']=(c.mid*100).round().astype(int)
def sc(g): return pd.Series(dict(snapshots=len(g),actual_yes_pct=g.y.mean()*100,kalshi_c=g.mid.mean()*100,current_model_c=g.model_now.mean()*100,new_model_c=g.p_dist.mean()*100,
   brier_kalshi=br(g.mid,g.y),brier_current=br(g.model_now,g.y),brier_new=br(g.p_dist,g.y),brier_trees=br(g.p_gb,g.y)))
bc=c.groupby(['cent','per']).apply(sc).round(5).reset_index(); bt=c.groupby(['left','per']).apply(sc).round(5).reset_index()
c[['ticker','t','per','left','mid','ask','bid','y','model_now','p_dist','p_gb','dist_pct','in_prev_hour','in_last15','in_this_hour']].to_pickle('dist_scored.pkl')
with pd.ExcelWriter('/mnt/project-files/strategy/range_distance_model_scores.xlsx') as w:
    B.round(5).to_excel(w,sheet_name='overall',index=False); bc.to_excel(w,sheet_name='by_cent',index=False); bt.to_excel(w,sheet_name='by_minutes_left',index=False)
s=bt[bt.per=='Sep-Oct'].set_index('left')[['brier_kalshi','brier_current','brier_new','brier_trees']]
print(s.loc[[30,20,15,10,5,3,1]])
s=bc[bc.per=='Sep-Oct'].set_index('cent')[['snapshots','actual_yes_pct','brier_kalshi','brier_current','brier_new']]
print(s.loc[[5,10,20,30,50,70,80,90,95]])
