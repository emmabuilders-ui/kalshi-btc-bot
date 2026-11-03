import numpy as np, pandas as pd, json
from sklearn.linear_model import LogisticRegression
c=pd.read_pickle('snaps_tib.pkl'); c=c[c.in_this_hour.notna()].copy()
fee=lambda p: np.ceil(np.round(0.07*20*p*(1-p)*100,6))/100/20
lv=lambda x: pd.cut(x,[-0.001,0.0,0.25,0.5,0.75,1.0],labels=['0%','1-25%','25-50%','50-75%','75%+'])
for w in ('in_this_hour','in_prev_hour','in_last15'): c[w+'_lvl']=lv(c[w])
c['yes_p']=c.ask; c['no_p']=1-c.bid
rows=[]
for side in ('yes','no'):
    px=c[f'{side}_p']; won=(c.y==1) if side=='yes' else (c.y==0)
    d=c.assign(px=px,won=won,pnl=won-px-fee(px),cent=(px*100).round().astype(int))
    d=d[d.cent.between(2,98)]
    for w in ('in_this_hour','in_prev_hour','in_last15'):
        g=d.groupby([w+'_lvl','per'],observed=True).agg(band_minutes=('won','size'),bands=('ticker','nunique'),win_pct=('won','mean'),avg_price=('px','mean'),pnl=('pnl','mean')).reset_index()
        g['win_minus_price']=(g.win_pct-g.avg_price)*100; g['win_pct']*=100; g['avg_price']*=100; g['pnl']*=100
        g.insert(0,'window',w); g.insert(0,'side',side.upper()); rows.append(g.rename(columns={w+'_lvl':'time_in_band'}))
    for w in ('in_this_hour','in_prev_hour','in_last15'):
        g=d.groupby(['cent',w+'_lvl','per'],observed=True).agg(band_minutes=('won','size'),win_pct=('won','mean'),avg_price=('px','mean'),pnl=('pnl','mean')).reset_index()
        g['win_minus_price']=(g.win_pct-g.avg_price)*100; g['win_pct']*=100; g['avg_price']*=100; g['pnl']*=100
        g.insert(0,'window',w); g.insert(0,'side',side.upper()); g.rename(columns={w+'_lvl':'time_in_band'}).round(2).to_pickle(f'bycent_{side}_{w}.pkl')
summ=pd.concat(rows).round(2)
# ---- fit add-on
L=lambda p: np.log(np.clip(p,1e-4,1-1e-4)/(1-np.clip(p,1e-4,1-1e-4)))
c['Lm']=L(c.model_now)
F=['in_this_hour','in_prev_hour','in_last15']
for f in F: c[f+'_x']=c[f]*c.Lm
tr=c.per=='Dec-Aug'
m0=LogisticRegression(C=1e6,max_iter=2000).fit(c.loc[tr,['Lm']],c.y[tr])
X=['Lm']+F+[f+'_x' for f in F]
m1=LogisticRegression(C=1e6,max_iter=3000).fit(c.loc[tr,X],c.y[tr])
c['p_recal']=m0.predict_proba(c[['Lm']])[:,1]; c['p_tib']=m1.predict_proba(c[X])[:,1]
coef={'intercept':float(m1.intercept_[0]),**{k:float(v) for k,v in zip(X,m1.coef_[0])}}
coef0={'intercept':float(m0.intercept_[0]),'Lm':float(m0.coef_[0][0])}
json.dump({'time_in_band_addon':coef,'recalibration_only':coef0},open('tib_addon.json','w'),indent=1)
br=lambda p,y: float(np.mean((p-y)**2))
B=[]
for per in ('Dec-Aug','Sep-Oct'):
    g=c[c.per==per]
    B.append(dict(period=per,snapshots=len(g),kalshi=br(g.mid,g.y),model_now=br(g.model_now,g.y),model_recalibrated=br(g.p_recal,g.y),model_plus_time_in_band=br(g.p_tib,g.y)))
B=pd.DataFrame(B); print(B.round(5).to_string()); print(json.dumps(coef,indent=0))
c['cent']=(c.mid*100).round().astype(int)
def sc(g): return pd.Series(dict(snapshots=len(g),actual_yes_pct=g.y.mean()*100,kalshi_c=g.mid.mean()*100,model_now_c=g.model_now.mean()*100,model_tib_c=g.p_tib.mean()*100,brier_kalshi=br(g.mid,g.y),brier_model_now=br(g.model_now,g.y),brier_recal=br(g.p_recal,g.y),brier_tib=br(g.p_tib,g.y)))
bc=c.groupby(['cent','per']).apply(sc).round(5).reset_index()
bt=c.groupby(['left','per']).apply(sc).round(5).reset_index()
c[['ticker','t','per','left','mid','ask','bid','y','model_now','p_recal','p_tib']+F].to_pickle('tib_scored.pkl')
with pd.ExcelWriter('/mnt/project-files/strategy/range_time_in_band_v2.xlsx') as w:
    summ.to_excel(w,sheet_name='by_time_in_band',index=False)
    B.round(5).to_excel(w,sheet_name='brier_overall',index=False)
    bc.to_excel(w,sheet_name='brier_by_cent',index=False); bt.to_excel(w,sheet_name='brier_by_minutes_left',index=False)
    for side in ('yes','no'):
        pd.concat([pd.read_pickle(f'bycent_{side}_{w2}.pkl') for w2 in F]).to_excel(w,sheet_name=f'{side.upper()}_each_cent',index=False)
pd.set_option('display.width',220)
print(summ.to_string())
