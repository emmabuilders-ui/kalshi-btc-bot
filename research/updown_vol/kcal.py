import numpy as np, pandas as pd, json
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
c=pd.read_pickle('updown_scored.pkl')
fee=lambda p: np.ceil(np.round(0.07*20*p*(1-p)*100,6))/100/20
tr=(c.per=='Dec-Aug').values
L=np.log(c.mid/(1-c.mid)); c['L']=L; c['lg']=np.log(c.left-0.5)
X=pd.DataFrame({'L':L,'lg':c.lg,'Lxlg':L*c.lg,'L2':L**2,'L2xlg':L**2*c.lg})
m=LogisticRegression(C=1e4,max_iter=5000).fit(X[tr],c.y[tr]); c['p_kcal']=m.predict_proba(X)[:,1]
# lookup table: cent x minute group, from Dec-Aug, isotonic per minute group
c['mg']=pd.cut(c.left,[0,2,5,10,15,20,30],labels=['1-2','3-5','6-10','11-15','16-20','21-30'])
c['p_tab']=np.nan
for g,d in c.groupby('mg',observed=True):
    t=d[d.per=='Dec-Aug']; iso=IsotonicRegression(out_of_bounds='clip').fit(t.mid,t.y)
    c.loc[d.index,'p_tab']=iso.predict(d.mid)
json.dump({'features':list(X.columns),'intercept':float(m.intercept_[0]),'coef':dict(zip(X.columns,map(float,m.coef_[0])))},open('kcal.json','w'),indent=1)
br=lambda p,y: float(np.mean((p-y)**2))
B=[]
for per in ('Dec-Aug','Sep-Oct'):
    g=c[c.per==per]; B.append(dict(period=per,kalshi=br(g.mid,g.y),kalshi_corrected_formula=br(g.p_kcal,g.y),kalshi_corrected_table=br(g.p_tab,g.y),rebuilt_range_model=br(g.p_sym,g.y),current_model=br(g.model_now,g.y)))
B=pd.DataFrame(B); print(B.round(5).T)
bt=c.groupby(['left','per']).apply(lambda g: pd.Series(dict(kalshi=br(g.mid,g.y),corrected=br(g.p_kcal,g.y),table=br(g.p_tab,g.y),rebuilt=br(g.p_sym,g.y)))).reset_index()
c['cent']=(c.mid*100).round().astype(int)
bc=c.groupby(['cent','per']).apply(lambda g: pd.Series(dict(n=len(g),actual=g.y.mean()*100,kalshi=br(g.mid,g.y),corrected=br(g.p_kcal,g.y),table=br(g.p_tab,g.y),corrected_c=g.p_kcal.mean()*100))).reset_index()
print(bt[bt.per=='Sep-Oct'].set_index('left').loc[[30,20,15,10,5,1]].round(5))
# trades: YES at ask when corrected >= ask + gap, first minute per band; also NO when 1-corrected >= no_ask+gap
d=c.copy(); d['won']=d.y==1; d['pnl']=d.won-d.ask-fee(d.ask); d['no_p']=1-d.bid; d['no_pnl']=(~d.won)-d.no_p-fee(d.no_p)
f=lambda g,col,px: pd.Series(dict(bands=len(g),win=round((g.won if col=='pnl' else ~g.won).mean()*100,1),px=round(g[px].mean()*100,1),pnl=round(g[col].mean()*100,2)))
T=[]
for model in ('p_kcal','p_tab'):
    for gap in (0.0,0.02,0.03,0.05):
        s=d[(d[model]>=d.ask+gap)&d.ask.between(0.05,0.97)].sort_values('t').groupby('ticker').head(1)
        r=s.groupby('per').apply(lambda g: f(g,'pnl','ask')); 
        for per in r.index: T.append(dict(model=model,side='YES',gap=gap,period=per,**r.loc[per].to_dict()))
        s=d[((1-d[model])>=d.no_p+gap)&d.no_p.between(0.05,0.97)].sort_values('t').groupby('ticker').head(1)
        r=s.groupby('per').apply(lambda g: f(g,'no_pnl','no_p'))
        for per in r.index: T.append(dict(model=model,side='NO',gap=gap,period=per,**r.loc[per].to_dict()))
T=pd.DataFrame(T); print(T.pivot_table(index=['model','side','gap'],columns='period',values=['bands','win','px','pnl']).to_string())
with pd.ExcelWriter('/mnt/project-files/strategy/kalshi_corrected_model.xlsx') as w:
    B.round(5).to_excel(w,sheet_name='overall',index=False); bt.round(5).to_excel(w,sheet_name='by_minutes_left',index=False); bc.round(5).to_excel(w,sheet_name='by_cent',index=False); T.to_excel(w,sheet_name='trades',index=False)
c[['ticker','t','per','left','mid','ask','bid','y','p_kcal','p_tab','p_sym']].to_pickle('kcal_scored.pkl')
