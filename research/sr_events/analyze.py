import numpy as np, pandas as pd
S=pd.read_parquet('work/replay_scored.parquet',columns=['ticker','k','ask','bid','res','period','min_left','mid'])
E=pd.read_parquet('work/sr_events.parquet')
def mk(df,side):
    d=df.copy(); d['side']=side
    d['a']=d.ask if side=='YES' else 1-d.bid
    d['win']=(d.res==1) if side=='YES' else (d.res==0)
    d['mid_s']=d.mid if side=='YES' else 1-d.mid
    return d
T=pd.concat([mk(S,'YES'),mk(S,'NO')]); T=T[(T.a>=0.01)&(T.a<=0.99)&T.a.notna()].copy()
T['win']=T.win.astype(float); T['fee']=0.07*T.a*(1-T.a); T['pn']=100*(T.win-T.a-T.fee)
T['pb']=pd.cut(T.a*100,[0,10,20,30,40,50,60,70,80,90,100],labels=['01-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-99'])
T['mg']=T.min_left.round().astype(int)
cell=['side','mg','pb','period']
base=T.groupby(cell,observed=True).pn.mean().rename('base'); T=T.join(base,on=cell); T['ex']=T.pn-T.base
T.to_parquet('work/sr_trades_all.parquet')
X=E.merge(T,on=['ticker','k','side'],how='inner'); X.to_parquet('work/sr_events_trades.parquet')
print(len(E),len(X),'baseline all',T.groupby('period').pn.mean().round(2).to_dict())
def summ(g):
    n=len(g); sd=g.ex.std()
    return pd.Series(dict(n=n,ask=round(100*g.a.mean(),1),win=round(100*g.win.mean(),1),edge=round(100*(g.win.mean()-g.a.mean()),1),avg=round(g.pn.mean(),2),med=round(g.pn.median(),2),ctrl=round(g.base.mean(),2),excess=round(g.ex.mean(),2),t=round(g.ex.mean()/(sd/np.sqrt(n)),1) if n>30 else np.nan))
D=X.drop_duplicates(['ticker','k','side','ev']) # any level, one per snapshot
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
tabs={}
tabs['overall']=D.groupby(['ev','side','period']).apply(summ,include_groups=False)
tabs['by_dir']=X.groupby(['ev','dir','side','period']).apply(summ,include_groups=False)
tabs['by_level']=X.groupby(['ev','dir','level','period']).apply(summ,include_groups=False)
tabs['by_minute']=D.groupby(['ev','side','mg','period']).apply(summ,include_groups=False)
tabs['by_price']=D.groupby(['ev','side','pb','period'],observed=True).apply(summ,include_groups=False)
tabs['minute_x_price']=D.groupby(['ev','side','mg','pb','period'],observed=True).apply(summ,include_groups=False)
with pd.ExcelWriter('work/sr_events_results.xlsx') as w:
    for k,v in tabs.items(): v.reset_index().to_excel(w,sheet_name=k,index=False)
print(tabs['overall'].to_string()); print(tabs['by_dir'].to_string())
# chance benchmark for minute_x_price cells: both periods excess>=1c with n>=60
m=tabs['minute_x_price'].reset_index(); m=m[m.n>=60]
p=m.pivot_table(index=['ev','side','mg','pb'],columns='period',values=['n','excess'],observed=True).dropna()
good=p[(p[('excess','Dec-Aug')]>=1)&(p[('excess','Sep-Oct')]>=1)]; bad=p[(p[('excess','Dec-Aug')]<=-1)&(p[('excess','Sep-Oct')]<=-1)]
print('cells',len(p),'both>=+1c',len(good),'both<=-1c',len(bad))
for ev in ('reject','break'):
    q=p.loc[ev]; print(ev,len(q),((q[('excess','Dec-Aug')]>=1)&(q[('excess','Sep-Oct')]>=1)).sum(),((q[('excess','Dec-Aug')]<=-1)&(q[('excess','Sep-Oct')]<=-1)).sum())
print(good.sort_values(('excess','Sep-Oct'),ascending=False).head(12).to_string())
