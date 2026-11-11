import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
name=sys.argv[1]
S=pd.read_parquet(f'work/{name}_scored.parquet')
S['day']=pd.to_datetime(S.t,unit='s').dt.floor('D'); S['month']=pd.to_datetime(S.t,unit='s').dt.strftime('%Y-%m')
S['per']=np.where(S.t>=pd.Timestamp('2026-10-26',tz='UTC').timestamp(),'Oct26-31',np.where(S.t>=pd.Timestamp('2026-09-01',tz='UTC').timestamp(),'Sep-Oct','Mar-Aug'))
MODELS=['kalshi','direct','corr','lasso']
rng=np.random.default_rng(0)
def blk(df,m):
    d=(df['p_'+m]-df.res)**2-(df.p_kalshi-df.res)**2
    g=d.groupby(df.day).agg(['sum','count']); s=g['sum'].to_numpy(); c=g['count'].to_numpy()
    bs=[s[i].sum()/c[i].sum() for i in (rng.integers(0,len(s),len(s)) for _ in range(300))]
    return d.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)
rows=[]
for per,df in [('Mar-Aug',S[(S.t<pd.Timestamp('2026-09-01',tz='UTC').timestamp())&S.p_direct.notna()]),('Sep-Oct',S[S.t>=pd.Timestamp('2026-09-01',tz='UTC').timestamp()]),('Oct26-31',S[S.per=='Oct26-31'])]:
    for m in MODELS:
        b=((df['p_'+m]-df.res)**2).mean(); auc=roc_auc_score(df.res,df['p_'+m])
        dm,lo,hi=blk(df,m) if m!='kalshi' else (0,0,0)
        rows.append(dict(period=per,model=m,rows=len(df),markets=df.ticker.nunique(),brier=round(b,5),brier_minus_kalshi_x1000=round(dm*1000,3),ci_lo=round(lo*1000,3),ci_hi=round(hi*1000,3),auc=round(auc,4)))
R=pd.DataFrame(rows); print(R.to_string())
# Brier by minutes left (Sep-Oct)
bm=[]
df=S[S.per!='Mar-Aug']
for g,lab in [((1,3),'1-3'),((4,8),'4-8'),((9,15),'9-15'),((16,30),'16-30')]:
    d=df[(df.min_left>=g[0])&(df.min_left<=g[1])]
    bm.append({'min_left':lab,'rows':len(d),**{m:round(((d['p_'+m]-d.res)**2).mean(),5) for m in MODELS}})
BM=pd.DataFrame(bm); print(BM.to_string())
# trades: one per market, first qualifying minute (largest secs_left), gap g
def fee(p): return 0.07*p*(1-p)
def trades(df,m,g,lo=0.0,hi=1.0,sides=('Y','N')):
    d=df.dropna(subset=['p_'+m]).copy(); p=d['p_'+m]
    ya=d.yes_ask; nb=1-d.yes_bid
    ey=p-ya-fee(ya); en=(1-p)-nb-fee(nb)
    d['sideY']=(ey>=g)&(ya>=lo)&(ya<=hi)&('Y' in sides); d['sideN']=(en>=g)&(nb>=lo)&(nb<=hi)&('N' in sides)
    out=[]
    for sd,c in (('Y','sideY'),('N','sideN')):
        x=d[d[c]].sort_values('secs_left',ascending=False).drop_duplicates('ticker')
        if sd=='Y': x=x.assign(price=x.yes_ask,win=x.res)
        else: x=x.assign(price=1-x.yes_bid,win=1-x.res)
        x=x.assign(side=sd); out.append(x[['ticker','month','side','price','win','t','per']])
    return pd.concat(out) if out else pd.DataFrame()
def summ(x):
    if len(x)==0: return None
    f=fee(x.price); ev=x.win-x.price-f; mo=ev.groupby(x.month).sum()
    return dict(n=len(x),avg_price_c=round(x.price.mean()*100,1),win_pct=round(x.win.mean()*100,1),breakeven_pct=round((x.price+f).mean()*100,1),edge_pts=round(ev.mean()*100,2),ev_c=round(ev.mean()*100,2),se=round(ev.std()*100/np.sqrt(len(x)),2),months_pos=f'{int((mo>0).sum())}/{len(mo)}')
T=[]
for m in ['kalshi','direct','corr','lasso']:
    if m=='kalshi': continue
    for g in (0.03,0.05,0.10):
        for lab,(lo,hi) in {'all':(0,1),'fav70-99':(0.70,0.99)}.items():
            x=trades(S,m,g,lo,hi)
            for per,xs in [('Mar-Aug',x[x.per=='Mar-Aug']),('Sep-Oct',x[x.per!='Mar-Aug']),('Oct26-31',x[x.per=='Oct26-31'])]:
                for sd in ('Y','N','both'):
                    xx=xs if sd=='both' else xs[xs.side==sd]
                    s=summ(xx)
                    if s: T.append(dict(model=m,gap_pts=int(g*100),price_filter=lab,period=per,side=sd,**s))
TT=pd.DataFrame(T); print(TT[(TT.side=='both')&(TT.price_filter=='all')].to_string())
with pd.ExcelWriter(f'work/{name}_ml_results.xlsx') as w:
    R.to_excel(w,sheet_name='Brier AUC',index=False); BM.to_excel(w,sheet_name='Brier by min left',index=False); TT.to_excel(w,sheet_name='Trades',index=False)
