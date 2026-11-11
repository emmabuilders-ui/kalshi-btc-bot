import numpy as np, pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score
def fee(p): return 0.07*p*(1-p)
S=pd.read_parquet('work/bands_scored.parquet')
S=S[S.p_corr.notna()].copy(); S['per']=np.where(S.t>=pd.Timestamp('2026-09-01',tz='UTC').timestamp(),'Sep-Oct','Mar-Aug')
S['gap']=S.p_corr-S.yes_ask-fee(S.yes_ask)
x=S[S.gap>=0.05].sort_values('secs_left',ascending=False).drop_duplicates('ticker').copy()
x['price']=x.yes_ask; x['win']=x.res; x['ev']=x.win-x.price-fee(x.price)
x['et_hour']=pd.to_datetime(x.t,unit='s',utc=True).dt.tz_convert('America/New_York').dt.hour
x['vol_state']=x.groupby('per').vol_900.transform(lambda s:pd.qcut(s,3,labels=['low','mid','high']))
x['place']=np.where(x.inside==1,'spot inside band','spot outside band')
x['dir']=np.where(x.dist_lo<0,'spot below band',np.where(x.dist_hi>0,'spot above band','inside'))
x['gapb']=pd.cut(x.gap*100,[5,7,10,15,100],right=False,labels=['5-7','7-10','10-15','15+'])
x['priceb']=pd.cut(x.price*100,[0,20,35,50,65,80,100],labels=['<20','20-35','35-50','50-65','65-80','80+'])
x['mlb']=pd.cut(x.min_left,[0,5,10,15,20,25,30],labels=['1-5','6-10','11-15','16-20','21-25','26-30'])
x['bandpos']=pd.cut(x.z_edge,[0,0.5,1,1.5,2,100],labels=['<0.5','0.5-1','1-1.5','1.5-2','2+'])
def tab(col):
    rows=[]
    for per in ('Mar-Aug','Sep-Oct'):
        for k,g in x[x.per==per].groupby(col,observed=True):
            rows.append(dict(feature=col,bucket=k,period=per,n=len(g),avg_price_c=round(g.price.mean()*100,1),win_pct=round(g.win.mean()*100,1),edge_c=round(g.ev.mean()*100,2),se=round(g.ev.std()*100/np.sqrt(len(g)),2)))
    return rows
R=pd.DataFrame(sum([tab(c) for c in ['gapb','priceb','mlb','place','dir','vol_state','bandpos','et_hour']],[]))
print('trades',x.groupby('per').size().to_dict())
for c in ['gapb','priceb','mlb','place','dir','vol_state','bandpos']:
    t=R[R.feature==c].pivot(index='bucket',columns='period',values=['n','avg_price_c','win_pct','edge_c'])
    print('\n',c); print(t.to_string())
print('\nET hour edge'); print(R[R.feature=='et_hour'].pivot(index='bucket',columns='period',values=['n','edge_c']).to_string())
# tau: edge vs features
for f in ['gap','price','min_left','z_edge','vol_900','p_sa','spread']:
    out=[]
    for per in ('Mar-Aug','Sep-Oct'):
        g=x[x.per==per]; tau=kendalltau(g[f],g.ev)[0]; auc=roc_auc_score(g.win,g[f]) if g.win.nunique()>1 else np.nan
        out.append(f'{per} tau {tau:+.3f} AUC {auc:.3f}')
    print(f,'|'.join(out))
R.to_excel('work/band_yes_trade_anatomy.xlsx',index=False)
x[['ticker','t','min_left','yes_ask','p_corr','mid','gap','res','ev','per','z_edge','vol_900','inside','dist_lo','dist_hi']].to_csv('work/band_yes_trades.csv',index=False)
