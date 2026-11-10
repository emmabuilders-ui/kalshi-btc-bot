import pandas as pd, numpy as np
from scipy.stats import norm, kendalltau
from sklearn.metrics import roc_auc_score
pd.set_option('display.width',250); pd.set_option('display.max_rows',600)
T0=1764460800; B=pd.Series(np.load('/home/user/kalshi-btc-bot/research/ml_replay/work/brti.npy')).ffill().to_numpy(); LB=np.log(B)
A=pd.read_parquet('emp_rows.parquet')   # longshot rows with p_emp, p_norm
e=(A.snap_ts.to_numpy()-T0).astype(np.int64); h=(A.secs_left.to_numpy()-40).clip(1).astype(np.int64); gap=A.gap.to_numpy()
def pct(W):
    out=np.empty(len(A))
    for i in range(len(A)):
        t=np.arange(e[i]-W+1, e[i]+1); mv=np.abs(LB[t]-LB[t-h[i]])
        out[i]=(mv>=gap[i]).mean()
    return out
A['exceed60']=pct(3600); A['exceed120']=pct(7200)
A['p_rank60']=A.exceed60/2; A['p_rank120']=A.exceed120/2    # one-direction chance
A['edge_rank60']=A.p_rank60-A.price-A.fee; A['edge_rank120']=A.p_rank120-A.price-A.fee
def summ(g):
    n=len(g); p=g.price.mean()*100; w=g.won.mean()*100; ev=g.pnl.mean()*100; se=g.pnl.std()*100/np.sqrt(max(n,1))
    return pd.Series({'trades':n,'avg_price_c':round(p,2),'win_pct':round(w,2),'breakeven_pct':round((g.price+g.fee).mean()*100,2),'win_minus_price':round(w-p,2),'ev_c':round(ev,2),'se_c':round(se,2),'t':round(ev/se if se>0 else np.nan,2)})
def fe(x): return x.sort_values('min_left',ascending=False).groupby(['ticker','side']).head(1)
DA=A[A.per=='DecAug']
print('AUC vs win (Dec-Aug, all longshot rows) — higher is better predictor of a win:')
for c in ['p_norm','p_emp','p_rank60','p_rank120']: print(c, round(roc_auc_score(DA.won,DA[c]),4), 'tau', round(kendalltau(DA[c],DA.won)[0],4))
print('also vs the SD-based z (negated):', round(roc_auc_score(DA.won,-DA.zgap),4))
# calibration of rank fair value, Sep-Oct
SO=A[A.per=='SepOct'].copy()
for c in ['p_rank60','p_rank120']:
    SO['fb']=pd.cut(SO[c],[-0.001,0.03,0.06,0.1,0.15,0.2,0.3,1]); print(c,'Sep-Oct calibration'); print(SO.groupby('fb',observed=True).agg(n=('won','size'),pred=(c,'mean'),actual=('won','mean')).round(3).to_string())
rows=[]
for col,name in [('edge_rank60','rank60'),('edge_rank120','rank120')]:
    for m in [0,0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.10]:
        s=fe(A[A[col]>=m]).groupby('per').apply(summ)
        for per in s.index: rows.append(dict(rule=name,margin_c=m*100,per=per,**s.loc[per].to_dict()))
M=pd.DataFrame(rows); print(M.set_index(['rule','margin_c','per']).unstack('per').to_string()); M.to_csv('rank_margins.csv',index=False)
# percentile shape: win% by exceed60 bins (both periods), 1-6 min
A['xb']=pd.cut(A.exceed60,[-0.001,0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.6,1.0]).astype(str)
A['pg']=pd.cut(A.price*100,[0,5,10,15],labels=['1-5c','6-10c','11-15c']).astype(str)
sh=A[A.min_left<=6].groupby(['per','pg','xb']).apply(summ).reset_index(); print(sh.set_index(['pg','xb','per']).unstack('per')[['trades','avg_price_c','win_pct','ev_c']].to_string()); sh.to_csv('rank_shape.csv',index=False)
A.to_parquet('rank_rows.parquet')
