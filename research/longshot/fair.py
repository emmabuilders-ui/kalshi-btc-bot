import pandas as pd, numpy as np
from scipy.stats import norm
pd.set_option('display.width',250); pd.set_option('display.max_rows',600)
def summ(g):
    n=len(g); p=g.price.mean()*100; w=g.won.mean()*100; ev=g.pnl.mean()*100; se=g.pnl.std()*100/np.sqrt(max(n,1))
    return pd.Series({'n':n,'price':round(p,2),'win%':round(w,2),'be%':round((g.price+g.fee).mean()*100,2),'w-p':round(w-p,2),'ev':round(ev,2),'se':round(se,2),'t':round(ev/se if se>0 else np.nan,2)})
def fe(x): return x.sort_values('min_left',ascending=False).groupby(['ticker','side']).head(1)
# ---- 15-min
R=pd.read_parquet('m15_long.parquet')
R['p_fair']=norm.cdf(-R.zgap)        # sig15 vol; chance BTC ends past strike
R['p_fair60']=norm.cdf(-R.zgap60)
R['edge']=R.p_fair-R.price-R.fee; R['edge60']=R.p_fair60-R.price-R.fee
print('=== 15-MIN longshots <=15c, buy when fair(normal, 15-min vol) - price - fee >= margin; first entry per contract/side')
rows=[]
for m in [0,0.01,0.02,0.03,0.05,0.08]:
    for lg,(a,b) in {'1-6m':(1,6),'7-14m':(7,14),'all':(1,14)}.items():
        c=R[(R.edge>=m)&(R.min_left>=a)&(R.min_left<=b)]
        s=fe(c).groupby('per').apply(summ)
        for per in ['DecAug','SepOct']:
            if per in s.index: rows.append(dict(margin=m,left=lg,per=per,**s.loc[per].to_dict()))
T=pd.DataFrame(rows).set_index(['margin','left','per']).unstack('per'); print(T.to_string())
# calibration of fair model on longshot rows: win% by p_fair bin
R['fb']=pd.cut(R.p_fair,[0,0.02,0.05,0.1,0.15,0.2,0.3,0.5,1])
print('calibration: actual win% vs fair bin (Dec-Aug / Sep-Oct), longshot rows'); print(R.groupby(['per','fb'],observed=True).agg(n=('won','size'),price=('price','mean'),fair=('p_fair','mean'),win=('won','mean')).round(3).unstack('per').to_string())
# sig60 variant
rows=[]
for m in [0,0.02,0.05]:
    c=R[(R.edge60>=m)]; s=fe(c).groupby('per').apply(summ)
    for per in s.index: rows.append(dict(margin=m,per=per,**s.loc[per].to_dict()))
print('sig60 fair'); print(pd.DataFrame(rows).set_index(['margin','per']).unstack('per').to_string())
# ---- hourly bands
d=pd.read_parquet('hb_rows.parquet'); d=d[d.spread<=0.10]
Teff=(d.left*60-40).clip(lower=1); s=d.brti*d.sig15.clip(lower=1e-7)*np.sqrt(Teff)
d['p_in']=norm.cdf((d.cap_strike-d.brti)/s)-norm.cdf((d.floor_strike-d.brti)/s)
Y=d[d.yes_ask<=0.15].copy(); Y['side']='YES'; Y['price']=Y.yes_ask; Y['won']=Y.won_yes; Y['p_fair']=Y.p_in
N=d[(1-d.yes_bid)<=0.15].copy(); N['side']='NO'; N['price']=1-N.yes_bid; N['won']=1-N.won_yes; N['p_fair']=1-N.p_in
H=pd.concat([Y,N]); H=H[H.price>0]; H['fee']=0.07*H.price*(1-H.price); H['pnl']=H.won-H.price-H.fee; H['edge']=H.p_fair-H.price-H.fee
def feh(x): return x.sort_values('left',ascending=False).groupby(['ticker','side']).head(1)
print('=== HOURLY BANDS longshots <=15c, fair = normal band probability (15-min vol), first entry per band/side')
rows=[]
for m in [0,0.01,0.02,0.03,0.05,0.08,0.12]:
    for lg,(a,b) in {'1-10m':(1,10),'11-30m':(11,30),'all':(1,30)}.items():
        c=H[(H.edge>=m)&(H.left>=a)&(H.left<=b)]
        s=feh(c).groupby('per').apply(summ)
        for per in s.index: rows.append(dict(margin=m,left=lg,per=per,**s.loc[per].to_dict()))
print(pd.DataFrame(rows).set_index(['margin','left','per']).unstack('per').to_string())
H['fb']=pd.cut(H.p_fair,[0,0.02,0.05,0.1,0.15,0.2,0.3,0.5,1])
print(H.groupby(['per','side','fb'],observed=True).agg(n=('won','size'),price=('price','mean'),fair=('p_fair','mean'),win=('won','mean')).round(3).unstack('per').to_string())
R.to_parquet('m15_long.parquet'); H.to_parquet('hb_long.parquet')
